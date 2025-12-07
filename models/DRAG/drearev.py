import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from models.ReaRev.rearev import ReaRev, FactSelectorHead, GraphRepresentation


def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True, eps: float = 1e-10):
    """
    Binary straight-through Gumbel-Softmax used for differentiable fact selection.
    """
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y_soft = torch.sigmoid((logits + g) / temperature)
    if not hard:
        return y_soft
    y_hard = (y_soft > 0.5).float()
    return y_hard.detach() - y_soft.detach() + y_soft


@dataclass
class FactSelectionOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    selections: torch.Tensor
    fact_emb: torch.Tensor
    structural_emb: torch.Tensor
    semantic_texts: List[str]
    prompt: str
    batch_ids: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, torch.Tensor]] = None


class NeuralFactPromptBuilder(nn.Module):
    """
    Builds structural embeddings for facts and assembles text prompts for the generator.
    """
    def __init__(self, entity_dim: int, prompt_dim: int, id2entity: Dict[int, str], id2relation: Dict[int, str]):
        super().__init__()
        self.struct_proj = nn.Linear(3 * entity_dim, prompt_dim)
        self.id2entity = id2entity
        self.id2relation = id2relation

    def _semantic_strings(self, graph_repr: GraphRepresentation) -> List[str]:
        """
        Convert fact indices to human-readable triples using stored ID mappings.
        """
        max_local_entity = graph_repr.meta["max_local_entity"]
        local_entity_ids = graph_repr.meta["local_entity_ids"]
        heads = graph_repr.meta["head_index"]
        tails = graph_repr.meta["tail_index"]
        rels = graph_repr.meta["relation_index"]
        batch_ids = graph_repr.meta["batch_ids"]
        semantic = []
        for h_idx, r_idx, t_idx, b_idx in zip(heads, rels, tails, batch_ids):
            b = int(b_idx.item())
            h_local = int(h_idx.item()) - b * max_local_entity
            t_local = int(t_idx.item()) - b * max_local_entity
            # guard against malformed indices
            h_local = max(min(h_local, local_entity_ids.size(1) - 1), 0)
            t_local = max(min(t_local, local_entity_ids.size(1) - 1), 0)
            h_global = int(local_entity_ids[b, h_local].item())
            t_global = int(local_entity_ids[b, t_local].item())
            h_name = self.id2entity.get(h_global, str(h_global))
            t_name = self.id2entity.get(t_global, str(t_global))
            r_name = self.id2relation.get(int(r_idx.item()), str(int(r_idx.item())))
            semantic.append(f"{h_name}, {r_name}, {t_name}")
        return semantic

    @staticmethod
    def _format_structural_vec(vec: torch.Tensor, keep: int = 6) -> str:
        vals = vec.detach().cpu().tolist()[:keep]
        return " ".join([f"{v:.4f}" for v in vals])

    def forward(
        self,
        graph_repr: GraphRepresentation,
        selections: torch.Tensor,
        question_text: Optional[str] = None,
    ):
        """
        Returns (structural_embeddings, semantic_texts, prompt_text).
        """
        struct_emb = self.struct_proj(graph_repr.fact_emb)
        semantic_texts = self._semantic_strings(graph_repr)
        question_line = question_text if question_text is not None else "[UNK question]"
        lines = [
            "Answer the question based on the provided facts.",
            f"Question: {question_line}",
            "Provided facts:",
        ]
        for sel, s_emb, sem in zip(selections.detach().cpu(), struct_emb, semantic_texts):
            lines.append(f"{sel.item():.4f}\t{self._format_structural_vec(s_emb)}\t{sem}")
        lines.append("Answer:")
        prompt = "\n".join(lines)
        return struct_emb, semantic_texts, prompt


class DReaRev(ReaRev):
    """
    D-RAG wrapper over ReaRev that adds differentiable fact selection and prompt construction.
    """
    def __init__(
        self,
        args,
        num_entity: int,
        num_relation: int,
        num_word: int,
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
    ):
        super().__init__(args, num_entity, num_relation, num_word)
        selector_hidden = args.get("selector_hidden_dim", 256)
        prompt_dim = args.get("drag_prompt_dim", self.entity_dim)
        self.drag_temperature = args.get("drag_temperature", 1.0)
        self.lambda_gen = args.get("lambda_gen", 1.0)
        self.lambda_sel = args.get("lambda_sel", 0.0)
        self.lambda_sparse = args.get("selector_sparsity_weight", 0.0)
        self.selector_target_ratio = args.get("selector_sparsity_target", None)
        self.lambda_entropy = args.get("selector_entropy_weight", 0.0)

        self.fact_selector = FactSelectorHead(in_dim=3 * self.entity_dim, hidden_dim=selector_hidden)
        self.prompt_builder = NeuralFactPromptBuilder(
            entity_dim=self.entity_dim,
            prompt_dim=prompt_dim,
            id2entity=id2entity,
            id2relation=id2relation,
        )
        self.generator: Optional[Callable[..., torch.Tensor]] = None
        self.bce_selector = nn.BCEWithLogitsLoss()
        self.last_selection_metrics: Optional[Dict[str, float]] = None

    def set_generator(self, generator_fn: Callable[..., torch.Tensor]):
        """
        Plug in a generator loss function that accepts (prompt_text, structural_emb, selections)
        and returns a scalar loss.
        """
        self.generator = generator_fn

    def forward(self, batch, training=False):
        """
        Override forward to run D-RAG pipeline (fact selection + gated reasoning).
        Returns same tuple interface as ReaRev: loss, pred, pred_dist, tp_list.
        """
        out = self.forward_drag(batch, training=training)
        return out["loss"], out["pred"], out["pred_dist"], out["tp_list"]

    def forward_drag(
        self,
        batch,
        question_text: Optional[str] = None,
        gumbel_hard: bool = True,
        generator_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        """
        Joint retriever-generator forward pass.
        - retriever: ReaRev forward
        - selector: differentiable fact sampling via Gumbel-Softmax
        - prompt: neural fact prompt assembly
        - generator: optional callable to compute generator loss
        - subgraph: optional selector regularizers (coverage, sparsity, entropy) to learn compact answer-focused subgraphs
        """
        # 1) Run vanilla ReaRev to build graph representations (grad-tracked).
        retriever_loss_base, _, _, _ = super().forward(batch, training=training)

        graph_repr = self.get_graph_representations(use_final_entity_state=True)
        logits, probs = self.fact_selector(graph_repr.fact_emb)
        selections = gumbel_sigmoid(logits, temperature=self.drag_temperature, hard=gumbel_hard) if training else probs

        struct_emb, semantic_texts, prompt = self.prompt_builder(
            graph_repr=graph_repr, selections=selections, question_text=question_text
        )

        selection_output = FactSelectionOutput(
            logits=logits,
            probs=probs,
            selections=selections,
            fact_emb=graph_repr.fact_emb,
            structural_emb=struct_emb,
            semantic_texts=semantic_texts,
            prompt=prompt,
            batch_ids=graph_repr.meta.get("batch_ids", None),
            meta=graph_repr.meta,
        )

        # 3) Optional weak supervision for selector using answer entities.
        selection_labels = self._label_answer_facts(graph_repr, batch[-1])
        selection_loss = None
        lambda_sel = self.lambda_sel
        if lambda_sel > 0:
            selection_loss = self.bce_selector(logits, selection_labels)

        # Encourage compact subgraphs while keeping gradients smooth.
        sparsity_loss = None
        if self.lambda_sparse > 0:
            mean_sel = selections.mean()
            if self.selector_target_ratio is None:
                sparsity_loss = mean_sel
            else:
                target = torch.tensor(self.selector_target_ratio, device=selections.device, dtype=selections.dtype)
                sparsity_loss = (mean_sel - target).abs()

        entropy_loss = None
        if self.lambda_entropy > 0:
            probs_for_entropy = probs.clamp(min=1e-6, max=1 - 1e-6)
            entropy_loss = -(
                probs_for_entropy * torch.log(probs_for_entropy) +
                (1 - probs_for_entropy) * torch.log(1 - probs_for_entropy)
            ).mean()

        # 4) Run gated reasoning pass using sampled selections.
        retriever_loss_gated, pred, pred_dist, tp_list = super().forward(batch, training=training, fact_gate=selections)

        generator_loss = None
        gen_fn = generator_loss_fn or self.generator
        if gen_fn is not None:
            generator_loss = gen_fn(prompt, struct_emb, selections)

        total_loss = retriever_loss_gated
        if selection_loss is not None:
            total_loss = total_loss + lambda_sel * selection_loss
        if sparsity_loss is not None:
            total_loss = total_loss + self.lambda_sparse * sparsity_loss
        if entropy_loss is not None:
            total_loss = total_loss + self.lambda_entropy * entropy_loss
        if generator_loss is not None:
            total_loss = total_loss + self.lambda_gen * generator_loss

        return {
            "loss": total_loss,
            "retriever_loss": retriever_loss_gated,
            "generator_loss": generator_loss,
            "selection_loss": selection_loss,
            "sparsity_loss": sparsity_loss,
            "entropy_loss": entropy_loss,
            "pred": pred,
            "pred_dist": pred_dist,
            "tp_list": tp_list,
            "selection": selection_output,
            "selection_metrics": self._compute_selection_metrics(
                selection_output=selection_output,
                labels=selection_labels,
            ),
        }

    def _compute_selection_metrics(
        self,
        selection_output: FactSelectionOutput,
        labels: Optional[torch.Tensor],
    ) -> Optional[Dict[str, float]]:
        """
        Computes simple precision/recall/F1 for selected facts vs. weak labels.
        Used for subgraph quality monitoring; returns None if labels not provided.
        """
        if labels is None:
            self.last_selection_metrics = None
            return None
        preds = (selection_output.selections > 0.5).float()
        labels = labels.float()
        tp = torch.sum(preds * labels)
        fp = torch.sum(preds) - tp
        fn = torch.sum(labels) - tp
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else torch.tensor(0.0, device=preds.device)
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else torch.tensor(0.0, device=preds.device)
        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
            if (precision + recall) > 0
            else torch.tensor(0.0, device=preds.device)
        )
        coverage = torch.sum(selection_output.probs * labels) / (torch.sum(labels) + 1e-8)
        sparsity = torch.mean(selection_output.selections)
        metrics = {
            "subgraph_precision": precision.detach().cpu().item(),
            "subgraph_recall": recall.detach().cpu().item(),
            "subgraph_f1": f1.detach().cpu().item(),
            "subgraph_coverage": coverage.detach().cpu().item(),
            "subgraph_sparsity": sparsity.detach().cpu().item(),
            "num_facts": selection_output.selections.numel(),
            "num_labeled_pos": torch.sum(labels).detach().cpu().item(),
            "num_selected": torch.sum(preds).detach().cpu().item(),
        }
        self.last_selection_metrics = metrics
        return metrics

    def _label_answer_facts(self, graph_repr: GraphRepresentation, answer_dist_np: Any) -> torch.Tensor:
        """
        Builds weak labels for facts: 1 if head or tail is an answer entity in the local graph.
        """
        if not torch.is_tensor(answer_dist_np):
            answer_dist = torch.from_numpy(answer_dist_np).to(graph_repr.fact_emb.device)
        else:
            answer_dist = answer_dist_np.to(graph_repr.fact_emb.device)
        max_local = graph_repr.meta["max_local_entity"]
        batch_ids = graph_repr.meta["batch_ids"]
        heads = graph_repr.meta["head_index"]
        tails = graph_repr.meta["tail_index"]

        head_local = heads - batch_ids * max_local
        tail_local = tails - batch_ids * max_local
        head_local = torch.clamp(head_local, min=0, max=answer_dist.size(1) - 1)
        tail_local = torch.clamp(tail_local, min=0, max=answer_dist.size(1) - 1)
        head_is_ans = answer_dist[batch_ids, head_local]
        tail_is_ans = answer_dist[batch_ids, tail_local]
        labels = ((head_is_ans + tail_is_ans) > 0).float()
        return labels
