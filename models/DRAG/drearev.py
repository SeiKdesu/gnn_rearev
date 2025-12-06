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

        self.fact_selector = FactSelectorHead(in_dim=3 * self.entity_dim, hidden_dim=selector_hidden)
        self.prompt_builder = NeuralFactPromptBuilder(
            entity_dim=self.entity_dim,
            prompt_dim=prompt_dim,
            id2entity=id2entity,
            id2relation=id2relation,
        )
        self.generator: Optional[Callable[..., torch.Tensor]] = None

    def set_generator(self, generator_fn: Callable[..., torch.Tensor]):
        """
        Plug in a generator loss function that accepts (prompt_text, structural_emb, selections)
        and returns a scalar loss.
        """
        self.generator = generator_fn

    def forward(self, batch, training=False):
        # Preserve ReaRev behaviour for compatibility with existing trainer/evaluator.
        return super().forward(batch, training=training)

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
        """
        retriever_loss, pred, pred_dist, tp_list = super().forward(batch, training=training)

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
        )

        generator_loss = None
        gen_fn = generator_loss_fn or self.generator
        if gen_fn is not None:
            generator_loss = gen_fn(prompt, struct_emb, selections)

        total_loss = retriever_loss
        if generator_loss is not None:
            total_loss = total_loss + self.lambda_gen * generator_loss

        return {
            "loss": total_loss,
            "retriever_loss": retriever_loss,
            "generator_loss": generator_loss,
            "pred": pred,
            "pred_dist": pred_dist,
            "tp_list": tp_list,
            "selection": selection_output,
        }
