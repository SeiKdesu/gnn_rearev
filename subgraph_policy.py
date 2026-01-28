"""
Policy model + expansion helper for Policy-based Subgraph Expansion (PSE).

This module is intentionally independent of the KGQA model stack (ReaRev/NSM/GraftNet).

Typical workflow:
  1) `python main.py --mode train_policy --data_folder data/CWQ/ --lm lstm`
  2) Enable at KGQA training/eval time with:
       `--enable_policy_expand true --union_graph_cache ... --policy_ckpt ...`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None


def _stable_hash_ids(ids: torch.Tensor, num_buckets: int) -> torch.Tensor:
    ids = ids.to(torch.int64)
    # Knuth multiplicative hash; deterministic across runs.
    return torch.remainder(ids * 2654435761, int(num_buckets)).to(torch.long)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom.squeeze(dim)


def _lm_name_from_args(lm: str) -> str:
    if lm == "bert":
        return "bert-base-uncased"
    if lm == "roberta":
        return "roberta-base"
    if lm == "sbert":
        return "sentence-transformers/all-MiniLM-L6-v2"
    if lm == "sbert2":
        return "sentence-transformers/all-mpnet-base-v2"
    if lm == "simcse":
        return "princeton-nlp/sup-simcse-bert-base-uncased"
    if lm == "t5":
        return "t5-small"
    if lm == "relbert":
        return "pretrained_lms/sr-simbert/"
    raise ValueError(f"Unsupported lm for policy: {lm!r}")


@dataclass
class PolicyConfig:
    lm: str = "lstm"
    q_max_len: int = 64
    entity_buckets: int = 50000
    emb_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    use_inverse_relation: bool = False
    num_base_rel: int = 0
    lm_frozen: bool = True
    device: str = "cpu"
    use_message_passing: bool = False
    mp_rounds: int = 2

    @property
    def num_rel(self) -> int:
        return (2 * self.num_base_rel) if self.use_inverse_relation else self.num_base_rel


class SubgraphPolicyGNN(nn.Module):
    def __init__(
        self,
        cfg: PolicyConfig,
        word2id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.word2id = word2id or {}

        self.ent_emb = nn.Embedding(cfg.entity_buckets, cfg.emb_dim)
        self.rel_emb = nn.Embedding(cfg.num_rel, cfg.emb_dim)

        self.q_proj = nn.Linear(self._question_hidden_dim(), cfg.emb_dim)

        in_dim = 3 * cfg.emb_dim + cfg.emb_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

        if cfg.use_message_passing:
            self.edge_msg = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            )
            self.node_upd = nn.Sequential(
                nn.Linear(3 * cfg.emb_dim + 1 + cfg.emb_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            )

        self._init_question_encoder()

    def _question_hidden_dim(self) -> int:
        if self.cfg.lm == "lstm":
            return self.cfg.emb_dim
        # Match common transformer hidden sizes.
        if self.cfg.lm == "sbert":
            return 384
        return 768

    def _init_question_encoder(self) -> None:
        if self.cfg.lm == "lstm":
            vocab_size = (max(self.word2id.values()) + 2) if self.word2id else 50000
            self.word_emb = nn.Embedding(vocab_size, self.cfg.emb_dim)
            self.tokenizer = None
            self.lm_model = None
            return

        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError(
                "transformers is not available, but policy lm != 'lstm'. "
                "Install transformers or use `--lm lstm` for policy training."
            )
        name = _lm_name_from_args(self.cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.lm_model = AutoModel.from_pretrained(name)
        if self.cfg.lm_frozen:
            for p in self.lm_model.parameters():
                p.requires_grad = False

    def encode_question(self, questions: Sequence[str]) -> torch.Tensor:
        """
        Returns q_emb: [B, emb_dim]
        """
        if self.cfg.lm == "lstm":
            ids = []
            for q in questions:
                toks = q.strip().split()
                tok_ids = [self.word2id.get(w, len(self.word2id)) for w in toks][: self.cfg.q_max_len]
                if not tok_ids:
                    tok_ids = [len(self.word2id)]
                ids.append(tok_ids)
            max_len = max(len(x) for x in ids)
            pad_id = len(self.word2id)
            arr = torch.full((len(ids), max_len), pad_id, dtype=torch.long, device=self.cfg.device)
            mask = torch.zeros((len(ids), max_len), dtype=torch.float32, device=self.cfg.device)
            for i, seq in enumerate(ids):
                arr[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.cfg.device)
                mask[i, : len(seq)] = 1.0
            h = self.word_emb(arr)  # [B, T, D]
            q_h = _masked_mean(h, mask, dim=1)  # [B, D]
            return self.q_proj(q_h)

        tok = self.tokenizer(
            list(questions),
            padding=True,
            truncation=True,
            max_length=self.cfg.q_max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(self.cfg.device) for k, v in tok.items()}
        if self.cfg.lm == "t5":
            out = self.lm_model.encoder(**tok)
            h = out.last_hidden_state
        else:
            out = self.lm_model(**tok)
            h = out.last_hidden_state
        q_h = _masked_mean(h, tok["attention_mask"], dim=1)
        return self.q_proj(q_h)

    def _edge_logits(
        self,
        h_ids: torch.Tensor,
        r_ids: torch.Tensor,
        t_ids: torch.Tensor,
        q_emb: torch.Tensor,
    ) -> torch.Tensor:
        h_emb = self.ent_emb(_stable_hash_ids(h_ids, self.cfg.entity_buckets))
        t_emb = self.ent_emb(_stable_hash_ids(t_ids, self.cfg.entity_buckets))
        r_emb = self.rel_emb(r_ids.to(torch.long))
        x = torch.cat([h_emb, r_emb, t_emb, q_emb], dim=-1)
        return self.edge_mlp(x).squeeze(-1)

    def score_edges(
        self,
        edges: torch.Tensor,
        q_emb: torch.Tensor,
        seed_set: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """
        edges: int64 [E, 3] (h, r, t)
        q_emb: [1, emb_dim] or [E, emb_dim]
        Returns scores in (0,1): [E]
        """
        if edges.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.cfg.device)

        if q_emb.dim() == 2 and q_emb.size(0) == 1:
            q_emb = q_emb.expand(edges.size(0), -1)

        if not self.cfg.use_message_passing:
            logits = self._edge_logits(edges[:, 0], edges[:, 1], edges[:, 2], q_emb)
            return torch.sigmoid(logits)

        # Lightweight message passing on induced subgraph of candidate edges.
        nodes, inv = torch.unique(torch.cat([edges[:, 0], edges[:, 2]]), return_inverse=True)
        h_local = inv[: edges.size(0)]
        t_local = inv[edges.size(0) :]
        node_emb = self.ent_emb(_stable_hash_ids(nodes, self.cfg.entity_buckets))

        seed_indicator = torch.zeros((nodes.size(0), 1), dtype=torch.float32, device=self.cfg.device)
        if seed_set is not None:
            seed_set = set(int(x) for x in seed_set)
            if seed_set:
                # Small sets only (question entity seeds are usually small).
                seed_tensor = torch.tensor(sorted(seed_set), dtype=nodes.dtype, device=self.cfg.device)
                try:
                    is_seed = torch.isin(nodes, seed_tensor)
                except Exception:
                    is_seed = (nodes.unsqueeze(1) == seed_tensor.unsqueeze(0)).any(dim=1)
                seed_indicator[is_seed] = 1.0

        for _ in range(int(self.cfg.mp_rounds)):
            h_emb = node_emb[h_local]
            t_emb = node_emb[t_local]
            r_emb = self.rel_emb(edges[:, 1].to(torch.long))
            x = torch.cat([h_emb, r_emb, t_emb, q_emb], dim=-1)
            e_msg = self.edge_msg(x)  # [E, D]
            node_msg = torch.zeros_like(node_emb)
            node_msg.index_add_(0, h_local, e_msg)
            node_msg.index_add_(0, t_local, e_msg)
            node_emb = self.node_upd(torch.cat([node_emb, node_msg, seed_indicator, q_emb[:1].expand(nodes.size(0), -1)], dim=-1))

        # Score with updated node embeddings.
        h_emb = node_emb[h_local]
        t_emb = node_emb[t_local]
        r_emb = self.rel_emb(edges[:, 1].to(torch.long))
        x = torch.cat([h_emb, r_emb, t_emb, q_emb], dim=-1)
        logits = self.edge_mlp(x).squeeze(-1)
        return torch.sigmoid(logits)


def save_policy_ckpt(path: str, model: SubgraphPolicyGNN, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "cfg": model.cfg.__dict__,
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_policy_ckpt(
    path: str,
    word2id: Optional[Dict[str, int]],
    map_location: str = "cpu",
    override_device: Optional[str] = None,
) -> SubgraphPolicyGNN:
    payload = torch.load(path, map_location=map_location)
    cfg = PolicyConfig(**payload["cfg"])
    if override_device is not None:
        cfg.device = override_device
    model = SubgraphPolicyGNN(cfg, word2id=word2id)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(cfg.device)
    model.eval()
    return model


def expand_sample_subgraph(
    sample: Dict[str, Any],
    union_adj: Dict[int, np.ndarray],
    policy: SubgraphPolicyGNN,
    policy_expand_hops: int = 2,
    topk_per_node: int = 20,
    max_new_edges_per_hop: int = 200,
    max_new_edges_total: int = 800,
    frontier_mode: str = "new",
) -> Dict[str, Any]:
    """
    Inference-time subgraph expansion (NO answers used).

    Mutates and returns `sample`.
    """
    sub = sample.get("subgraph", {})
    tuples = sub.get("tuples", [])
    entities = sub.get("entities", [])

    # Ensure ints (datasets typically already store global ids).
    clean_tuples: List[List[int]] = []
    edge_set = set()
    for tpl in tuples:
        if not isinstance(tpl, (list, tuple)) or len(tpl) != 3:
            continue
        h, r, t = int(tpl[0]), int(tpl[1]), int(tpl[2])
        key = (h, r, t)
        if key in edge_set:
            continue
        edge_set.add(key)
        clean_tuples.append([h, r, t])

    ent_set = set(int(e) for e in entities) if entities else set()
    for h, _, t in edge_set:
        ent_set.add(int(h))
        ent_set.add(int(t))

    seed_key = "entities_cid" if "entities_cid" in sample else "entities"
    seeds = [int(x) for x in sample.get(seed_key, [])]
    seed_set = set(seeds)
    frontier = list(seed_set)

    if not frontier:
        sub["tuples"] = clean_tuples
        sub["entities"] = list(ent_set)
        sample["subgraph"] = sub
        return sample

    with torch.no_grad():
        q_emb = policy.encode_question([sample.get("question", "")])

    total_added = 0
    for _step in range(int(policy_expand_hops)):
        if total_added >= max_new_edges_total:
            break
        if not frontier:
            break

        per_head_candidates: Dict[int, List[Tuple[int, int, int]]] = {}
        for h in frontier:
            arr = union_adj.get(int(h))
            if arr is None or len(arr) == 0:
                continue
            cand = []
            for r, t in arr:
                key = (int(h), int(r), int(t))
                if key in edge_set:
                    continue
                cand.append(key)
            if cand:
                per_head_candidates[int(h)] = cand

        if not per_head_candidates:
            break

        selected: List[Tuple[float, Tuple[int, int, int]]] = []
        for h, cand in per_head_candidates.items():
            edges = torch.tensor(cand, dtype=torch.long, device=policy.cfg.device)
            scores = policy.score_edges(edges, q_emb, seed_set=seed_set).detach().cpu().numpy()
            if scores.size == 0:
                continue
            k = min(int(topk_per_node), scores.size)
            top_idx = np.argpartition(-scores, k - 1)[:k]
            for i in top_idx:
                selected.append((float(scores[i]), cand[int(i)]))

        if not selected:
            break

        # Deterministic tie-breaking for reproducibility.
        selected.sort(key=lambda x: (-x[0], x[1][0], x[1][1], x[1][2]))
        hop_cap = min(int(max_new_edges_per_hop), int(max_new_edges_total) - total_added)
        chosen = selected[:hop_cap]

        new_nodes = set()
        for _s, (h, r, t) in chosen:
            if (h, r, t) in edge_set:
                continue
            edge_set.add((h, r, t))
            clean_tuples.append([h, r, t])
            if h not in ent_set:
                ent_set.add(h)
                new_nodes.add(h)
            if t not in ent_set:
                ent_set.add(t)
                new_nodes.add(t)
            total_added += 1
            if total_added >= max_new_edges_total:
                break

        if frontier_mode == "all":
            frontier = list(set(frontier).union(new_nodes))
        else:
            frontier = list(new_nodes)

    sub["tuples"] = clean_tuples
    sub["entities"] = list(ent_set)
    sample["subgraph"] = sub
    return sample
