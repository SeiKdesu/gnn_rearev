"""
Train the PSE SubgraphPolicyGNN using train split ONLY (no dev/test leakage).

Recommended usage (no KGQA training involved):
  - Train policy + build union graph cache:
      `python3 main.py --mode train_policy --data_folder data/CWQ/ --lm lstm`

Key outputs:
  - Union graph cache: `<data_folder>/cache/union_graph.pkl` (train-only)
  - Policy checkpoint: `<data_folder>/cache/policy_ckpt.pt`
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from subgraph_union_graph import build_union_graph, load_union_graph
from subgraph_policy import PolicyConfig, SubgraphPolicyGNN, save_policy_ckpt


def _load_dict(path: str) -> Dict[str, int]:
    d: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            d[w] = len(d)
    return d


def _load_dict_int(path: str) -> Dict[int, int]:
    d: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            i = int(w)
            d[i] = i
    return d


def _get_train_json_path(data_folder: str) -> str:
    if not data_folder.endswith("/"):
        data_folder = data_folder + "/"
    return os.path.join(data_folder, "train.json")


def _answers_to_global_ids(sample: Dict[str, Any], entity2id: Dict[Any, int]) -> Set[int]:
    if "answers_cid" in sample and sample["answers_cid"] is not None:
        return set(int(x) for x in sample["answers_cid"])
    out: Set[int] = set()
    for ans in sample.get("answers", []) or []:
        if isinstance(ans, int):
            out.add(int(ans))
            continue
        if not isinstance(ans, dict):
            continue
        kb_id = ans.get("kb_id")
        key = "text" if isinstance(kb_id, int) else "kb_id"
        raw = ans.get(key) or ans.get("text") or ans.get("kb_id")
        if raw is None:
            continue
        if isinstance(raw, int):
            out.add(int(raw))
            continue
        if raw in entity2id:
            out.add(int(entity2id[raw]))
            continue
        try:
            out.add(int(raw))
        except Exception:
            continue
    return out


def _seeds_to_global_ids(sample: Dict[str, Any], entity2id: Dict[Any, int]) -> List[int]:
    key = "entities_cid" if "entities_cid" in sample else "entities"
    seeds = sample.get(key, []) or []
    out: List[int] = []
    for x in seeds:
        if isinstance(x, int):
            out.append(int(x))
        elif isinstance(x, dict) and "text" in x and x["text"] in entity2id:
            out.append(int(entity2id[x["text"]]))
        elif isinstance(x, str) and x in entity2id:
            out.append(int(entity2id[x]))
        else:
            try:
                out.append(int(x))
            except Exception:
                continue
    return out


def _bfs_shortest_path_edges(
    adj: Dict[int, np.ndarray],
    seeds: Sequence[int],
    answers: Set[int],
    hop_limit: int,
) -> Tuple[Set[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Multi-source BFS with parent tracking (shortest paths) up to hop_limit.
    Returns:
      - pos_edges: union of edges on any shortest path from any seed to any reachable answer
      - explored_edges: all edges traversed within hop_limit frontier
    """
    seeds_set = set(int(s) for s in seeds)
    if not seeds_set or not answers:
        return set(), []

    depth: Dict[int, int] = {s: 0 for s in seeds_set}
    parents: Dict[int, List[Tuple[int, int, int]]] = {}
    explored: List[Tuple[int, int, int]] = []

    frontier = list(seeds_set)
    for d in range(int(hop_limit)):
        if not frontier:
            break
        next_frontier: List[int] = []
        nd = d + 1
        for u in frontier:
            arr = adj.get(int(u))
            if arr is None:
                continue
            for r, t in arr:
                e = (int(u), int(r), int(t))
                explored.append(e)
                v = int(t)
                if v not in depth:
                    depth[v] = nd
                    parents[v] = [e]
                    next_frontier.append(v)
                elif depth[v] == nd:
                    parents[v].append(e)
        frontier = next_frontier

    reachable_answers = [a for a in answers if a in depth and depth[a] <= hop_limit]
    if not reachable_answers:
        return set(), explored

    pos: Set[Tuple[int, int, int]] = set()
    stack: List[int] = list(reachable_answers)
    seen: Set[int] = set(stack)
    while stack:
        v = stack.pop()
        if v in seeds_set:
            continue
        for e in parents.get(v, []):
            pos.add(e)
            u = e[0]
            if u not in seen and u not in seeds_set:
                seen.add(u)
                stack.append(u)
    return pos, explored


def train_policy_from_args(args: argparse.Namespace) -> None:
    data_folder = args.data_folder
    if not data_folder.endswith("/"):
        data_folder = data_folder + "/"

    # Dicts (global ids).
    entity_path = os.path.join(data_folder, args.entity2id)
    rel_path = os.path.join(data_folder, args.relation2id)
    word_path = os.path.join(data_folder, args.word2id)

    if "sr-cwq" in data_folder:
        entity2id = _load_dict_int(entity_path)
    else:
        entity2id = _load_dict(entity_path)
    relation2id = _load_dict(rel_path)
    word2id = _load_dict(word_path)

    train_json_path = _get_train_json_path(data_folder)
    union_cache = args.union_graph_cache or os.path.join(data_folder, "cache/union_graph.pkl")
    ckpt_path = args.policy_ckpt or os.path.join(data_folder, "cache/policy_ckpt.pt")
    os.makedirs(os.path.dirname(union_cache), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if args.rebuild_union_graph or not os.path.exists(union_cache):
        build_union_graph(
            train_json_path=train_json_path,
            entity2id=entity2id,
            relation2id=relation2id,
            use_inverse_relation=bool(args.use_inverse_relation),
            cache_path=union_cache,
        )
    union_adj, union_meta = load_union_graph(union_cache)
    if os.path.abspath(union_meta.get("train_json_path", "")) != os.path.abspath(train_json_path):
        print(f"[WARN] union_graph train path mismatch: cache={union_meta.get('train_json_path')} expected={train_json_path}")

    cfg = PolicyConfig(
        lm=args.lm,
        q_max_len=args.policy_max_q_len,
        entity_buckets=args.policy_entity_buckets,
        emb_dim=args.policy_emb_dim,
        hidden_dim=args.policy_hidden_dim,
        dropout=args.policy_dropout,
        use_inverse_relation=bool(args.use_inverse_relation),
        num_base_rel=len(relation2id),
        lm_frozen=bool(args.policy_lm_frozen),
        device=args.policy_device,
        use_message_passing=bool(args.policy_use_message_passing),
        mp_rounds=args.policy_mp_rounds,
    )
    model = SubgraphPolicyGNN(cfg, word2id=word2id).to(cfg.device)
    model.train()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.policy_lr)

    hop_limit = int(args.policy_hop_limit)
    neg_per_pos = int(args.policy_neg_per_pos)

    for epoch in range(int(args.policy_epochs)):
        total_loss = 0.0
        total_steps = 0
        pos_edges_total = 0
        samples_with_pos = 0
        seen_samples = 0

        with open(train_json_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Policy epoch {epoch+1}/{args.policy_epochs}"):
                if not line.strip():
                    continue
                sample = json.loads(line)
                seen_samples += 1
                if args.max_train_samples and seen_samples > int(args.max_train_samples):
                    break
                q = sample.get("question", "")
                seeds = _seeds_to_global_ids(sample, entity2id)
                answers = _answers_to_global_ids(sample, entity2id)

                pos_edges, explored = _bfs_shortest_path_edges(
                    adj=union_adj,
                    seeds=seeds,
                    answers=answers,
                    hop_limit=hop_limit,
                )
                if not pos_edges:
                    continue

                explored_set = set(explored)
                neg_pool = list(explored_set.difference(pos_edges))
                if not neg_pool:
                    continue

                pos_list = list(pos_edges)
                neg_target = min(len(neg_pool), max(1, neg_per_pos * len(pos_list)))
                rng.shuffle(neg_pool)
                neg_list = neg_pool[:neg_target]

                edges = torch.tensor(pos_list + neg_list, dtype=torch.long, device=cfg.device)
                labels = torch.tensor(
                    [1.0] * len(pos_list) + [0.0] * len(neg_list),
                    dtype=torch.float32,
                    device=cfg.device,
                )

                q_emb = model.encode_question([q])  # [1, D]
                scores = model.score_edges(edges, q_emb, seed_set=set(seeds)).clamp(1e-6, 1 - 1e-6)
                loss = F.binary_cross_entropy(scores, labels)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.policy_grad_clip)
                opt.step()

                total_loss += float(loss.detach().cpu().item())
                total_steps += 1
                pos_edges_total += len(pos_list)
                samples_with_pos += 1

        avg_loss = total_loss / max(1, total_steps)
        print(
            f"[Policy] epoch={epoch+1} loss={avg_loss:.4f} steps={total_steps} "
            f"samples_with_pos={samples_with_pos} avg_pos_edges={pos_edges_total / max(1, samples_with_pos):.2f}"
        )
        save_policy_ckpt(
            ckpt_path,
            model,
            extra={"epoch": epoch + 1, "union_graph_cache": union_cache, "union_meta": union_meta},
        )

    print(f"[Policy] saved checkpoint to {ckpt_path}")


def build_policy_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Accepted for compatibility with `python3 main.py --mode train_policy ...`.
    p.add_argument("--mode", default="train_policy", type=str)
    p.add_argument("--data_folder", type=str, required=True)
    p.add_argument("--word2id", default="vocab.txt", type=str)
    p.add_argument("--relation2id", default="relations.txt", type=str)
    p.add_argument("--entity2id", default="entities.txt", type=str)

    p.add_argument("--seed", default=19960626, type=int)
    p.add_argument("--lm", default="lstm", type=str)
    p.add_argument("--use_inverse_relation", action="store_true")

    p.add_argument("--union_graph_cache", default=None, type=str)
    p.add_argument("--policy_ckpt", default=None, type=str)
    p.add_argument("--rebuild_union_graph", action="store_true")

    p.add_argument("--policy_device", default="cpu", type=str)
    p.add_argument("--policy_epochs", default=3, type=int)
    p.add_argument("--policy_lr", default=1e-3, type=float)
    p.add_argument("--policy_grad_clip", default=1.0, type=float)

    p.add_argument("--policy_hop_limit", default=3, type=int)
    p.add_argument("--policy_neg_per_pos", default=5, type=int)

    p.add_argument("--policy_max_q_len", default=64, type=int)
    p.add_argument("--policy_entity_buckets", default=50000, type=int)
    p.add_argument("--policy_emb_dim", default=128, type=int)
    p.add_argument("--policy_hidden_dim", default=256, type=int)
    p.add_argument("--policy_dropout", default=0.1, type=float)
    p.add_argument("--policy_lm_frozen", default=1, type=int)

    p.add_argument("--policy_use_message_passing", action="store_true")
    p.add_argument("--policy_mp_rounds", default=2, type=int)
    p.add_argument("--max_train_samples", default=0, type=int)
    return p


if __name__ == "__main__":
    parser = build_policy_arg_parser()
    train_policy_from_args(parser.parse_args())
