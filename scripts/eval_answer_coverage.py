"""
Answer coverage evaluation for (optionally) Policy-based Subgraph Expansion (PSE).

Usage:
  - Baseline coverage on a split:
      `python3 scripts/eval_answer_coverage.py --data_folder data/CWQ/ --split dev`

  - Coverage after expansion (requires train-only union graph + policy ckpt):
      `python3 scripts/eval_answer_coverage.py --data_folder data/CWQ/ --split dev \\
         --enable_expand true --union_graph_cache data/CWQ/cache/union_graph.pkl \\
         --policy_ckpt data/CWQ/cache/policy_ckpt.pt`
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Set

import numpy as np

from subgraph_union_graph import load_union_graph
from subgraph_policy import expand_sample_subgraph, load_policy_ckpt


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


def _answers_to_ids(sample: Dict[str, Any], entity2id: Dict[Any, int]) -> Set[int]:
    if "answers_cid" in sample and sample["answers_cid"] is not None:
        return set(int(x) for x in sample["answers_cid"])
    out: Set[int] = set()
    for a in sample.get("answers", []) or []:
        if isinstance(a, int):
            out.add(int(a))
            continue
        if not isinstance(a, dict):
            continue
        key = "text" if isinstance(a.get("kb_id"), int) else "kb_id"
        raw = a.get(key) or a.get("kb_id") or a.get("text")
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


def _subgraph_entity_set(sample: Dict[str, Any]) -> Set[int]:
    ent = set(int(x) for x in sample.get("subgraph", {}).get("entities", []) or [])
    if ent:
        return ent
    for tpl in sample.get("subgraph", {}).get("tuples", []) or []:
        if isinstance(tpl, (list, tuple)) and len(tpl) == 3:
            ent.add(int(tpl[0]))
            ent.add(int(tpl[2]))
    return ent


def _compute_coverage(json_path: str, entity2id: Dict[Any, int]) -> float:
    hit = 0
    total = 0
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            total += 1
            ans = _answers_to_ids(s, entity2id)
            ent = _subgraph_entity_set(s)
            if ans and ent.intersection(ans):
                hit += 1
    return hit / max(1, total)


def _avg_sizes(json_path: str) -> tuple[float, float]:
    ents = []
    tpls = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            ents.append(len(s.get("subgraph", {}).get("entities", []) or []))
            tpls.append(len(s.get("subgraph", {}).get("tuples", []) or []))
    return float(np.mean(ents)) if ents else 0.0, float(np.mean(tpls)) if tpls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True, type=str)
    ap.add_argument("--split", required=True, type=str, choices=["train", "dev", "test"])
    ap.add_argument("--entity2id", default="entities.txt", type=str)

    ap.add_argument("--enable_expand", default=False, type=lambda x: str(x).lower() in ("1", "true", "t", "yes", "y"))
    ap.add_argument("--union_graph_cache", default=None, type=str)
    ap.add_argument("--policy_ckpt", default=None, type=str)
    ap.add_argument("--policy_device", default="cpu", type=str)

    ap.add_argument("--policy_expand_hops", default=2, type=int)
    ap.add_argument("--policy_topk_per_node", default=20, type=int)
    ap.add_argument("--policy_max_new_edges_per_hop", default=200, type=int)
    ap.add_argument("--policy_max_new_edges_total", default=800, type=int)
    ap.add_argument("--policy_frontier_mode", default="new", type=str, choices=["new", "all"])
    args = ap.parse_args()

    data_folder = args.data_folder
    if not data_folder.endswith("/"):
        data_folder = data_folder + "/"
    json_path = os.path.join(data_folder, f"{args.split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    entity_path = os.path.join(data_folder, args.entity2id)
    entity2id = _load_dict_int(entity_path) if "sr-cwq" in data_folder else _load_dict(entity_path)

    base_cov = _compute_coverage(json_path, entity2id)
    base_avg_ent, base_avg_tpl = _avg_sizes(json_path)
    print(f"[Base] split={args.split} avg_entities={base_avg_ent:.2f} avg_tuples={base_avg_tpl:.2f} coverage={base_cov*100:.2f}%")

    if not args.enable_expand:
        return

    union_cache = args.union_graph_cache or os.path.join(data_folder, "cache/union_graph.pkl")
    policy_ckpt = args.policy_ckpt or os.path.join(data_folder, "cache/policy_ckpt.pt")
    union_adj, _ = load_union_graph(union_cache)
    policy = load_policy_ckpt(policy_ckpt, word2id=None, map_location=args.policy_device, override_device=args.policy_device)

    hit = 0
    total = 0
    ents = []
    tpls = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            s = expand_sample_subgraph(
                s,
                union_adj=union_adj,
                policy=policy,
                policy_expand_hops=args.policy_expand_hops,
                topk_per_node=args.policy_topk_per_node,
                max_new_edges_per_hop=args.policy_max_new_edges_per_hop,
                max_new_edges_total=args.policy_max_new_edges_total,
                frontier_mode=args.policy_frontier_mode,
            )
            total += 1
            ans = _answers_to_ids(s, entity2id)
            ent = _subgraph_entity_set(s)
            if ans and ent.intersection(ans):
                hit += 1
            ents.append(len(s.get("subgraph", {}).get("entities", []) or []))
            tpls.append(len(s.get("subgraph", {}).get("tuples", []) or []))

    cov = hit / max(1, total)
    avg_ent = float(np.mean(ents)) if ents else 0.0
    avg_tpl = float(np.mean(tpls)) if tpls else 0.0
    print(f"[PSE] split={args.split} avg_entities={avg_ent:.2f} avg_tuples={avg_tpl:.2f} coverage={cov*100:.2f}%")


if __name__ == "__main__":
    main()

