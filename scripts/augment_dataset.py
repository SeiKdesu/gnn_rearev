"""
Offline dataset augmentation with Policy-based Subgraph Expansion (PSE).

This writes a new JSONL split with expanded subgraphs, avoiding doing expansion at every epoch.

Example:
  `python3 scripts/augment_dataset.py --data_folder data/CWQ/ --split dev \\
     --out_path data/CWQ/dev.pse.json --union_graph_cache data/CWQ/cache/union_graph.pkl \\
     --policy_ckpt data/CWQ/cache/policy_ckpt.pt`
"""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from subgraph_union_graph import load_union_graph
from subgraph_policy import expand_sample_subgraph, load_policy_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True, type=str)
    ap.add_argument("--split", required=True, type=str, choices=["train", "dev", "test"])
    ap.add_argument("--out_path", required=True, type=str)

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
    in_path = os.path.join(data_folder, f"{args.split}.json")
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    union_cache = args.union_graph_cache or os.path.join(data_folder, "cache/union_graph.pkl")
    policy_ckpt = args.policy_ckpt or os.path.join(data_folder, "cache/policy_ckpt.pt")
    union_adj, _ = load_union_graph(union_cache)
    policy = load_policy_ckpt(policy_ckpt, word2id=None, map_location=args.policy_device, override_device=args.policy_device)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Augmenting {args.split}"):
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
            fout.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()

