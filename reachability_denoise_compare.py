#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Set

from modules.denoise_subgraph import SubgraphDenoiser


def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def load_relations_txt(path: str) -> Dict[str, int]:
    """relations.txt (1 relation per line) -> relation2id dict"""
    relation2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel and rel not in relation2id:
                relation2id[rel] = len(relation2id)
    return relation2id


def infer_relation2id_from_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Fallback: scan tuples and enumerate relations that appear."""
    relation2id = {}
    for sample in data:
        sub = sample.get("subgraph", {}) or {}
        for tpl in sub.get("tuples", []) or []:
            if not isinstance(tpl, (list, tuple)) or len(tpl) < 3:
                continue
            rel = tpl[1]
            # rel may be dict or str/int
            if isinstance(rel, dict):
                rel = rel.get("text", None) or rel.get("kb_id", None)
            if rel is None:
                continue
            rel = str(rel)
            if rel not in relation2id:
                relation2id[rel] = len(relation2id)
    return relation2id


def entity_key(ent):
    if isinstance(ent, dict):
        # IMPORTANT: prefer kb_id for FreebaseQA/CWQ
        if "kb_id" in ent and ent["kb_id"] is not None:
            return ent["kb_id"]
        if "text" in ent and ent["text"] is not None:
            return ent["text"]
    return ent


def extract_topics(sample: Dict[str, Any]) -> List[Any]:
    if "entities_cid" in sample and sample["entities_cid"] is not None:
        return [entity_key(e) for e in sample.get("entities_cid", [])]
    return [entity_key(e) for e in sample.get("entities", [])]


def extract_answers(sample):
    out = []
    for a in sample.get("answers", []):
        out.append(entity_key(a))  # dictなら kb_id を拾う
    return out


def tuple_endpoints(subgraph: Dict[str, Any], mid2id: Dict[str, int]) -> Set[int]:
    tpls = subgraph.get("tuples", []) or []
    s: Set[int] = set()
    for tpl in tpls:
        if not isinstance(tpl, (list, tuple)) or len(tpl) < 3:
            continue
        a = to_int_ent(tpl[0], mid2id)
        b = to_int_ent(tpl[2], mid2id)
        if a is not None:
            s.add(a)
        if b is not None:
            s.add(b)
    return s


def build_adj(subgraph: Dict[str, Any], directed: bool, mid2id: Dict[str, int]) -> Dict[int, List[int]]:
    tpls = subgraph.get("tuples", []) or []
    adj: Dict[int, List[int]] = defaultdict(list)
    for tpl in tpls:
        if not isinstance(tpl, (list, tuple)) or len(tpl) < 3:
            continue
        u = to_int_ent(tpl[0], mid2id)
        v = to_int_ent(tpl[2], mid2id)
        if u is None or v is None:
            continue
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    return adj



def min_dist_bfs(
    adj: Dict[Any, List[Any]],
    sources: Iterable[Any],
    targets: Set[Any],
    max_hops: int,
) -> int:
    sources = [s for s in sources if s is not None]
    if not sources or not targets:
        return -1

    for s in sources:
        if s in targets:
            return 0

    q = deque()
    dist = {}
    for s in sources:
        if s not in dist:
            dist[s] = 0
            q.append(s)

    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_hops:
            continue
        for v in adj.get(u, []):
            if v in dist:
                continue
            dv = du + 1
            dist[v] = dv
            if v in targets:
                return dv
            if dv < max_hops:
                q.append(v)
    return -1


def reachable_nodes(adj: Dict[Any, List[Any]], sources: Iterable[Any]) -> Set[Any]:
    sources = [s for s in sources if s is not None]
    if not sources:
        return set()
    seen = set()
    q = deque()
    for s in sources:
        if s not in seen:
            seen.add(s)
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            q.append(v)
    return seen


def count_entities(subgraph: Dict[str, Any], mid2id: Dict[str, int]) -> int:
    ents = subgraph.get("entities")
    if ents is not None:
        # entities が int のリストならそのまま。MIDなら変換して数える
        cnt = 0
        for e in ents:
            if to_int_ent(e, mid2id) is not None:
                cnt += 1
        return cnt
    return len(tuple_endpoints(subgraph, mid2id))


def load_mid2id_from_entities_txt(path: str) -> dict:
    mid2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            mid = line.strip()
            if mid:
                mid2id[mid] = i  # 0-based index
    return mid2id

def to_int_ent(x: Any, mid2id: Dict[str, int]) -> Optional[int]:
    x = entity_key(x)
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        return mid2id.get(x, None)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test.json (jsonl or json array)")
    ap.add_argument("--relations", default=None, help="relations.txt (1 relation per line). If omitted, infer from data.")
    ap.add_argument("--max_hops", type=int, default=6)
    ap.add_argument("--directed", action="store_true")

    # denoiser config
    ap.add_argument("--enable_denoise", type=str, default="true")
    ap.add_argument("--encoder_type", type=str, default="tfidf")
    ap.add_argument("--topM_strict", type=int, default=10)
    ap.add_argument("--topK_strict", type=int, default=50)
    ap.add_argument("--gamma_strict", type=float, default=0.2)
    ap.add_argument("--topM_loose", type=int, default=25)
    ap.add_argument("--topK_loose", type=int, default=150)
    ap.add_argument("--gamma_loose", type=float, default=0.05)
    ap.add_argument("--min_edges", type=float, default=50.0)
    ap.add_argument("--ensure_connectivity", type=str, default="true")
    ap.add_argument("--enable_pair_score", type=str, default="false")
    ap.add_argument("--pair_stats_path", type=str, default=None)
    ap.add_argument("--relation_desc_path", type=str, default=None)
    ap.add_argument("--encoder_model_name", type=str, default=None)
    ap.add_argument("--entities", required=True, help="entities.txt (one MID per line; line index == entity id)")


    ap.add_argument("--show_misses", type=int, default=5)
    args = ap.parse_args()
    mid2id = load_mid2id_from_entities_txt(args.entities)

    data = load_json_or_jsonl(args.data)

    if args.relations:
        relation2id = load_relations_txt(args.relations)
        if not relation2id:
            raise ValueError(f"relations file is empty: {args.relations}")
    else:
        relation2id = infer_relation2id_from_data(data)
        if not relation2id:
            raise ValueError("Could not infer relations from data. Provide --relations relations.txt")

    denoise_cfg = {
        "enable_denoise": str2bool(args.enable_denoise),
        "encoder_type": args.encoder_type,
        "topM_strict": args.topM_strict,
        "topK_strict": args.topK_strict,
        "gamma_strict": args.gamma_strict,
        "topM_loose": args.topM_loose,
        "topK_loose": args.topK_loose,
        "gamma_loose": args.gamma_loose,
        "min_edges": args.min_edges,
        "ensure_connectivity": str2bool(args.ensure_connectivity),
        "enable_pair_score": str2bool(args.enable_pair_score),
        "pair_stats_path": args.pair_stats_path,
        "relation_desc_path": args.relation_desc_path,
        "encoder_model_name": args.encoder_model_name,
    }

    denoiser = SubgraphDenoiser(relation2id, denoise_cfg)

    total = 0
    stats = {
        "ans_in_before": 0,
        "ans_in_after": 0,
        "reach_before": 0,
        "reach_after": 0,
        "drop_ans": 0,
        "drop_reach": 0,
        "off_seed_before": 0,
        "off_seed_after": 0,
    }
    seed_checked = 0

    sum_ent_before = 0
    sum_ent_after = 0
    sum_edge_before = 0
    sum_edge_after = 0

    misses = []

    for sample in data:

        total += 1
        q = sample.get("question", "")
        sub_before = sample.get("subgraph", {}) or {}
        topics_raw = extract_topics(sample)
        answers_raw = extract_answers(sample)

        topics = [to_int_ent(t, mid2id) for t in topics_raw]
        topics = [t for t in topics if t is not None]

        answers = [to_int_ent(a, mid2id) for a in answers_raw]
        answers = set(a for a in answers if a is not None)

        endpoints_before = tuple_endpoints(sub_before, mid2id)
        if total == 1:
            print("ANS:", answers)
            print("EP :", list(endpoints_before)[:20])  # 長いので先頭だけ推奨

        ans_in_before = bool(answers & endpoints_before)
        adj_before = build_adj(sub_before, directed=args.directed, mid2id=mid2id)
        d_before = min_dist_bfs(adj_before, topics, answers, args.max_hops)
        reach_before = (d_before >= 0)

        adj_before_undirected = build_adj(sub_before, directed=False, mid2id=mid2id)
        reachable_before = reachable_nodes(adj_before_undirected, topics)
        off_seed_before = False
        if topics:
            seed_checked += 1
            off_seed_before = bool(endpoints_before - reachable_before)

        sub_after = denoiser.denoise(q, sub_before, topics)

        endpoints_after = tuple_endpoints(sub_after, mid2id)
        ans_in_after = bool(answers & endpoints_after)
        adj_after = build_adj(sub_after, directed=args.directed, mid2id=mid2id)
        d_after = min_dist_bfs(adj_after, topics, answers, args.max_hops)
        reach_after = (d_after >= 0)

        adj_after_undirected = build_adj(sub_after, directed=False, mid2id=mid2id)
        reachable_after = reachable_nodes(adj_after_undirected, topics)
        off_seed_after = False
        if topics:
            off_seed_after = bool(endpoints_after - reachable_after)

        stats["ans_in_before"] += int(ans_in_before)
        stats["ans_in_after"] += int(ans_in_after)
        stats["reach_before"] += int(reach_before)
        stats["reach_after"] += int(reach_after)

        stats["drop_ans"] += int(ans_in_before and not ans_in_after)
        stats["drop_reach"] += int(reach_before and not reach_after)
        stats["off_seed_before"] += int(off_seed_before)
        stats["off_seed_after"] += int(off_seed_after)

        sum_ent_before += count_entities(sub_before, mid2id)
        sum_ent_after += count_entities(sub_after, mid2id)

        sum_edge_before += len(sub_before.get("tuples", []) or [])
        sum_edge_after += len(sub_after.get("tuples", []) or [])

        if len(misses) < args.show_misses and ans_in_before and not ans_in_after:
            misses.append({
                "id": sample.get("id"),
                "question": q[:120],
                "edges_before": len(sub_before.get("tuples", []) or []),
                "edges_after": len(sub_after.get("tuples", []) or []),
                "dist_before": d_before,
                "dist_after": d_after,
            })

    def rate(x, n): return x / n if n else 0.0

    print("==== Denoise Reachability Compare ====")
    print(f"data: {args.data}")
    print(f"mode: {'directed' if args.directed else 'undirected'}  max_hops={args.max_hops}")
    print(f"total: {total}")

    print("\n-- Answer-in-subgraph (endpoints) --")
    print(f"before: {stats['ans_in_before']} ({rate(stats['ans_in_before'], total):.4f})")
    print(f"after : {stats['ans_in_after']} ({rate(stats['ans_in_after'], total):.4f})")
    print(f"drop_ans (True->False): {stats['drop_ans']} ({rate(stats['drop_ans'], total):.4f})")

    print("\n-- Reachable@max_hops (topic -> answer) --")
    print(f"before: {stats['reach_before']} ({rate(stats['reach_before'], total):.4f})")
    print(f"after : {stats['reach_after']} ({rate(stats['reach_after'], total):.4f})")
    print(f"drop_reach (True->False): {stats['drop_reach']} ({rate(stats['drop_reach'], total):.4f})")

    print("\n-- Seed connectivity (all endpoints reachable from topics, undirected) --")
    print(f"checked: {seed_checked}")
    if seed_checked:
        print(f"off_seed before: {stats['off_seed_before']} ({rate(stats['off_seed_before'], seed_checked):.4f})")
        print(f"off_seed after : {stats['off_seed_after']} ({rate(stats['off_seed_after'], seed_checked):.4f})")

    print("\n-- Size (avg) --")
    print(f"avg_edges    before: {sum_edge_before/total:.2f}  after: {sum_edge_after/total:.2f}")
    print(f"avg_entities before: {sum_ent_before/total:.2f}  after: {sum_ent_after/total:.2f}")

    if misses:
        print("\n-- examples where denoise dropped answer --")
        for m in misses:
            print(json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
