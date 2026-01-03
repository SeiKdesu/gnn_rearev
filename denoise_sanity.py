# denoise_sanity.py
# Fix: answers are MIDs (str) but tuples use entity IDs (int).
# This script unifies the space by converting answers/topic entities to entity IDs using ent2id*.pickle.

import argparse
import glob
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dataset_load import load_dict, load_dict_int
from modules.denoise_subgraph import SubgraphDenoiser
from parsing import bool_flag


# ---------- Utilities ----------
def load_relation2id(data_folder: str, relation2id_file: str):
    path = os.path.join(data_folder, relation2id_file)
    if "sr-cwq" in data_folder:
        return load_dict_int(path)
    return load_dict(path)


def load_ent2id(data_folder: str) -> Optional[Dict[str, int]]:
    """
    Find ent2id*.pickle in data_folder and load it.
    """
    cands = glob.glob(os.path.join(data_folder, "ent2id*.pickle"))
    if not cands:
        return None
    path = sorted(cands)[0]
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"ent2id pickle is not a dict: {path}")
    return obj


def normalize_mid(x: str) -> str:
    """
    Normalize common MID formats.
    """
    x = x.strip()
    if x.startswith("/m/"):
        return "m." + x[3:]
    return x


def get_kb_id(ent: Any) -> Optional[str]:
    """
    Extract kb_id from dict or return string if looks like MID.
    """
    if isinstance(ent, dict):
        if "kb_id" in ent and isinstance(ent["kb_id"], str):
            return normalize_mid(ent["kb_id"])
        return None
    if isinstance(ent, str):
        return normalize_mid(ent)
    return None


def as_int_id(ent: Any) -> Optional[int]:
    """
    If ent is already an int ID, return it; else None.
    """
    return ent if isinstance(ent, int) else None


# ---------- Extraction in a unified (int-id) space ----------
def extract_answer_ids(sample: Dict[str, Any], ent2id: Optional[Dict[str, int]]) -> Set[int]:
    """
    Return gold answer entity IDs (int) to match tuples (int).
    Priority:
      - answers_cid if it is non-empty
      - else answers[].kb_id -> ent2id
    """
    ans_ids: Set[int] = set()

    # Prefer answers_cid only if non-empty
    answers_cid = sample.get("answers_cid") or []
    if answers_cid:
        for a in answers_cid:
            iid = as_int_id(a)
            if iid is not None:
                ans_ids.add(iid)
                continue
            kb = get_kb_id(a)
            if kb and ent2id and kb in ent2id:
                ans_ids.add(ent2id[kb])
        if ans_ids:
            return ans_ids

    # Fallback to answers (MID space)
    for a in sample.get("answers") or []:
        kb = get_kb_id(a)
        if kb and ent2id and kb in ent2id:
            ans_ids.add(ent2id[kb])

    return ans_ids


def extract_topic_ids(sample: Dict[str, Any], ent2id: Optional[Dict[str, int]]) -> Set[int]:
    """
    Return topic entity IDs (int) to match tuples (int).
    Priority:
      - entities_cid if non-empty
      - else entities -> ent2id
    """
    topic_ids: Set[int] = set()

    entities_cid = sample.get("entities_cid") or []
    if entities_cid:
        for e in entities_cid:
            iid = as_int_id(e)
            if iid is not None:
                topic_ids.add(iid)
                continue
            kb = get_kb_id(e)
            if kb and ent2id and kb in ent2id:
                topic_ids.add(ent2id[kb])
        if topic_ids:
            return topic_ids

    for e in sample.get("entities") or []:
        iid = as_int_id(e)
        if iid is not None:
            topic_ids.add(iid)
            continue
        kb = get_kb_id(e)
        if kb and ent2id and kb in ent2id:
            topic_ids.add(ent2id[kb])

    return topic_ids


# ---------- Graph operations (int-id space) ----------
def count_nodes_int(tuples: List[List[Any]]) -> int:
    nodes: Set[int] = set()
    for sbj, _, obj in tuples:
        if isinstance(sbj, int):
            nodes.add(sbj)
        if isinstance(obj, int):
            nodes.add(obj)
    return len(nodes)


def tuple_node_ids(tuples: List[List[Any]]) -> Set[int]:
    nodes: Set[int] = set()
    for sbj, _, obj in tuples:
        if isinstance(sbj, int):
            nodes.add(sbj)
        if isinstance(obj, int):
            nodes.add(obj)
    return nodes


def answer_in_tuples_ids(answer_ids: Set[int], tuples: List[List[Any]]) -> bool:
    if not answer_ids:
        return False
    return bool(answer_ids & tuple_node_ids(tuples))


def build_undirected_adj_ids(tuples: List[List[Any]]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    for sbj, _, obj in tuples:
        # Expect int ids
        if not isinstance(sbj, int) or not isinstance(obj, int):
            continue
        adj.setdefault(sbj, set()).add(obj)
        adj.setdefault(obj, set()).add(sbj)
    return adj


def reachable_within_hops(adj: Dict[int, Set[int]], start_nodes: Set[int], target_nodes: Set[int], max_hops: int) -> bool:
    if not start_nodes or not target_nodes:
        return False
    if start_nodes & target_nodes:
        return True

    visited = set(start_nodes)
    frontier = set(start_nodes)

    for _ in range(max_hops):
        next_frontier: Set[int] = set()
        for node in frontier:
            for nbr in adj.get(node, set()):
                if nbr in visited:
                    continue
                if nbr in target_nodes:
                    return True
                visited.add(nbr)
                next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    return False


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data/CWQ_freebase/", type=str)
    parser.add_argument("--split", default="dev", type=str, choices=["train", "dev", "test"])
    parser.add_argument("--sample_idx", default=0, type=int)
    parser.add_argument("--relation2id", default="relations.txt", type=str)

    parser.add_argument("--enable_denoise", default=True, type=bool_flag)
    parser.add_argument("--encoder_type", default="tfidf", type=str)
    parser.add_argument("--encoder_model_name", default=None, type=str)
    parser.add_argument("--relation_desc_path", default=None, type=str)
    parser.add_argument("--topM_strict", default=10, type=int)
    parser.add_argument("--topK_strict", default=50, type=int)
    parser.add_argument("--gamma_strict", default=0.2, type=float)
    parser.add_argument("--topM_loose", default=25, type=int)
    parser.add_argument("--topK_loose", default=150, type=int)
    parser.add_argument("--gamma_loose", default=0.05, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.2, type=float)
    parser.add_argument("--min_edges", default=50, type=float)
    parser.add_argument("--enable_pair_score", default=False, type=bool_flag)
    parser.add_argument("--pair_stats_path", default=None, type=str)
    parser.add_argument("--ensure_connectivity", default=True, type=bool_flag)

    args = parser.parse_args()

    sample_path = os.path.join(args.data_folder, f"{args.split}.json")
    sample = None
    with open(sample_path, "r", encoding="utf-8") as f_in:
        for idx, line in enumerate(f_in):
            if idx == args.sample_idx:
                sample = json.loads(line)
                break
    if sample is None:
        raise ValueError(f"sample_idx {args.sample_idx} out of range for {sample_path}")

    relation2id = load_relation2id(args.data_folder, args.relation2id)
    denoiser = SubgraphDenoiser(relation2id, vars(args))

    # Load ent2id for MID -> int conversion
    ent2id = load_ent2id(args.data_folder)
    if ent2id is None:
        print("[WARN] ent2id*.pickle not found in data_folder. "
              "Answer/topic checks may be invalid if tuples use int IDs.")
    else:
        print(f"[INFO] Loaded ent2id with {len(ent2id):,} entries.")

    original_tuples = sample["subgraph"]["tuples"]
    original_entities = sample["subgraph"].get("entities")
    original_entity_count = len(original_entities) if original_entities is not None else count_nodes_int(original_tuples)

    # Important: avoid passing empty entities_cid by using "or" fallback
    topic_entities_for_denoiser = sample.get("entities_cid") or sample.get("entities") or []

    denoised = denoiser.denoise(
        sample["question"],
        sample["subgraph"],
        topic_entities=topic_entities_for_denoiser,
    )

    denoised_tuples = denoised["tuples"]
    denoised_entities = denoised.get("entities")
    denoised_entity_count = len(denoised_entities) if denoised_entities is not None else count_nodes_int(denoised_tuples)

    # Unified checks in int-id space
    answer_ids = extract_answer_ids(sample, ent2id)
    topic_ids = extract_topic_ids(sample, ent2id)

    original_answer_in = answer_in_tuples_ids(answer_ids, original_tuples)
    denoised_answer_in = answer_in_tuples_ids(answer_ids, denoised_tuples)

    original_adj = build_undirected_adj_ids(original_tuples)
    denoised_adj = build_undirected_adj_ids(denoised_tuples)
    original_reachable = reachable_within_hops(original_adj, topic_ids, answer_ids, 3)
    denoised_reachable = reachable_within_hops(denoised_adj, topic_ids, answer_ids, 3)

    # ---------- Print ----------
    print("Question:", sample.get("question", ""))
    print("Original edges:", len(original_tuples), "entities:", original_entity_count)
    print("Denoised edges:", len(denoised_tuples), "entities:", denoised_entity_count)

    if original_tuples:
        reduction = 100.0 * (1.0 - (len(denoised_tuples) / len(original_tuples)))
        print(f"Edge reduction: {reduction:.1f}%")
    if original_entity_count:
        ent_reduction = 100.0 * (1.0 - (denoised_entity_count / original_entity_count))
        print(f"Entity reduction: {ent_reduction:.1f}%")

    # Debug prints to ensure we are in the same ID space
    print("answer_ids size:", len(answer_ids), "sample:", list(answer_ids)[:5])
    print("topic_ids  size:", len(topic_ids),  "sample:", list(topic_ids)[:5])
    print("tuple_nodes size:", len(tuple_node_ids(original_tuples)))

    print("Answer in tuples (original):", original_answer_in)
    print("Answer in tuples (denoised):", denoised_answer_in)
    print("Answer reachable within 3 hops (original, undirected):", original_reachable)
    print("Answer reachable within 3 hops (denoised, undirected):", denoised_reachable)


if __name__ == "__main__":
    main()
