"""
Train-only Union Graph builder/loader for Policy-based Subgraph Expansion (PSE).

Usage (typically via `python main.py --mode train_policy ...`):
  - Build union graph cache from `<data_folder>/train.json` only.
  - Cache path default: `<data_folder>/cache/union_graph.pkl`
"""

from __future__ import annotations

import os
import json
import pickle
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm


def _coerce_entity_id(x: Any, entity2id: Dict[Any, int]) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        if x in entity2id:
            return int(entity2id[x])
        # Sometimes ids may come as numeric strings.
        try:
            return int(x)
        except Exception as e:  # pragma: no cover
            raise KeyError(f"Unknown entity string: {x!r}") from e
    if isinstance(x, dict):
        if "text" in x and x["text"] in entity2id:
            return int(entity2id[x["text"]])
        if "kb_id" in x and x["kb_id"] in entity2id:
            return int(entity2id[x["kb_id"]])
        if "id" in x and x["id"] in entity2id:
            return int(entity2id[x["id"]])
        raise KeyError(f"Unknown entity dict keys: {list(x.keys())}")  # pragma: no cover
    raise TypeError(f"Unsupported entity type: {type(x)}")  # pragma: no cover


def _coerce_relation_id(x: Any, relation2id: Dict[Any, int]) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        if x in relation2id:
            return int(relation2id[x])
        try:
            return int(x)
        except Exception as e:  # pragma: no cover
            raise KeyError(f"Unknown relation string: {x!r}") from e
    if isinstance(x, dict):
        if "text" in x and x["text"] in relation2id:
            return int(relation2id[x["text"]])
        if "id" in x and x["id"] in relation2id:
            return int(relation2id[x["id"]])
        raise KeyError(f"Unknown relation dict keys: {list(x.keys())}")  # pragma: no cover
    raise TypeError(f"Unsupported relation type: {type(x)}")  # pragma: no cover


def build_union_graph(
    train_json_path: str,
    entity2id: Dict[Any, int],
    relation2id: Dict[Any, int],
    use_inverse_relation: bool,
    cache_path: str,
) -> Dict[int, np.ndarray]:
    """
    Build a train-only union adjacency index: adj[h] = int32 array of shape [deg, 2] of (r, t).

    Important: MUST be built from train split only (no dev/test leakage).
    """
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"train.json not found at {train_json_path!r}")

    num_base_rel = len(relation2id)
    adj_lists: Dict[int, list[Tuple[int, int]]] = defaultdict(list)
    bad_edges = 0
    total_edges = 0

    with open(train_json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building union graph (train-only)"):
            if not line.strip():
                continue
            sample = json.loads(line)
            tuples = sample.get("subgraph", {}).get("tuples", [])
            for tpl in tuples:
                if not isinstance(tpl, (list, tuple)) or len(tpl) != 3:
                    continue
                h_raw, r_raw, t_raw = tpl
                try:
                    h = _coerce_entity_id(h_raw, entity2id)
                    r = _coerce_relation_id(r_raw, relation2id)
                    t = _coerce_entity_id(t_raw, entity2id)
                except Exception:
                    bad_edges += 1
                    continue

                total_edges += 1
                adj_lists[int(h)].append((int(r), int(t)))
                if use_inverse_relation:
                    if int(r) >= num_base_rel:
                        # r already in inverse id space; add base direction.
                        adj_lists[int(t)].append((int(r) - num_base_rel, int(h)))
                    else:
                        adj_lists[int(t)].append((int(r) + num_base_rel, int(h)))

    adj: Dict[int, np.ndarray] = {}
    for h, lst in adj_lists.items():
        if not lst:
            continue
        arr = np.asarray(lst, dtype=np.int32)
        adj[int(h)] = arr

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "adj": adj,
        "meta": {
            "train_json_path": os.path.abspath(train_json_path),
            "num_base_rel": num_base_rel,
            "use_inverse_relation": bool(use_inverse_relation),
            "total_edges": int(total_edges),
            "bad_edges": int(bad_edges),
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return adj


def load_union_graph(cache_path: str) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Union graph cache not found at {cache_path!r}. "
            "Build it using `python main.py --mode train_policy ...`."
        )
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "adj" not in payload:
        raise ValueError(f"Invalid union graph cache format: {cache_path!r}")
    adj = payload["adj"]
    meta = payload.get("meta", {})
    return adj, meta
