#!/usr/bin/env python3
"""
Build a shared (merged) subgraph from one or more *.jsonl splits.

Output format (strict):
  {"tuples": [[h,r,t], ...], "entities": [e0,e1,...]}

Notes:
  - Deduplicates tuples by exact (h,r,t).
  - Unions entities from:
      * sample["subgraph"]["entities"]
      * endpoints of sample["subgraph"]["tuples"]
      * sample["entities"] / sample["entities_cid"] (question entities)
  - NEVER uses answers (no leakage).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Set, Tuple, Union


JsonScalar = Union[int, str]


def _load_dict(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if not k:
                continue
            out[k] = len(out)
    return out


def _load_dict_int_identity(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if not k:
                continue
            out[int(k)] = int(k)
    return out


def _maybe_int(x: Any) -> Any:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x)
        except Exception:
            return x
    return x


def _coerce_entity_id(x: Any, entity2id: Dict[Any, int] | None) -> JsonScalar:
    x = _maybe_int(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        if entity2id is not None and x in entity2id:
            return int(entity2id[x])
        return x
    if isinstance(x, dict):
        for key in ("text", "kb_id", "id"):
            if key not in x:
                continue
            v = _maybe_int(x[key])
            if isinstance(v, int):
                return int(v)
            if isinstance(v, str):
                if entity2id is not None and v in entity2id:
                    return int(entity2id[v])
                return v
        raise KeyError(f"Unsupported entity dict keys: {list(x.keys())}")
    raise TypeError(f"Unsupported entity type: {type(x)}")


def _coerce_relation_id(x: Any, relation2id: Dict[Any, int] | None) -> JsonScalar:
    x = _maybe_int(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        if relation2id is not None and x in relation2id:
            return int(relation2id[x])
        return x
    if isinstance(x, dict):
        for key in ("text", "id"):
            if key not in x:
                continue
            v = _maybe_int(x[key])
            if isinstance(v, int):
                return int(v)
            if isinstance(v, str):
                if relation2id is not None and v in relation2id:
                    return int(relation2id[v])
                return v
        raise KeyError(f"Unsupported relation dict keys: {list(x.keys())}")
    raise TypeError(f"Unsupported relation type: {type(x)}")


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_shared_subgraph(
    input_jsonl_paths: List[str],
    entity2id: Dict[Any, int] | None = None,
    relation2id: Dict[Any, int] | None = None,
) -> Tuple[List[List[JsonScalar]], List[JsonScalar]]:
    tuples_set: Set[Tuple[JsonScalar, JsonScalar, JsonScalar]] = set()
    entities_set: Set[JsonScalar] = set()
    total_tuples = 0
    kept_tuples = 0

    for path in input_jsonl_paths:
        for sample in _iter_jsonl(path):
            # Union question entities (NOT answers).
            key_ent = "entities_cid" if "entities_cid" in sample else "entities"
            for e in sample.get(key_ent, []) or []:
                try:
                    entities_set.add(_coerce_entity_id(e, entity2id))
                except Exception:
                    continue

            sg = sample.get("subgraph", {}) or {}
            for e in sg.get("entities", []) or []:
                try:
                    entities_set.add(_coerce_entity_id(e, entity2id))
                except Exception:
                    continue

            for tpl in sg.get("tuples", []) or []:
                if not isinstance(tpl, (list, tuple)) or len(tpl) != 3:
                    continue
                total_tuples += 1
                h_raw, r_raw, t_raw = tpl
                try:
                    h = _coerce_entity_id(h_raw, entity2id)
                    r = _coerce_relation_id(r_raw, relation2id)
                    t = _coerce_entity_id(t_raw, entity2id)
                except Exception:
                    continue

                entities_set.add(h)
                entities_set.add(t)
                key = (h, r, t)
                if key in tuples_set:
                    continue
                tuples_set.add(key)
                kept_tuples += 1

    tuples_out = [[h, r, t] for (h, r, t) in tuples_set]

    def _ent_key(v: JsonScalar):
        if isinstance(v, int):
            return (0, v)
        return (1, str(v))

    entities_out = sorted(entities_set, key=_ent_key)
    print(
        f"[build_shared_subgraph] inputs={len(input_jsonl_paths)} "
        f"|V|={len(entities_out)} tuples={kept_tuples} (from {total_tuples}, dedup+valid only)"
    )
    return tuples_out, entities_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input jsonl path. Repeatable (e.g., --input train.json --input dev.json).",
    )
    ap.add_argument("--output", required=True, help="Output subgraph.json path.")
    ap.add_argument(
        "--entities_txt",
        default=None,
        help="Optional entities.txt path (used only to coerce string ids to ints).",
    )
    ap.add_argument(
        "--relations_txt",
        default=None,
        help="Optional relations.txt path (used only to coerce string rels to ints).",
    )
    ap.add_argument(
        "--sr_cwq_int_entities",
        action="store_true",
        help="If set, treats entities.txt lines as integer ids (id->id).",
    )
    args = ap.parse_args()

    for p in args.input:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    entity2id = None
    if args.entities_txt:
        entity2id = (
            _load_dict_int_identity(args.entities_txt)
            if args.sr_cwq_int_entities
            else _load_dict(args.entities_txt)
        )

    relation2id = None
    if args.relations_txt:
        relation2id = _load_dict(args.relations_txt)

    tuples, entities = build_shared_subgraph(args.input, entity2id=entity2id, relation2id=relation2id)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"tuples": tuples, "entities": entities}, f)
    print(f"[build_shared_subgraph] wrote: {args.output}")


if __name__ == "__main__":
    main()

