#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter


def _iter_entities_from_merged_subgraph_json(path: str):
    """
    Streaming parse for files written by merge_cwq_subgraphs.py (starts with: {"entities":[...],"tuples":[...]}).
    Falls back to json.load if the prefix doesn't match.
    """
    with open(path, "r", encoding="utf-8") as f:
        prefix = f.read(len('{"entities":['))
        if prefix != '{"entities":[':
            f.seek(0)
            obj = json.load(f)
            for e in obj.get("entities", []):
                yield int(e)
            return

        buf = ""
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            for ch in chunk:
                if ch == "]":
                    if buf:
                        yield int(buf)
                        buf = ""
                    return
                if ch == "-" or ("0" <= ch <= "9"):
                    buf += ch
                    continue
                if buf:
                    yield int(buf)
                    buf = ""

        raise RuntimeError(f"unexpected EOF while parsing entities array in {path}")


def _extract_answer_value(answer):
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        # mirrors dataset_load.py logic:
        # keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
        if "kb_id" in answer and isinstance(answer["kb_id"], int):
            return answer.get("text", answer["kb_id"])
        if "kb_id" in answer:
            return answer["kb_id"]
        if "text" in answer:
            return answer["text"]
    return None


def _collect_answer_counts(train_jsonl_path: str):
    t0 = time.time()
    str_counts: Counter[str] = Counter()
    id_counts: Counter[int] = Counter()
    total_mentions = 0
    line_count = 0

    with open(train_jsonl_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            answers = obj.get("answers_cid", None)
            if answers is None:
                answers = obj.get("answers", [])

            for a in answers:
                v = _extract_answer_value(a)
                if v is None:
                    continue
                total_mentions += 1
                if isinstance(v, int):
                    id_counts[int(v)] += 1
                else:
                    str_counts[str(v)] += 1
            line_count += 1

            if line_count % 5000 == 0:
                dt = time.time() - t0
                print(
                    f"[{line_count:,} lines] answer_mentions={total_mentions:,} unique_int={len(id_counts):,} unique_str={len(str_counts):,} elapsed={dt:,.1f}s",
                    file=sys.stderr,
                )

    return str_counts, id_counts, total_mentions


def _map_answer_strings_to_ids(entities_txt_path: str, needed: set[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with open(entities_txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ent = line.strip()
            if ent in needed:
                mapping[ent] = i
    return mapping


def _count_with_sqlite_db(db_path: str, answer_id_counts: Counter[int]):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("CREATE TEMP TABLE answers (id INTEGER PRIMARY KEY, cnt INTEGER) WITHOUT ROWID;")

        rows = [(int(i), int(c)) for i, c in answer_id_counts.items()]
        conn.executemany("INSERT INTO answers(id, cnt) VALUES (?, ?);", rows)

        unique_total = conn.execute("SELECT COUNT(*) FROM answers;").fetchone()[0]
        unique_present = conn.execute(
            "SELECT COUNT(*) FROM answers a INNER JOIN entities e ON a.id = e.id;"
        ).fetchone()[0]
        mention_total = conn.execute("SELECT COALESCE(SUM(cnt), 0) FROM answers;").fetchone()[0]
        mention_present = conn.execute(
            "SELECT COALESCE(SUM(a.cnt), 0) FROM answers a INNER JOIN entities e ON a.id = e.id;"
        ).fetchone()[0]
        return unique_total, unique_present, mention_total, mention_present
    finally:
        conn.close()


def _count_with_subgraph_json(subgraph_json_path: str, answer_id_counts: Counter[int]):
    present_unique = 0
    present_mentions = 0
    seen: set[int] = set()
    for ent_id in _iter_entities_from_merged_subgraph_json(subgraph_json_path):
        if ent_id in answer_id_counts and ent_id not in seen:
            seen.add(ent_id)
            present_unique += 1
            present_mentions += answer_id_counts[ent_id]
    unique_total = len(answer_id_counts)
    mention_total = sum(answer_id_counts.values())
    return unique_total, present_unique, mention_total, present_mentions


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Count how many answer entities are included in the merged CWQ subgraph."
    )
    ap.add_argument("--train", default="data/CWQ/train.json", help="Path to CWQ train.json (JSONL)")
    ap.add_argument("--entities-txt", default="data/CWQ/entities.txt", help="Path to CWQ entities.txt")
    ap.add_argument(
        "--subgraph",
        default="data/CWQ/subgraph.json",
        help="Merged subgraph.json (written by merge_cwq_subgraphs.py)",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="Optional: SQLite DB from merge_cwq_subgraphs.py (faster than scanning subgraph.json)",
    )
    args = ap.parse_args()

    t0 = time.time()
    str_counts, id_counts, total_mentions_raw = _collect_answer_counts(args.train)

    str_to_id = _map_answer_strings_to_ids(args.entities_txt, set(str_counts.keys()))
    unmapped = 0
    for s, c in str_counts.items():
        if s not in str_to_id:
            unmapped += c
            continue
        id_counts[str_to_id[s]] += c

    if args.db is not None:
        unique_total, unique_present, mention_total, mention_present = _count_with_sqlite_db(args.db, id_counts)
    else:
        unique_total, unique_present, mention_total, mention_present = _count_with_subgraph_json(args.subgraph, id_counts)

    dt = time.time() - t0
    print(f"train_answer_mentions_total: {total_mentions_raw}")
    print(f"train_answer_mentions_unmapped: {unmapped}")
    print(f"train_answer_entities_unique: {unique_total}")
    print(f"merged_subgraph_answer_entities_unique: {unique_present}")
    print(f"train_answer_mentions_mapped: {mention_total}")
    print(f"merged_subgraph_answer_mentions_mapped: {mention_present}")
    print(f"elapsed_sec: {dt:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

