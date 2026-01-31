#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
import time
from statistics import mean


def _pctl(sorted_vals, p: float):
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
    conn.execute("CREATE INDEX IF NOT EXISTS tuples_by_o ON tuples(o, r, s);")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS frontier (id INTEGER PRIMARY KEY) WITHOUT ROWID;")
    conn.commit()


def _extract_answer_value(answer):
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        if "kb_id" in answer and isinstance(answer["kb_id"], int):
            return answer.get("text", answer["kb_id"])
        if "kb_id" in answer:
            return answer["kb_id"]
        if "text" in answer:
            return answer["text"]
    return None


def _load_entity2id_from_entities_txt(entities_txt_path: str):
    # entities.txt is line-aligned: id == line index
    entity2id = {}
    with open(entities_txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            entity2id[line.strip()] = i
    return entity2id


def _answers_to_ids(sample_obj, entity2id):
    answers = sample_obj.get("answers_cid", None)
    if answers is None:
        answers = sample_obj.get("answers", [])

    ids = set()
    for a in answers:
        v = _extract_answer_value(a)
        if v is None:
            continue
        if isinstance(v, int):
            ids.add(int(v))
        else:
            if v in entity2id:
                ids.add(int(entity2id[v]))
    return ids


def extract_subgraph_from_merged(
    conn: sqlite3.Connection,
    seed_entities: list[int],
    hops: int,
    max_entities: int,
    max_tuples: int,
    sql_limit: int,
):
    if not seed_entities:
        return set(), set()

    visited = set(int(e) for e in seed_entities)
    tuples = set()

    conn.execute("DELETE FROM frontier;")
    conn.executemany("INSERT OR IGNORE INTO frontier(id) VALUES (?);", [(int(e),) for e in visited])

    for _ in range(hops):
        if len(tuples) >= max_tuples:
            break
        remaining = max_tuples - len(tuples)
        lim = remaining if remaining < sql_limit else sql_limit
        if lim <= 0:
            break

        new_nodes = []
        new_nodes_set = set()

        cur = conn.execute(
            "SELECT t.s, t.r, t.o FROM frontier f JOIN tuples t ON t.s = f.id "
            "UNION ALL "
            "SELECT t.s, t.r, t.o FROM frontier f JOIN tuples t ON t.o = f.id "
            "LIMIT ?;",
            (int(lim),),
        )
        for s, r, o in cur:
            s = int(s)
            r = int(r)
            o = int(o)
            if (s, r, o) in tuples:
                continue

            missing = []
            if s not in visited:
                missing.append(s)
            if o not in visited:
                missing.append(o)
            if missing and len(visited) + len(missing) > max_entities:
                continue

            for m in missing:
                visited.add(m)
                if m not in new_nodes_set:
                    new_nodes_set.add(m)
                    new_nodes.append(m)

            tuples.add((s, r, o))
            if len(tuples) >= max_tuples:
                break

        if not new_nodes:
            break

        conn.execute("DELETE FROM frontier;")
        conn.executemany("INSERT OR IGNORE INTO frontier(id) VALUES (?);", [(int(e),) for e in new_nodes])

    return visited, tuples


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute k-hop subgraph sizes and coverage vs original per-sample subgraphs."
    )
    ap.add_argument("--data", default="data/CWQ/train.json", help="JSONL file (train/dev/test)")
    ap.add_argument("--db", default="data/CWQ/subgraph_merge.sqlite", help="SQLite DB from merge_cwq_subgraphs.py")
    ap.add_argument("--entities-txt", default="data/CWQ/entities.txt", help="entities.txt (for answer mapping)")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-entities", type=int, default=800)
    ap.add_argument("--max-tuples", type=int, default=4000)
    ap.add_argument("--sql-limit", type=int, default=20000)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = all lines, else stop after N lines")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--out-jsonl", default=None, help="Optional per-sample stats output (JSONL)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"merged DB not found: {args.db}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data file not found: {args.data}")

    t0 = time.time()
    entity2id = _load_entity2id_from_entities_txt(args.entities_txt) if args.entities_txt else {}

    conn = sqlite3.connect(args.db)
    try:
        _init_db(conn)

        ent_sizes = []
        edge_sizes = []
        ent_covs = []
        edge_covs = []
        answer_hit = 0
        answer_total = 0
        line_count = 0

        out_f = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
        try:
            with open(args.data, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    sg = obj.get("subgraph") or {}
                    orig_entities = set(int(e) for e in (sg.get("entities") or []))
                    orig_tuples = set(
                        (int(s), int(r), int(o)) for s, r, o in (sg.get("tuples") or [])
                    )

                    seed_key = "entities_cid" if "entities_cid" in obj else "entities"
                    seeds = obj.get(seed_key, []) or []
                    # CWQ stores seeds as ints already; handle dict/text defensively
                    seed_entities = []
                    for e in seeds:
                        if isinstance(e, int):
                            seed_entities.append(int(e))
                        elif isinstance(e, dict) and "text" in e and e["text"] in entity2id:
                            seed_entities.append(int(entity2id[e["text"]]))
                        elif isinstance(e, str) and e in entity2id:
                            seed_entities.append(int(entity2id[e]))

                    new_entities, new_tuples = extract_subgraph_from_merged(
                        conn=conn,
                        seed_entities=seed_entities,
                        hops=args.hops,
                        max_entities=args.max_entities,
                        max_tuples=args.max_tuples,
                        sql_limit=args.sql_limit,
                    )

                    ent_sizes.append(len(new_entities))
                    edge_sizes.append(len(new_tuples))

                    ent_cov = (len(orig_entities & new_entities) / len(orig_entities)) if orig_entities else 0.0
                    edge_cov = (len(orig_tuples & new_tuples) / len(orig_tuples)) if orig_tuples else 0.0
                    ent_covs.append(ent_cov)
                    edge_covs.append(edge_cov)

                    ans_ids = _answers_to_ids(obj, entity2id) if entity2id else set()
                    if ans_ids:
                        answer_total += 1
                        if any(a in new_entities for a in ans_ids):
                            answer_hit += 1

                    if out_f is not None:
                        out_f.write(
                            json.dumps(
                                {
                                    "id": obj.get("id"),
                                    "seed_entities": len(seed_entities),
                                    "orig_entities": len(orig_entities),
                                    "orig_tuples": len(orig_tuples),
                                    "khop_entities": len(new_entities),
                                    "khop_tuples": len(new_tuples),
                                    "entity_coverage": ent_cov,
                                    "tuple_coverage": edge_cov,
                                    "answer_in_subgraph": bool(ans_ids and any(a in new_entities for a in ans_ids)),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    line_count += 1
                    if args.max_lines and line_count >= args.max_lines:
                        break
                    if args.log_every and line_count % args.log_every == 0:
                        dt = time.time() - t0
                        print(f"[{line_count:,} lines] elapsed={dt:,.1f}s", file=sys.stderr)
        finally:
            if out_f is not None:
                out_f.close()

        ent_sizes_sorted = sorted(ent_sizes)
        edge_sizes_sorted = sorted(edge_sizes)
        ent_covs_sorted = sorted(ent_covs)
        edge_covs_sorted = sorted(edge_covs)

        print(f"lines: {line_count}")
        print(f"khop_entities_mean: {mean(ent_sizes):.2f}" if ent_sizes else "khop_entities_mean: 0")
        print(f"khop_entities_p50: {_pctl(ent_sizes_sorted, 50):.2f}")
        print(f"khop_entities_p90: {_pctl(ent_sizes_sorted, 90):.2f}")
        print(f"khop_entities_max: {ent_sizes_sorted[-1] if ent_sizes_sorted else 0}")

        print(f"khop_tuples_mean: {mean(edge_sizes):.2f}" if edge_sizes else "khop_tuples_mean: 0")
        print(f"khop_tuples_p50: {_pctl(edge_sizes_sorted, 50):.2f}")
        print(f"khop_tuples_p90: {_pctl(edge_sizes_sorted, 90):.2f}")
        print(f"khop_tuples_max: {edge_sizes_sorted[-1] if edge_sizes_sorted else 0}")

        print(f"entity_coverage_mean: {mean(ent_covs):.4f}" if ent_covs else "entity_coverage_mean: 0")
        print(f"entity_coverage_p50: {_pctl(ent_covs_sorted, 50):.4f}")
        print(f"entity_coverage_p90: {_pctl(ent_covs_sorted, 90):.4f}")

        print(f"tuple_coverage_mean: {mean(edge_covs):.4f}" if edge_covs else "tuple_coverage_mean: 0")
        print(f"tuple_coverage_p50: {_pctl(edge_covs_sorted, 50):.4f}")
        print(f"tuple_coverage_p90: {_pctl(edge_covs_sorted, 90):.4f}")

        if answer_total:
            print(f"answer_coverage_pct: {100.0 * answer_hit / answer_total:.2f}")
            print(f"answer_coverage_n: {answer_hit}/{answer_total}")
        else:
            print("answer_coverage_pct: 0.00")
            print("answer_coverage_n: 0/0")

        dt = time.time() - t0
        print(f"elapsed_sec: {dt:.2f}", file=sys.stderr)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

