#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
import time


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    conn.execute("PRAGMA mmap_size=268435456;")  # 256MB

    conn.execute("CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY) WITHOUT ROWID;")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tuples (s INTEGER, r INTEGER, o INTEGER, PRIMARY KEY (s, r, o)) WITHOUT ROWID;"
    )
    conn.commit()


def _export_json(conn: sqlite3.Connection, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write('{"entities":[')
        first = True
        for (eid,) in conn.execute("SELECT id FROM entities ORDER BY id;"):
            if not first:
                f_out.write(",")
            f_out.write(str(eid))
            first = False

        f_out.write('],"tuples":[')
        first = True
        for s, r, o in conn.execute("SELECT s, r, o FROM tuples ORDER BY s, r, o;"):
            if not first:
                f_out.write(",")
            f_out.write(f"[{s},{r},{o}]")
            first = False
        f_out.write("]}\n")


def merge_subgraphs_jsonl(
    train_jsonl_path: str,
    out_json_path: str,
    db_path: str,
    commit_every: int = 1000,
    log_every: int = 5000,
) -> None:
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    try:
        _init_db(conn)

        ent_rows: list[tuple[int]] = []
        tpl_rows: list[tuple[int, int, int]] = []

        line_count = 0
        ent_seen = 0
        tpl_seen = 0

        with open(train_jsonl_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sg = obj.get("subgraph") or {}

                entities = sg.get("entities") or []
                tuples = sg.get("tuples") or []

                ent_rows.extend((int(e),) for e in entities)
                tpl_rows.extend((int(s), int(r), int(o)) for s, r, o in tuples)

                ent_seen += len(entities)
                tpl_seen += len(tuples)
                line_count += 1

                if line_count % commit_every == 0:
                    conn.executemany("INSERT OR IGNORE INTO entities(id) VALUES (?);", ent_rows)
                    conn.executemany("INSERT OR IGNORE INTO tuples(s, r, o) VALUES (?, ?, ?);", tpl_rows)
                    conn.commit()
                    ent_rows.clear()
                    tpl_rows.clear()

                if line_count % log_every == 0:
                    dt = time.time() - t0
                    print(
                        f"[{line_count:,} lines] seen entities={ent_seen:,} tuples={tpl_seen:,} elapsed={dt:,.1f}s",
                        file=sys.stderr,
                    )

        if ent_rows or tpl_rows:
            conn.executemany("INSERT OR IGNORE INTO entities(id) VALUES (?);", ent_rows)
            conn.executemany("INSERT OR IGNORE INTO tuples(s, r, o) VALUES (?, ?, ?);", tpl_rows)
            conn.commit()

        print("exporting merged subgraph ...", file=sys.stderr)
        _export_json(conn, out_json_path)

        dt = time.time() - t0
        ent_unique = conn.execute("SELECT COUNT(*) FROM entities;").fetchone()[0]
        tpl_unique = conn.execute("SELECT COUNT(*) FROM tuples;").fetchone()[0]
        print(
            f"done: lines={line_count:,} unique_entities={ent_unique:,} unique_tuples={tpl_unique:,} elapsed={dt:,.1f}s",
            file=sys.stderr,
        )
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge all CWQ train.json subgraphs (JSONL) into one big subgraph.json."
    )
    ap.add_argument("--train", default="data/CWQ/train.json", help="Path to CWQ train.json (JSONL)")
    ap.add_argument("--out", default="data/CWQ/subgraph.json", help="Output path for merged subgraph.json")
    ap.add_argument(
        "--db",
        default="data/CWQ/subgraph_merge.sqlite",
        help="SQLite path used for on-disk deduplication (can be large)",
    )
    ap.add_argument("--commit-every", type=int, default=1000, help="Commit every N lines")
    ap.add_argument("--log-every", type=int, default=5000, help="Log progress every N lines")
    args = ap.parse_args()

    merge_subgraphs_jsonl(
        train_jsonl_path=args.train,
        out_json_path=args.out,
        db_path=args.db,
        commit_every=args.commit_every,
        log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

