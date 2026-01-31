#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from typing import Iterable


def _scan_json_value_end(s: str, start: int) -> int:
    n = len(s)
    if start >= n:
        return start
    ch = s[start]

    if ch == '"':
        i = start + 1
        esc = False
        while i < n:
            c = s[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                return i + 1
            i += 1
        raise ValueError("unterminated JSON string")

    if ch in "{[":
        open_ch, close_ch = ("{", "}") if ch == "{" else ("[", "]")
        depth = 0
        i = start
        in_str = False
        esc = False
        while i < n:
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                i += 1
                continue

            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise ValueError("unterminated JSON container")

    i = start
    while i < n and s[i] not in ",}":
        i += 1
    return i


def _strip_top_level_key_json_line(line: str, key: str) -> str:
    key_token = '"' + key + '"'
    n = len(line)
    i = 0
    in_str = False
    esc = False
    while i < n:
        c = line[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue

        if c == '"':
            if line.startswith(key_token, i):
                j = i + len(key_token)
                while j < n and line[j].isspace():
                    j += 1
                if j >= n or line[j] != ":":
                    in_str = True
                    i += 1
                    continue

                k = j + 1
                while k < n and line[k].isspace():
                    k += 1
                val_end = _scan_json_value_end(line, k)

                t = val_end
                while t < n and line[t].isspace():
                    t += 1
                has_trailing_comma = t < n and line[t] == ","

                p = i - 1
                while p >= 0 and line[p].isspace():
                    p -= 1
                has_preceding_comma = p >= 0 and line[p] == ","

                if has_preceding_comma:
                    remove_start = p
                    remove_end = val_end
                else:
                    remove_start = i
                    remove_end = (t + 1) if has_trailing_comma else val_end

                return line[:remove_start] + line[remove_end:]

            in_str = True
            i += 1
            continue

        i += 1
    return line


_CONN = None
_HOPS = None
_MAX_ENTITIES = None
_MAX_TUPLES = None
_SQL_LIMIT = None


def _init_worker(db_path: str, hops: int, max_entities: int, max_tuples: int, sql_limit: int) -> None:
    global _CONN, _HOPS, _MAX_ENTITIES, _MAX_TUPLES, _SQL_LIMIT
    _HOPS = int(hops)
    _MAX_ENTITIES = int(max_entities)
    _MAX_TUPLES = int(max_tuples)
    _SQL_LIMIT = int(sql_limit)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("CREATE INDEX IF NOT EXISTS tuples_by_o ON tuples(o, r, s);")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS frontier (id INTEGER PRIMARY KEY) WITHOUT ROWID;")
    conn.commit()
    _CONN = conn


def _extract_subgraph_from_merged(seed_entities: list[int]):
    if not seed_entities:
        return {"entities": [], "tuples": []}

    conn = _CONN
    visited = set(int(e) for e in seed_entities)
    tuples = set()

    conn.execute("DELETE FROM frontier;")
    conn.executemany("INSERT OR IGNORE INTO frontier(id) VALUES (?);", [(int(e),) for e in visited])

    for _ in range(_HOPS):
        if len(tuples) >= _MAX_TUPLES:
            break
        remaining = _MAX_TUPLES - len(tuples)
        lim = remaining if remaining < _SQL_LIMIT else _SQL_LIMIT
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
            t = (s, r, o)
            if t in tuples:
                continue

            missing = []
            if s not in visited:
                missing.append(s)
            if o not in visited:
                missing.append(o)
            if missing and len(visited) + len(missing) > _MAX_ENTITIES:
                continue

            for m in missing:
                visited.add(m)
                if m not in new_nodes_set:
                    new_nodes_set.add(m)
                    new_nodes.append(m)

            tuples.add(t)
            if len(tuples) >= _MAX_TUPLES:
                break

        if not new_nodes:
            break
        conn.execute("DELETE FROM frontier;")
        conn.executemany("INSERT OR IGNORE INTO frontier(id) VALUES (?);", [(int(e),) for e in new_nodes])

    return {"entities": list(visited), "tuples": [list(t) for t in tuples]}


def _process_line(stripped_line: str) -> str:
    obj = json.loads(stripped_line)
    seed_key = "entities_cid" if "entities_cid" in obj else "entities"
    seeds = obj.get(seed_key, []) or []
    seed_entities = [int(e) for e in seeds if isinstance(e, int)]
    obj["subgraph"] = _extract_subgraph_from_merged(seed_entities)
    return json.dumps(obj, ensure_ascii=False)


def _iter_stripped_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            yield _strip_top_level_key_json_line(line, "subgraph")


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute per-sample k-hop subgraphs from a merged DB and write a new JSONL file.")
    ap.add_argument("--input", required=True, help="Input JSONL file (e.g., data/CWQ/train.json)")
    ap.add_argument("--output", required=True, help="Output JSONL file (e.g., data/CWQ/train_khop.json)")
    ap.add_argument("--db", required=True, help="Merged SQLite DB (e.g., data/CWQ/subgraph_train.sqlite)")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-entities", type=int, default=800)
    ap.add_argument("--max-tuples", type=int, default=4000)
    ap.add_argument("--sql-limit", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    if not os.path.exists(args.db):
        raise FileNotFoundError(args.db)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    n = 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.db, args.hops, args.max_entities, args.max_tuples, args.sql_limit),
        ) as pool:
            for out_line in pool.imap(_process_line, _iter_stripped_lines(args.input), chunksize=args.chunksize):
                f_out.write(out_line + "\n")
                n += 1
                if args.log_every and n % args.log_every == 0:
                    dt = time.time() - t0
                    speed = n / dt if dt > 0 else 0.0
                    print(f"[{n:,}] elapsed={dt:,.1f}s speed={speed:.2f} lines/s", file=sys.stderr)

    dt = time.time() - t0
    speed = n / dt if dt > 0 else 0.0
    print(f"done: lines={n:,} elapsed={dt:,.1f}s speed={speed:.2f} lines/s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

