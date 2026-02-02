#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import deque
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
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _extract_entity_value(v):
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "kb_id" in v:
            return v["kb_id"]
        if "text" in v:
            return v["text"]
    return None


def _init_entities_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entities (name TEXT PRIMARY KEY, id INTEGER) WITHOUT ROWID;"
    )
    conn.commit()


def build_entities_sqlite(entities_txt_path: str, sqlite_path: str, *, commit_every: int = 50000) -> None:
    if not os.path.exists(entities_txt_path):
        raise FileNotFoundError(entities_txt_path)
    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)

    conn = sqlite3.connect(sqlite_path)
    try:
        _init_entities_sqlite(conn)
        rows = []
        n = 0
        t0 = time.time()
        with open(entities_txt_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if not name:
                    continue
                rows.append((name, int(i)))
                if len(rows) >= commit_every:
                    conn.executemany("INSERT OR REPLACE INTO entities(name, id) VALUES (?, ?);", rows)
                    conn.commit()
                    n += len(rows)
                    rows.clear()
                    dt = time.time() - t0
                    speed = n / dt if dt > 0 else 0.0
                    print(f"[entities sqlite] inserted={n:,} elapsed={dt:,.1f}s speed={speed:,.0f}/s", file=sys.stderr)
        if rows:
            conn.executemany("INSERT OR REPLACE INTO entities(name, id) VALUES (?, ?);", rows)
            conn.commit()
            n += len(rows)
        conn.execute("ANALYZE;")
        conn.commit()
        dt = time.time() - t0
        print(f"done building entities sqlite: rows={n:,} elapsed={dt:,.1f}s", file=sys.stderr)
    finally:
        conn.close()


class EntityIdMapper:
    def __init__(self, *, entities_txt: str | None, entities_sqlite: str | None, build_sqlite: bool):
        self._conn = None
        self._cur = None
        self._entity2id = None
        self._mode = "none"

        if entities_sqlite:
            if not os.path.exists(entities_sqlite):
                if not build_sqlite:
                    raise FileNotFoundError(
                        f"entities sqlite not found: {entities_sqlite} (pass --build-entities-sqlite to create it)"
                    )
                if not entities_txt:
                    raise ValueError("--entities-txt is required to build --entities-sqlite")
                build_entities_sqlite(entities_txt_path=entities_txt, sqlite_path=entities_sqlite)

            conn = sqlite3.connect(entities_sqlite)
            conn.execute("PRAGMA journal_mode=OFF;")
            conn.execute("PRAGMA synchronous=OFF;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA cache_size=-200000;")
            conn.execute("PRAGMA mmap_size=268435456;")
            self._conn = conn
            self._cur = conn.cursor()
            self._mode = "sqlite"
            return

        if entities_txt:
            # WARNING: this can be memory-heavy for CWQ (millions of entities).
            t0 = time.time()
            entity2id = {}
            with open(entities_txt, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    entity2id[line.strip()] = i
            dt = time.time() - t0
            print(f"loaded entities.txt into memory: {len(entity2id):,} entries in {dt:,.1f}s", file=sys.stderr)
            self._entity2id = entity2id
            self._mode = "dict"
            return

        self._mode = "none"

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._cur = None
        self._entity2id = None

    def map(self, v):
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, dict):
            v = _extract_entity_value(v)
            if v is None:
                return None
            if isinstance(v, int):
                return int(v)
        if not isinstance(v, str):
            return None

        if self._mode == "dict":
            return self._entity2id.get(v)
        if self._mode == "sqlite":
            self._cur.execute("SELECT id FROM entities WHERE name = ?;", (v,))
            row = self._cur.fetchone()
            return int(row[0]) if row else None
        return None


def _answer_ids(sample, mapper: EntityIdMapper) -> set[int]:
    answers = sample.get("answers_cid", None)
    if answers is None:
        answers = sample.get("answers", [])
    ids = set()
    for a in answers:
        aid = mapper.map(a)
        if isinstance(aid, int):
            ids.add(aid)
    return ids


def _seed_ids(sample, mapper: EntityIdMapper) -> list[int]:
    seed_key = "entities_cid" if "entities_cid" in sample else "entities"
    seeds = sample.get(seed_key, []) or []
    out = []
    for s in seeds:
        sid = mapper.map(s)
        if isinstance(sid, int):
            out.append(sid)
    return out


def _reachable_undirected_union_find(entities: list, tuples: list, seeds: list[int], answers: set[int]) -> bool:
    if not seeds or not answers:
        return False

    idx: dict[int, int] = {}
    parent: list[int] = []
    rank: list[int] = []

    def add_node(n: int) -> int:
        i = idx.get(n)
        if i is not None:
            return i
        i = len(parent)
        idx[n] = i
        parent.append(i)
        rank.append(0)
        return i

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for e in entities:
        try:
            add_node(int(e))
        except Exception:
            continue

    for s, _, o in tuples:
        try:
            si = add_node(int(s))
            oi = add_node(int(o))
        except Exception:
            continue
        union(si, oi)

    seed_roots = set()
    for s in seeds:
        si = idx.get(int(s))
        if si is not None:
            seed_roots.add(find(si))

    if not seed_roots:
        return False

    for a in answers:
        ai = idx.get(int(a))
        if ai is None:
            continue
        if find(ai) in seed_roots:
            return True
    return False


def _reachable_directed_bfs(entities: list, tuples: list, seeds: list[int], answers: set[int]) -> bool:
    if not seeds or not answers:
        return False

    idx: dict[int, int] = {}

    def add_node(n: int) -> int:
        i = idx.get(n)
        if i is not None:
            return i
        i = len(idx)
        idx[n] = i
        return i

    for e in entities:
        try:
            add_node(int(e))
        except Exception:
            continue

    edges: list[tuple[int, int]] = []
    for s, _, o in tuples:
        try:
            si = add_node(int(s))
            oi = add_node(int(o))
        except Exception:
            continue
        edges.append((si, oi))

    n = len(idx)
    if n == 0:
        return False

    adj = [[] for _ in range(n)]
    for si, oi in edges:
        adj[si].append(oi)

    ans_idx = {idx[a] for a in answers if a in idx}
    if not ans_idx:
        return False

    q = deque()
    seen = [False] * n
    for s in seeds:
        si = idx.get(int(s))
        if si is None or seen[si]:
            continue
        seen[si] = True
        q.append(si)

    while q:
        u = q.popleft()
        if u in ans_idx:
            return True
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute how often answers are reachable from seed entities within each sample subgraph (JSONL)."
    )
    ap.add_argument("--data", required=True, help="Input JSONL (e.g., data/CWQ/train_khop.json)")
    ap.add_argument("--entities-txt", default=None, help="Optional entities.txt (for mapping kb_id->int)")
    ap.add_argument("--entities-sqlite", default=None, help="Optional sqlite index for entities (name->id)")
    ap.add_argument(
        "--build-entities-sqlite",
        action="store_true",
        help="If --entities-sqlite is missing, build it from --entities-txt",
    )
    ap.add_argument("--mode", default="undirected", choices=["undirected", "directed", "both"])
    ap.add_argument("--max-lines", type=int, default=0, help="0 = all lines, else stop after N")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--heartbeat-sec", type=float, default=10.0)
    ap.add_argument("--out-jsonl", default=None, help="Optional per-sample output (JSONL)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)

    mapper = EntityIdMapper(
        entities_txt=args.entities_txt,
        entities_sqlite=args.entities_sqlite,
        build_sqlite=bool(args.build_entities_sqlite),
    )

    t0 = time.time()
    line_count = 0
    answerable = 0
    in_subgraph = 0
    reachable_undir = 0
    reachable_dir = 0
    missing_answer_id = 0

    ent_sizes = []
    tpl_sizes = []

    next_heartbeat = time.time()

    out_f = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    try:
        with open(args.data, "r", encoding="utf-8") as f_in:
            for raw in f_in:
                if not raw.strip():
                    continue
                obj = json.loads(raw)
                sg = obj.get("subgraph") or {}
                entities = sg.get("entities") or []
                tuples = sg.get("tuples") or []

                ent_sizes.append(len(entities))
                tpl_sizes.append(len(tuples))

                seeds = _seed_ids(obj, mapper)
                ans_ids = _answer_ids(obj, mapper)
                if not ans_ids:
                    missing_answer_id += 1
                else:
                    answerable += 1
                    ent_set = set()
                    for e in entities:
                        try:
                            ent_set.add(int(e))
                        except Exception:
                            continue
                    hit = bool(ans_ids & ent_set)
                    if hit:
                        in_subgraph += 1

                    is_reachable_undir = False
                    is_reachable_dir = False
                    if args.mode in ("undirected", "both") and hit:
                        is_reachable_undir = _reachable_undirected_union_find(entities, tuples, seeds, ans_ids)
                        if is_reachable_undir:
                            reachable_undir += 1

                    if args.mode in ("directed", "both") and hit:
                        is_reachable_dir = _reachable_directed_bfs(entities, tuples, seeds, ans_ids)
                        if is_reachable_dir:
                            reachable_dir += 1

                if out_f is not None:
                    out_obj = {
                        "id": obj.get("id"),
                        "seed_n": len(seeds),
                        "answer_n": len(ans_ids),
                        "subgraph_entities": len(entities),
                        "subgraph_tuples": len(tuples),
                    }
                    if ans_ids:
                        out_obj["answer_in_subgraph"] = bool(ans_ids & ent_set)
                    if args.mode in ("undirected", "both") and ans_ids:
                        out_obj["answer_reachable_undirected"] = bool(is_reachable_undir)
                    if args.mode in ("directed", "both") and ans_ids:
                        out_obj["answer_reachable_directed"] = bool(is_reachable_dir)
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                line_count += 1
                if args.max_lines and line_count >= args.max_lines:
                    break

                now = time.time()
                if args.log_every and line_count % args.log_every == 0:
                    dt = now - t0
                    speed = line_count / dt if dt > 0 else 0.0
                    print(f"[{line_count:,}] elapsed={dt:,.1f}s speed={speed:.2f} lines/s", file=sys.stderr)
                    next_heartbeat = now
                elif args.heartbeat_sec and now - next_heartbeat >= args.heartbeat_sec:
                    dt = now - t0
                    speed = line_count / dt if dt > 0 else 0.0
                    print(f"[{line_count:,}] elapsed={dt:,.1f}s speed={speed:.2f} lines/s", file=sys.stderr)
                    next_heartbeat = now
    finally:
        if out_f is not None:
            out_f.close()
        mapper.close()

    ent_sizes_sorted = sorted(ent_sizes)
    tpl_sizes_sorted = sorted(tpl_sizes)

    print(f"lines: {line_count}")
    print(f"answerable_lines: {answerable}")
    print(f"missing_answer_id_lines: {missing_answer_id}")
    if ent_sizes:
        print(f"subgraph_entities_mean: {mean(ent_sizes):.2f}")
        print(f"subgraph_entities_p50: {_pctl(ent_sizes_sorted, 50):.2f}")
        print(f"subgraph_entities_p90: {_pctl(ent_sizes_sorted, 90):.2f}")
        print(f"subgraph_entities_max: {ent_sizes_sorted[-1]}")
    if tpl_sizes:
        print(f"subgraph_tuples_mean: {mean(tpl_sizes):.2f}")
        print(f"subgraph_tuples_p50: {_pctl(tpl_sizes_sorted, 50):.2f}")
        print(f"subgraph_tuples_p90: {_pctl(tpl_sizes_sorted, 90):.2f}")
        print(f"subgraph_tuples_max: {tpl_sizes_sorted[-1]}")

    if answerable:
        print(f"answer_in_subgraph_pct: {100.0 * in_subgraph / answerable:.2f}")
        print(f"answer_in_subgraph_n: {in_subgraph}/{answerable}")

        if args.mode in ("undirected", "both"):
            print(f"answer_reachable_undirected_pct: {100.0 * reachable_undir / answerable:.2f}")
            print(f"answer_reachable_undirected_n: {reachable_undir}/{answerable}")
        if args.mode in ("directed", "both"):
            print(f"answer_reachable_directed_pct: {100.0 * reachable_dir / answerable:.2f}")
            print(f"answer_reachable_directed_n: {reachable_dir}/{answerable}")
    else:
        print("answer_in_subgraph_pct: 0.00")
        print("answer_in_subgraph_n: 0/0")
        if args.mode in ("undirected", "both"):
            print("answer_reachable_undirected_pct: 0.00")
            print("answer_reachable_undirected_n: 0/0")
        if args.mode in ("directed", "both"):
            print("answer_reachable_directed_pct: 0.00")
            print("answer_reachable_directed_n: 0/0")

    dt = time.time() - t0
    print(f"elapsed_sec: {dt:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
