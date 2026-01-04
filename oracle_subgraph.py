import os
import pickle
from collections import OrderedDict
import random


_KB_CACHE = {}


def _infer_kb_format(kb_path, kb_format):
    if kb_format and kb_format != "auto":
        return kb_format
    _, ext = os.path.splitext(kb_path)
    if ext.lower() in [".pkl", ".pickle"]:
        return "adj"
    return "triples"


class KBGraph:
    def __init__(self, kb_path, kb_format="auto", entity2id=None, relation2id=None):
        if not kb_path:
            raise ValueError("kb_path must be set when using oracle_min_subgraph.")
        self.kb_path = kb_path
        self.kb_format = _infer_kb_format(kb_path, kb_format)
        self.entity2id = entity2id or {}
        self.relation2id = relation2id or {}
        self.adj = self._load_adj()

    def _to_entity_id(self, ent):
        if ent is None:
            return None
        if isinstance(ent, dict) and "text" in ent:
            ent = ent["text"]
        if isinstance(ent, int):
            return ent
        if ent in self.entity2id:
            return self.entity2id[ent]
        try:
            return int(ent)
        except (TypeError, ValueError):
            return None

    def _to_rel_id(self, rel):
        if rel is None:
            return None
        if isinstance(rel, dict) and "text" in rel:
            rel = rel["text"]
        if isinstance(rel, int):
            return rel
        if rel in self.relation2id:
            return self.relation2id[rel]
        try:
            return int(rel)
        except (TypeError, ValueError):
            return None

    def _load_adj(self):
        if self.kb_format == "adj":
            with open(self.kb_path, "rb") as f_in:
                raw = pickle.load(f_in)
            if isinstance(raw, list):
                # List of triples.
                raw_triples = raw
            elif isinstance(raw, dict):
                raw_triples = None
            else:
                raise ValueError("Unsupported adjacency list format for kb_path.")

            adj = {}
            if raw_triples is not None:
                for tpl in raw_triples:
                    if len(tpl) < 3:
                        continue
                    head_id = self._to_entity_id(tpl[0])
                    rel_id = self._to_rel_id(tpl[1])
                    tail_id = self._to_entity_id(tpl[2])
                    if head_id is None or rel_id is None or tail_id is None:
                        continue
                    adj.setdefault(head_id, []).append((rel_id, tail_id))
                return adj

            for head, edges in raw.items():
                head_id = self._to_entity_id(head)
                if head_id is None:
                    continue
                for edge in edges:
                    if len(edge) < 2:
                        continue
                    rel_id = self._to_rel_id(edge[0])
                    tail_id = self._to_entity_id(edge[1])
                    if rel_id is None or tail_id is None:
                        continue
                    adj.setdefault(head_id, []).append((rel_id, tail_id))
            return adj

        if self.kb_format == "triples":
            adj = {}
            with open(self.kb_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        parts = line.split()
                    if len(parts) < 3:
                        continue
                    head_id = self._to_entity_id(parts[0])
                    rel_id = self._to_rel_id(parts[1])
                    tail_id = self._to_entity_id(parts[2])
                    if head_id is None or rel_id is None or tail_id is None:
                        continue
                    adj.setdefault(head_id, []).append((rel_id, tail_id))
            return adj

        raise ValueError("Unknown kb_format: {}".format(self.kb_format))

    def to_entity_id(self, ent):
        return self._to_entity_id(ent)

    def get_neighbors(self, head_id):
        return self.adj.get(head_id, [])


def get_kb_graph(kb_path, kb_format, entity2id, relation2id):
    key = (kb_path, kb_format, len(entity2id), len(relation2id))
    if key in _KB_CACHE:
        return _KB_CACHE[key]
    kb = KBGraph(kb_path, kb_format=kb_format, entity2id=entity2id, relation2id=relation2id)
    _KB_CACHE[key] = kb
    return kb


class OracleSubgraphBuilder:
    def __init__(
        self,
        kb_graph,
        max_hop=3,
        fallback_mode="extend",
        fallback_max_hop=4,
        answer_strategy="random",
        pad_neighbors=0,
        pad_max_degree=0,
        pad_max_facts=0,
        cache_size=0,
        seed=0,
    ):
        self.kb = kb_graph
        self.max_hop = max_hop
        self.fallback_mode = fallback_mode
        self.fallback_max_hop = fallback_max_hop
        self.answer_strategy = answer_strategy
        self.pad_neighbors = pad_neighbors
        self.pad_max_degree = pad_max_degree
        self.pad_max_facts = pad_max_facts
        self.cache_size = cache_size
        self.rng = random.Random(seed)
        self._bfs_cache = OrderedDict()

    def _normalize_entities(self, entities):
        mapped = []
        for ent in entities:
            ent_id = self.kb.to_entity_id(ent)
            if ent_id is not None:
                mapped.append(ent_id)
        return mapped

    def _bfs_early(self, start_nodes, answers_set, max_hop):
        if not start_nodes or not answers_set:
            return [], {}, None
        start_set = set(start_nodes)
        if answers_set & start_set:
            return list(answers_set & start_set), {}, 0
        visited = set(start_nodes)
        parent = {}
        frontier = list(start_nodes)
        depth = 0
        while depth < max_hop:
            next_frontier = []
            found = []
            for node in frontier:
                for rel_id, tail_id in self.kb.get_neighbors(node):
                    if tail_id in visited:
                        continue
                    visited.add(tail_id)
                    parent[tail_id] = (node, rel_id)
                    if tail_id in answers_set:
                        found.append(tail_id)
                    if depth + 1 < max_hop:
                        next_frontier.append(tail_id)
            if found:
                return found, parent, depth + 1
            frontier = next_frontier
            depth += 1
        return [], parent, None

    def _bfs_full(self, start_node, max_hop):
        parent = {}
        depth = {start_node: 0}
        frontier = [start_node]
        for d in range(max_hop):
            next_frontier = []
            for node in frontier:
                for rel_id, tail_id in self.kb.get_neighbors(node):
                    if tail_id in depth:
                        continue
                    depth[tail_id] = d + 1
                    parent[tail_id] = (node, rel_id)
                    if d + 1 < max_hop:
                        next_frontier.append(tail_id)
            frontier = next_frontier
        return parent, depth

    def _bfs_cached(self, start_nodes, answers_set, max_hop):
        if len(start_nodes) != 1 or self.cache_size <= 0:
            return None
        start_node = start_nodes[0]
        key = (start_node, max_hop)
        if key in self._bfs_cache:
            self._bfs_cache.move_to_end(key)
            parent, depth = self._bfs_cache[key]
        else:
            parent, depth = self._bfs_full(start_node, max_hop)
            self._bfs_cache[key] = (parent, depth)
            if len(self._bfs_cache) > self.cache_size:
                self._bfs_cache.popitem(last=False)

        if start_node in answers_set:
            return [start_node], parent, 0
        found = [a for a in answers_set if a in depth]
        if not found:
            return [], parent, None
        min_len = min(depth[a] for a in found)
        answers_at_min = [a for a in found if depth[a] == min_len]
        return answers_at_min, parent, min_len

    def _reconstruct_path(self, answer, parent, start_set):
        edges = []
        nodes = set()
        curr = answer
        nodes.add(curr)
        while curr not in start_set:
            if curr not in parent:
                return [], set()
            prev, rel_id = parent[curr]
            edges.append((prev, rel_id, curr))
            nodes.add(prev)
            curr = prev
        edges.reverse()
        return edges, nodes

    def _pad_neighbors(self, node_set, edge_set):
        if self.pad_neighbors <= 0:
            return node_set, edge_set
        nodes = set(node_set)
        edges = set(edge_set)
        total_cap = self.pad_max_facts if self.pad_max_facts > 0 else None
        for node_id in list(node_set):
            neighbors = list(self.kb.get_neighbors(node_id))
            if not neighbors:
                continue
            if self.pad_max_degree > 0 and len(neighbors) > self.pad_max_degree:
                neighbors = self.rng.sample(neighbors, self.pad_max_degree)
            if self.pad_neighbors > 0 and len(neighbors) > self.pad_neighbors:
                neighbors = self.rng.sample(neighbors, self.pad_neighbors)
            for rel_id, tail_id in neighbors:
                edge = (node_id, rel_id, tail_id)
                if edge in edges:
                    continue
                edges.add(edge)
                nodes.add(tail_id)
                if total_cap is not None and len(edges) >= total_cap:
                    return nodes, edges
        return nodes, edges

    def build(self, topic_entities, answers_raw):
        meta = {
            "found": False,
            "path_len": None,
            "answer_total": len(answers_raw),
            "answer_covered": 0,
            "answer_any": False,
            "used_fallback": False,
        }
        start_nodes = self._normalize_entities(topic_entities)
        answers = self._normalize_entities(answers_raw)
        if not start_nodes or not answers:
            return None, meta

        answers_set = set(answers)
        start_set = set(start_nodes)
        if answers_set & start_set:
            answers_at_min = list(answers_set & start_set)
            parent = {}
            path_len = 0
        else:
            cached = self._bfs_cached(start_nodes, answers_set, self.max_hop)
            if cached is None:
                answers_at_min, parent, path_len = self._bfs_early(
                    start_nodes, answers_set, self.max_hop
                )
            else:
                answers_at_min, parent, path_len = cached

            if not answers_at_min and self.fallback_mode == "extend":
                if self.fallback_max_hop > self.max_hop:
                    meta["used_fallback"] = True
                    cached = self._bfs_cached(start_nodes, answers_set, self.fallback_max_hop)
                    if cached is None:
                        answers_at_min, parent, path_len = self._bfs_early(
                            start_nodes, answers_set, self.fallback_max_hop
                        )
                    else:
                        answers_at_min, parent, path_len = cached

        if not answers_at_min:
            return None, meta

        if self.answer_strategy == "random":
            if len(answers_at_min) > 1:
                answers_sel = [self.rng.choice(answers_at_min)]
            else:
                answers_sel = answers_at_min
        elif self.answer_strategy == "union":
            answers_sel = answers_at_min
        else:
            answers_sel = answers_at_min[:1]

        edge_set = set()
        node_set = set()
        if path_len == 0:
            node_set.update(answers_sel)
        else:
            for ans in answers_sel:
                edges, nodes = self._reconstruct_path(ans, parent, start_set)
                edge_set.update(edges)
                node_set.update(nodes)

        node_set, edge_set = self._pad_neighbors(node_set, edge_set)

        tuples = sorted(edge_set, key=lambda x: (x[0], x[1], x[2]))
        entities = sorted(node_set)
        subgraph = {
            "entities": entities,
            "tuples": [[h, r, t] for (h, r, t) in tuples],
        }

        covered = sum(1 for a in answers if a in node_set)
        meta["found"] = True
        meta["path_len"] = path_len
        meta["answer_covered"] = covered
        meta["answer_any"] = covered > 0
        return subgraph, meta
