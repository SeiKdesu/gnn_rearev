from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
except ImportError:  # pragma: no cover - fallback handled at runtime
    TfidfVectorizer = None
    normalize = None

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - fallback handled at runtime
    sp = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


logger = logging.getLogger(__name__)

# Connectivity augmentation parameters (config追加禁止のため定数で管理)
MAX_HOPS: Optional[int] = None  # None=unlimited
NODE_SCORE_THRESHOLD: float = 0.2
MAX_NODE_CONTEXT_RELS: int = 12
MAX_NODE_CONTEXT_NEIGHBORS: int = 12


@dataclass
class StageConfig:
    top_m: int
    top_k_rel: int
    gamma: float


@dataclass
class DenoiseConfig:
    enable: bool
    encoder_type: str
    relation_desc_path: Optional[str]
    alpha: float
    beta: float
    strict: StageConfig
    loose: StageConfig
    min_edges: float
    enable_pair_score: bool
    pair_stats_path: Optional[str]
    ensure_connectivity: bool
    encoder_model_name: Optional[str] = None


@dataclass
class Edge:
    head: Any
    rel_id: int
    tail: Any
    original: Tuple[Any, Any, Any]
    index: int
    score: float = 0.0


class PairStats:
    def __init__(self, pair_counts: Dict[Tuple[int, int], float]):
        self.pair_counts = pair_counts
        self.max_count = max(pair_counts.values()) if pair_counts else 0.0

    def score(self, r_prev: int, r_next: int) -> float:
        if self.max_count <= 0:
            return 0.0
        return self.pair_counts.get((r_prev, r_next), 0.0) / self.max_count


class BaseEncoder:
    def encode_text(self, text: str):
        raise NotImplementedError

    def get_relation_vectors(self, rel_ids: List[int], rel_text_fn):
        raise NotImplementedError

    def similarity(self, q_vec, rel_vecs) -> np.ndarray:
        raise NotImplementedError


class TfidfEncoder(BaseEncoder):
    def __init__(self, relation_texts: List[str]):
        if TfidfVectorizer is None or normalize is None or sp is None:
            raise ImportError("scikit-learn/scipy is required for TF-IDF encoding.")
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.rel_matrix = self.vectorizer.fit_transform(relation_texts)
        self.rel_matrix = normalize(self.rel_matrix)
        self._extra_cache: Dict[int, Any] = {}

    def encode_text(self, text: str):
        vec = self.vectorizer.transform([text])
        return normalize(vec)

    def get_relation_vectors(self, rel_ids: List[int], rel_text_fn):
        rows = []
        for rel_id in rel_ids:
            if 0 <= rel_id < self.rel_matrix.shape[0]:
                rows.append(self.rel_matrix[rel_id])
                continue
            if rel_id not in self._extra_cache:
                vec = self.vectorizer.transform([rel_text_fn(rel_id)])
                self._extra_cache[rel_id] = normalize(vec)
            rows.append(self._extra_cache[rel_id])
        return sp.vstack(rows)

    def similarity(self, q_vec, rel_vecs) -> np.ndarray:
        scores = q_vec.dot(rel_vecs.T)
        if sp is not None and sp.issparse(scores):
            scores = scores.toarray()
        return np.asarray(scores).ravel()


class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, relation_texts: List[str], model_name: str):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
        self.model = SentenceTransformer(model_name)
        self.rel_matrix = self.model.encode(
            relation_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._extra_cache: Dict[int, np.ndarray] = {}

    def encode_text(self, text: str):
        return self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )

    def get_relation_vectors(self, rel_ids: List[int], rel_text_fn):
        rows = []
        for rel_id in rel_ids:
            if 0 <= rel_id < self.rel_matrix.shape[0]:
                rows.append(self.rel_matrix[rel_id])
                continue
            if rel_id not in self._extra_cache:
                desc = rel_text_fn(rel_id)
                self._extra_cache[rel_id] = self.model.encode(
                    [desc], convert_to_numpy=True, normalize_embeddings=True
                )[0]
            rows.append(self._extra_cache[rel_id])
        return np.vstack(rows)

    def similarity(self, q_vec, rel_vecs) -> np.ndarray:
        return np.dot(q_vec, rel_vecs.T).ravel()


def build_denoise_config(config: Dict[str, Any]) -> DenoiseConfig:
    enable = bool(config.get("enable_denoise", False))
    encoder_type = config.get("encoder_type", "tfidf")
    relation_desc_path = config.get("relation_desc_path")
    alpha = float(config.get("alpha", 1.0))
    beta = float(config.get("beta", 1.0))
    gamma_default = float(config.get("gamma", 0.2))
    strict = StageConfig(
        top_m=int(config.get("topM_strict", 10)),
        top_k_rel=int(config.get("topK_strict", 50)),
        gamma=float(config.get("gamma_strict", gamma_default)),
    )
    loose = StageConfig(
        top_m=int(config.get("topM_loose", 25)),
        top_k_rel=int(config.get("topK_loose", 150)),
        gamma=float(config.get("gamma_loose", 0.05)),
    )
    min_edges = float(config.get("min_edges", 50))
    enable_pair_score = bool(config.get("enable_pair_score", False))
    pair_stats_path = config.get("pair_stats_path")
    ensure_connectivity = bool(config.get("ensure_connectivity", True))
    encoder_model_name = config.get("encoder_model_name")
    return DenoiseConfig(
        enable=enable,
        encoder_type=encoder_type,
        relation_desc_path=relation_desc_path,
        alpha=alpha,
        beta=beta,
        strict=strict,
        loose=loose,
        min_edges=min_edges,
        enable_pair_score=enable_pair_score,
        pair_stats_path=pair_stats_path,
        ensure_connectivity=ensure_connectivity,
        encoder_model_name=encoder_model_name,
    )


def normalize_relation_text(text: str) -> str:
    return text.replace(".", " ").replace("_", " ").strip()


def _parse_rel_key(rel_key: Any, relation2id: Dict[str, int]) -> Optional[int]:
    if isinstance(rel_key, int):
        return rel_key
    if isinstance(rel_key, str):
        if rel_key in relation2id:
            return relation2id[rel_key]
        try:
            return int(rel_key)
        except ValueError:
            return None
    return None


def load_relation_descriptions(
    desc_path: Optional[str], relation2id: Dict[str, int]
) -> Dict[int, str]:
    id2relation = {v: k for k, v in relation2id.items()}
    descs: Dict[int, str] = {}
    if desc_path and os.path.exists(desc_path):
        try:
            if desc_path.endswith(".json"):
                with open(desc_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                if isinstance(data, dict):
                    for k, v in data.items():
                        rel_id = _parse_rel_key(k, relation2id)
                        if rel_id is not None:
                            descs[rel_id] = str(v)
                elif isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        key = item.get("relation") or item.get("rel") or item.get("id")
                        rel_id = _parse_rel_key(key, relation2id)
                        if rel_id is None:
                            continue
                        desc = item.get("desc") or item.get("description") or item.get("text")
                        if desc is not None:
                            descs[rel_id] = str(desc)
            else:
                with open(desc_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) < 2:
                            continue
                        rel_id = _parse_rel_key(parts[0], relation2id)
                        if rel_id is None:
                            continue
                        descs[rel_id] = parts[1]
        except Exception as exc:
            logger.warning("Failed to load relation descriptions from %s: %s", desc_path, exc)

    for rel_id, rel_name in id2relation.items():
        if rel_id not in descs:
            descs[rel_id] = normalize_relation_text(rel_name)
    return descs


def load_pair_stats(
    stats_path: Optional[str], relation2id: Dict[str, int]
) -> Optional[PairStats]:
    if not stats_path or not os.path.exists(stats_path):
        return None
    pair_counts: Dict[Tuple[int, int], float] = {}
    try:
        if stats_path.endswith(".json"):
            with open(stats_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        r_prev = _parse_rel_key(k, relation2id)
                        if r_prev is None:
                            continue
                        for k2, c in v.items():
                            r_next = _parse_rel_key(k2, relation2id)
                            if r_next is None:
                                continue
                            pair_counts[(r_prev, r_next)] = float(c)
                    else:
                        if "|||" in k:
                            k1, k2 = k.split("|||", 1)
                        elif "\t" in k:
                            k1, k2 = k.split("\t", 1)
                        else:
                            continue
                        r_prev = _parse_rel_key(k1, relation2id)
                        r_next = _parse_rel_key(k2, relation2id)
                        if r_prev is None or r_next is None:
                            continue
                        pair_counts[(r_prev, r_next)] = float(v)
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    r_prev = _parse_rel_key(item.get("r_prev"), relation2id)
                    r_next = _parse_rel_key(item.get("r_next"), relation2id)
                    if r_prev is None or r_next is None:
                        continue
                    pair_counts[(r_prev, r_next)] = float(item.get("count", 0.0))
        else:
            with open(stats_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    r_prev = _parse_rel_key(parts[0], relation2id)
                    r_next = _parse_rel_key(parts[1], relation2id)
                    if r_prev is None or r_next is None:
                        continue
                    pair_counts[(r_prev, r_next)] = float(parts[2])
    except Exception as exc:
        logger.warning("Failed to load pair stats from %s: %s", stats_path, exc)
        return None
    return PairStats(pair_counts)


def build_local_pair_stats(edges: Iterable[Edge]) -> PairStats:
    incoming: Dict[Any, List[int]] = {}
    outgoing: Dict[Any, List[int]] = {}
    for edge in edges:
        incoming.setdefault(edge.tail, []).append(edge.rel_id)
        outgoing.setdefault(edge.head, []).append(edge.rel_id)
    pair_counts: Dict[Tuple[int, int], float] = {}
    for node in set(incoming) | set(outgoing):
        for r_prev in incoming.get(node, []):
            for r_next in outgoing.get(node, []):
                pair_counts[(r_prev, r_next)] = pair_counts.get((r_prev, r_next), 0.0) + 1.0
    return PairStats(pair_counts)


class SubgraphDenoiser:
    """
    Denoise a subgraph by pruning edges using question-relation similarity and optional
    relation-pair transition scores. Returns a subgraph with the same format as input.
    """

    def __init__(self, relation2id: Dict[str, int], config: Dict[str, Any]):
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.base_relation_count = len(relation2id)
        self.config = build_denoise_config(config)
        self._raw_config = dict(config)
        self._unknown_rel_map: Dict[str, int] = {}
        self._unknown_rel_desc: Dict[int, str] = {}
        self._next_unknown_rel_id = -1
        self._entity2id_cache: Optional[Dict[str, int]] = None
        self._entity2id_loaded = False

        self.rel_descs = load_relation_descriptions(
            self.config.relation_desc_path, relation2id
        )
        relation_texts = [
            self.rel_descs.get(rel_id, self.id2relation.get(rel_id, str(rel_id)))
            for rel_id in range(self.base_relation_count)
        ]
        self.encoder = self._build_encoder(relation_texts)
        self.global_pair_stats = None
        if self.config.enable_pair_score:
            self.global_pair_stats = load_pair_stats(
                self.config.pair_stats_path, relation2id
            )

    def _load_entity2id(self) -> Optional[Dict[str, int]]:
        if self._entity2id_loaded:
            return self._entity2id_cache
        self._entity2id_loaded = True

        raw = self._raw_config.get("entity2id")
        if isinstance(raw, dict):
            self._entity2id_cache = {
                self._normalize_entity_key(k): int(v) for k, v in raw.items()
            }
            return self._entity2id_cache
        if not isinstance(raw, str) or not raw:
            return None

        resolved = raw
        if not os.path.isabs(resolved):
            data_folder = self._raw_config.get("data_folder") or ""
            if data_folder:
                resolved = os.path.join(data_folder, resolved)

        if not os.path.exists(resolved):
            logger.debug("entity2id file not found: %s", resolved)
            return None

        entity2id: Dict[str, int] = {}
        try:
            with open(resolved, "r", encoding="utf-8") as f_in:
                for idx, line in enumerate(f_in):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        ent_key = self._normalize_entity_key(parts[0])
                        try:
                            ent_id = int(parts[1])
                        except ValueError:
                            ent_id = idx
                    else:
                        ent_key = self._normalize_entity_key(line)
                        ent_id = idx
                    entity2id[ent_key] = ent_id
        except Exception as exc:
            logger.warning("Failed to load entity2id from %s: %s", resolved, exc)
            self._entity2id_cache = None
            return None

        self._entity2id_cache = entity2id
        return self._entity2id_cache

    def _to_int_entity_id(self, ent: Any) -> Optional[int]:
        if ent is None:
            return None
        if isinstance(ent, bool):
            return None
        if isinstance(ent, int):
            return ent
        if isinstance(ent, float) and ent.is_integer():
            return int(ent)
        if isinstance(ent, dict):
            if "kb_id" in ent and ent["kb_id"] is not None:
                return self._to_int_entity_id(ent["kb_id"])
            if "text" in ent and ent["text"] is not None:
                return self._to_int_entity_id(ent["text"])
            if "id" in ent and ent["id"] is not None:
                return self._to_int_entity_id(ent["id"])
            return None
        if isinstance(ent, str):
            key = self._normalize_entity_key(ent)
            try:
                return int(key)
            except ValueError:
                pass
            entity2id = self._load_entity2id()
            if entity2id is not None and key in entity2id:
                return entity2id[key]
            return None
        return None

    def _normalize_seed_ids(
        self, topic_entities: Optional[Iterable[Any]]
    ) -> Tuple[List[int], int, List[Any]]:
        if not topic_entities:
            return [], 0, []
        seeds: List[int] = []
        seen = set()
        skipped = 0
        skipped_examples: List[Any] = []
        for ent in topic_entities:
            ent_id = self._to_int_entity_id(ent)
            if ent_id is None:
                skipped += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append(ent)
                continue
            if ent_id in seen:
                continue
            seen.add(ent_id)
            seeds.append(ent_id)
        return seeds, skipped, skipped_examples

    def _build_node_context(
        self, base_tuples: List[Tuple[Any, Any, Any]]
    ) -> Tuple[Dict[int, Dict[str, Any]], List[Tuple[int, Any, int]]]:
        ctx: Dict[int, Dict[str, Any]] = {}
        base_edges_int: List[Tuple[int, Any, int]] = []
        skipped = 0
        for tpl in base_tuples:
            if len(tpl) != 3:
                continue
            sbj, rel, obj = tpl
            u = self._to_int_entity_id(sbj)
            v = self._to_int_entity_id(obj)
            if u is None or v is None:
                skipped += 1
                continue
            base_edges_int.append((u, rel, v))
            rel_id = self._relation_id(rel)

            cu = ctx.setdefault(u, {"out_rels": set(), "in_rels": set(), "nbrs": set()})
            cv = ctx.setdefault(v, {"out_rels": set(), "in_rels": set(), "nbrs": set()})
            cu["out_rels"].add(rel_id)
            cv["in_rels"].add(rel_id)
            cu["nbrs"].add(v)
            cv["nbrs"].add(u)
        if skipped:
            logger.warning(
                "Node context: skipped %d tuples with non-int-convertible endpoints.", skipped
            )
        return ctx, base_edges_int

    def _entity_text(self, eid: int, node_ctx: Dict[int, Dict[str, Any]]) -> str:
        parts: List[str] = [f"entity:{eid}"]
        info = node_ctx.get(eid)
        if not info:
            return " ".join(parts)

        out_rels = list(info.get("out_rels") or [])
        in_rels = list(info.get("in_rels") or [])
        nbrs = list(info.get("nbrs") or [])

        rel_texts: List[str] = []
        for rel_id in out_rels[: MAX_NODE_CONTEXT_RELS // 2]:
            rel_texts.append(self._relation_desc(rel_id))
        for rel_id in in_rels[: MAX_NODE_CONTEXT_RELS // 2]:
            rel_texts.append(self._relation_desc(rel_id))
        if rel_texts:
            parts.append("relations:" + " | ".join(rel_texts))

        if nbrs:
            nbrs_sorted = sorted(nbrs)[:MAX_NODE_CONTEXT_NEIGHBORS]
            parts.append("neighbors:" + " ".join(f"entity:{n}" for n in nbrs_sorted))

        return " ".join(parts)

    @staticmethod
    def _cosine_similarity(q_vec: Any, x_vec: Any) -> float:
        if sp is not None and (sp.issparse(q_vec) or sp.issparse(x_vec)):
            q = q_vec
            x = x_vec
            if not sp.issparse(q):
                q = sp.csr_matrix(np.asarray(q))
            if not sp.issparse(x):
                x = sp.csr_matrix(np.asarray(x))
            if q.shape[0] != 1:
                q = q[:1]
            if x.shape[0] != 1:
                x = x[:1]
            dot = float(q.multiply(x).sum())
            qn = math.sqrt(float(q.multiply(q).sum()))
            xn = math.sqrt(float(x.multiply(x).sum()))
            if qn <= 0.0 or xn <= 0.0:
                return 0.0
            return dot / (qn * xn)

        q = np.asarray(q_vec)
        x = np.asarray(x_vec)
        if q.ndim == 2:
            q = q[0]
        if x.ndim == 2:
            x = x[0]
        qn = float(np.linalg.norm(q))
        xn = float(np.linalg.norm(x))
        if qn <= 0.0 or xn <= 0.0:
            return 0.0
        return float(np.dot(q, x) / (qn * xn))

    def _score_nodes(
        self, question: str, node_ids: Iterable[int], node_ctx: Dict[int, Dict[str, Any]]
    ) -> Dict[int, float]:
        q_vec = self.encoder.encode_text(question)
        scores: Dict[int, float] = {}
        for eid in node_ids:
            text = self._entity_text(eid, node_ctx)
            e_vec = self.encoder.encode_text(text)
            scores[eid] = self._cosine_similarity(q_vec, e_vec)
        return scores

    @staticmethod
    def _relation_key(rel: Any) -> Any:
        if isinstance(rel, dict):
            try:
                return json.dumps(rel, sort_keys=True, ensure_ascii=False)
            except Exception:
                return str(rel)
        try:
            hash(rel)
            return rel
        except Exception:
            return str(rel)

    def _tuple_key(self, sbj: int, rel: Any, obj: int) -> Tuple[int, Any, int]:
        return (sbj, self._relation_key(rel), obj)

    def _reachable_out(self, tuples: Iterable[Tuple[int, Any, int]], seeds: List[int]) -> set:
        reachable = set(seeds)
        if not seeds:
            return reachable
        adj: Dict[int, List[int]] = {}
        for sbj, _, obj in tuples:
            adj.setdefault(sbj, []).append(obj)
        q: deque = deque(seeds)
        hops: Optional[int] = MAX_HOPS
        if hops is None:
            while q:
                node = q.popleft()
                for nbr in adj.get(node, []):
                    if nbr in reachable:
                        continue
                    reachable.add(nbr)
                    q.append(nbr)
            return reachable
        depth: Dict[int, int] = {s: 0 for s in seeds}
        while q:
            node = q.popleft()
            d = depth.get(node, 0)
            if d >= hops:
                continue
            for nbr in adj.get(node, []):
                if nbr in reachable:
                    continue
                reachable.add(nbr)
                depth[nbr] = d + 1
                q.append(nbr)
        return reachable

    def _bfs_parent_edges(
        self,
        tuples: Iterable[Tuple[Any, Any, Any]],
        seeds: List[int],
        *,
        allow_reverse: bool,
    ) -> Dict[int, Tuple[int, Any]]:
        parent: Dict[int, Tuple[int, Any]] = {}
        adj: Dict[int, List[Tuple[int, Any]]] = {}
        for tpl in tuples:
            if len(tpl) != 3:
                continue
            sbj, rel, obj = tpl
            u = self._to_int_entity_id(sbj)
            v = self._to_int_entity_id(obj)
            if u is None or v is None:
                continue
            adj.setdefault(u, []).append((v, rel))
            if allow_reverse:
                adj.setdefault(v, []).append((u, rel))

        visited = set(seeds)
        q: deque = deque(seeds)
        hops: Optional[int] = MAX_HOPS
        if hops is None:
            while q:
                node = q.popleft()
                for nbr, rel in adj.get(node, []):
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    parent[nbr] = (node, rel)
                    q.append(nbr)
            return parent

        depth: Dict[int, int] = {s: 0 for s in seeds}
        while q:
            node = q.popleft()
            d = depth.get(node, 0)
            if d >= hops:
                continue
            for nbr, rel in adj.get(node, []):
                if nbr in visited:
                    continue
                visited.add(nbr)
                parent[nbr] = (node, rel)
                depth[nbr] = d + 1
                q.append(nbr)
        return parent

    def _restore_path_edges(
        self, target: int, seeds_set: set, parent: Dict[int, Tuple[int, Any]]
    ) -> List[Tuple[int, Any, int]]:
        path_edges: List[Tuple[int, Any, int]] = []
        node = target
        guard = 0
        while node not in seeds_set:
            if node not in parent:
                return []
            prev, rel = parent[node]
            path_edges.append((prev, rel, node))
            node = prev
            guard += 1
            if guard > 1_000_000:
                raise RuntimeError("Connectivity augment: parent loop detected.")
        path_edges.reverse()
        return path_edges

    def _ensure_connectivity_augment(
        self,
        out_subgraph: Dict[str, Any],
        base_tuples: List[Tuple[Any, Any, Any]],
        topic_entities: Optional[Iterable[Any]],
    ) -> Dict[str, Any]:
        seeds, skipped, skipped_examples = self._normalize_seed_ids(topic_entities)
        if skipped:
            logger.warning(
                "Connectivity(augment): skipped %d unconvertible seeds; examples=%s",
                skipped,
                skipped_examples,
            )
        if not seeds:
            raise RuntimeError(
                "Connectivity(augment) failed: no valid seed entity ids after conversion."
            )

        out_tuples_raw = list(out_subgraph.get("tuples") or [])
        out_entities_raw = list(out_subgraph.get("entities") or [])

        out_tuple_map: Dict[Tuple[int, Any, int], Tuple[int, Any, int]] = {}
        for tpl in out_tuples_raw:
            if len(tpl) != 3:
                continue
            sbj, rel, obj = tpl
            u = self._to_int_entity_id(sbj)
            v = self._to_int_entity_id(obj)
            if u is None or v is None:
                raise RuntimeError(
                    f"Connectivity(augment) failed: non-convertible tuple endpoints: {tpl}"
                )
            key = self._tuple_key(u, rel, v)
            out_tuple_map.setdefault(key, (u, rel, v))

        out_entity_ids: List[int] = []
        out_entity_set = set()
        bad_entities: List[Any] = []
        for ent in out_entities_raw:
            ent_id = self._to_int_entity_id(ent)
            if ent_id is None:
                if len(bad_entities) < 5:
                    bad_entities.append(ent)
                continue
            if ent_id in out_entity_set:
                continue
            out_entity_set.add(ent_id)
            out_entity_ids.append(ent_id)
        if bad_entities:
            raise RuntimeError(
                f"Connectivity(augment) failed: non-convertible entities in out_subgraph['entities']: {bad_entities}"
            )

        seeds_set = set(seeds)
        out_entity_set |= seeds_set

        endpoints = set()
        for sbj, _, obj in out_tuple_map.values():
            endpoints.add(sbj)
            endpoints.add(obj)

        v_out = set(endpoints) | set(out_entity_set) | seeds_set

        reachable = self._reachable_out(out_tuple_map.values(), seeds)
        unreachable = v_out - reachable

        logger.info(
            "Connectivity(augment): seeds=%d, V_out=%d, edges=%d, unreachable=%d",
            len(seeds),
            len(v_out),
            len(out_tuple_map),
            len(unreachable),
        )

        added_edges = 0
        if unreachable:
            parent_dir = self._bfs_parent_edges(base_tuples, seeds, allow_reverse=False)
            for t in list(unreachable):
                if t in seeds_set:
                    continue
                if t not in parent_dir:
                    continue
                for u, rel, v in self._restore_path_edges(t, seeds_set, parent_dir):
                    key = self._tuple_key(u, rel, v)
                    if key in out_tuple_map:
                        continue
                    out_tuple_map[key] = (u, rel, v)
                    added_edges += 1

            reachable = self._reachable_out(out_tuple_map.values(), seeds)
            unreachable = v_out - reachable

        if unreachable:
            parent_rev = self._bfs_parent_edges(base_tuples, seeds, allow_reverse=True)
            for t in list(unreachable):
                if t in seeds_set:
                    continue
                if t not in parent_rev:
                    continue
                for u, rel, v in self._restore_path_edges(t, seeds_set, parent_rev):
                    key = self._tuple_key(u, rel, v)
                    if key in out_tuple_map:
                        continue
                    out_tuple_map[key] = (u, rel, v)
                    added_edges += 1

            reachable = self._reachable_out(out_tuple_map.values(), seeds)
            unreachable = v_out - reachable

        if unreachable:
            examples = sorted(unreachable)[:5]
            raise RuntimeError(
                f"Connectivity(augment) failed: candidate graph disconnected; remaining unreachable={len(unreachable)}, examples={examples}"
            )

        endpoints_final = set()
        for sbj, _, obj in out_tuple_map.values():
            endpoints_final.add(sbj)
            endpoints_final.add(obj)
        entities_final = sorted(endpoints_final | out_entity_set | seeds_set)
        v_final = set(entities_final)

        reachable_final = self._reachable_out(out_tuple_map.values(), seeds)
        if not v_final.issubset(reachable_final):
            remain = sorted(v_final - reachable_final)[:10]
            raise RuntimeError(
                f"Connectivity(augment) assertion failed: unreachable nodes remain: {remain}"
            )

        out = dict(out_subgraph)
        out["tuples"] = [
            out_tuple_map[k]
            for k in sorted(out_tuple_map.keys(), key=lambda x: (x[0], str(x[1]), x[2]))
        ]
        out["entities"] = entities_final
        logger.info(
            "Connectivity(augment): added_edges=%d, final_edges=%d, final_entities=%d",
            added_edges,
            len(out_tuple_map),
            len(entities_final),
        )
        return out

    def _build_encoder(self, relation_texts: List[str]) -> BaseEncoder:
        encoder_type = (self.config.encoder_type or "tfidf").lower()
        if encoder_type in ("sbert", "sentence-transformer", "sentence_transformers"):
            model_name = (
                self.config.encoder_model_name
                or "sentence-transformers/all-MiniLM-L6-v2"
            )
            if SentenceTransformer is not None:
                return SentenceTransformerEncoder(relation_texts, model_name)
            logger.warning("sentence-transformers not available, falling back to TF-IDF.")
        return TfidfEncoder(relation_texts)

    def _entity_key(self, ent: Any) -> Any:
        if isinstance(ent, dict):
            if "kb_id" in ent and ent["kb_id"] is not None:
                return self._normalize_entity_key(ent["kb_id"])
            if "text" in ent and ent["text"] is not None:
                return self._normalize_entity_key(ent["text"])
        return self._normalize_entity_key(ent)

    @staticmethod
    def _normalize_entity_key(key: Any) -> Any:
        if not isinstance(key, str):
            return key
        key = key.strip()
        if key.startswith("/m/"):
            return "m." + key[3:]
        if key.startswith("/g/"):
            return "g." + key[3:]
        return key

    def _tuple_entity_keys(self, tuples: Iterable[Tuple[Any, Any, Any]]) -> List[Any]:
        keys: List[Any] = []
        seen = set()
        for tpl in tuples:
            if len(tpl) != 3:
                continue
            sbj, _, obj = tpl
            for ent in (sbj, obj):
                key = self._entity_key(ent)
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
        return keys

    def _rebuild_entities(
        self,
        tuples: List[Tuple[Any, Any, Any]],
        original_entities: Optional[Iterable[Any]],
        topic_entities: Optional[Iterable[Any]],
    ) -> List[Any]:
        entity_lookup: Dict[Any, Any] = {}
        if original_entities:
            for ent in original_entities:
                key = self._entity_key(ent)
                if key not in entity_lookup:
                    entity_lookup[key] = ent

        def select_entity(ent: Any) -> Any:
            key = self._entity_key(ent)
            return entity_lookup.get(key, ent)

        rebuilt: List[Any] = []
        seen = set()
        for sbj, _, obj in tuples:
            for ent in (sbj, obj):
                key = self._entity_key(ent)
                if key in seen:
                    continue
                seen.add(key)
                rebuilt.append(select_entity(ent))
        if topic_entities:
            for ent in topic_entities:
                key = self._entity_key(ent)
                if key in seen:
                    continue
                seen.add(key)
                rebuilt.append(select_entity(ent))
        return rebuilt

    def _relation_id(self, rel: Any) -> int:
        rel_key = rel
        if isinstance(rel, dict):
            if "text" in rel:
                rel_key = rel["text"]
            elif "id" in rel:
                rel_key = rel["id"]
        rel_id = _parse_rel_key(rel_key, self.relation2id)
        if rel_id is not None:
            return rel_id
        rel_str = str(rel_key)
        if rel_str not in self._unknown_rel_map:
            self._unknown_rel_map[rel_str] = self._next_unknown_rel_id
            self._unknown_rel_desc[self._next_unknown_rel_id] = normalize_relation_text(
                rel_str
            )
            self._next_unknown_rel_id -= 1
        return self._unknown_rel_map[rel_str]

    def _relation_desc(self, rel_id: int) -> str:
        if rel_id in self._unknown_rel_desc:
            return self._unknown_rel_desc[rel_id]
        if rel_id in self.rel_descs:
            return self.rel_descs[rel_id]
        if rel_id >= self.base_relation_count:
            base_id = rel_id - self.base_relation_count
            if 0 <= base_id < self.base_relation_count:
                return "inverse of " + self.rel_descs.get(
                    base_id, self.id2relation.get(base_id, str(base_id))
                )
        return self.id2relation.get(rel_id, str(rel_id))

    def _parse_edges(self, tuples: List[Tuple[Any, Any, Any]]) -> List[Edge]:
        edges: List[Edge] = []
        for idx, tpl in enumerate(tuples):
            if len(tpl) != 3:
                continue
            sbj, rel, obj = tpl
            head = self._entity_key(sbj)
            tail = self._entity_key(obj)
            rel_id = self._relation_id(rel)
            edges.append(Edge(head=head, rel_id=rel_id, tail=tail, original=tpl, index=idx))
        return edges

    def _score_relations(self, question: str, rel_ids: List[int]) -> Dict[int, float]:
        if not rel_ids:
            return {}
        q_vec = self.encoder.encode_text(question)
        rel_vecs = self.encoder.get_relation_vectors(rel_ids, self._relation_desc)
        scores = self.encoder.similarity(q_vec, rel_vecs)
        return {rel_id: float(score) for rel_id, score in zip(rel_ids, scores)}

    def _edge_scores(
        self, edges: List[Edge], rel_scores: Dict[int, float], pair_stats: Optional[PairStats], gamma: float
    ) -> Dict[int, float]:
        incoming_rels: Dict[Any, List[int]] = {}
        out_degree: Dict[Any, int] = {}
        for edge in edges:
            incoming_rels.setdefault(edge.tail, []).append(edge.rel_id)
            out_degree[edge.head] = out_degree.get(edge.head, 0) + 1

        score_map: Dict[int, float] = {}
        for edge in edges:
            sq = rel_scores.get(edge.rel_id, 0.0)
            best_prev = 0.0
            if pair_stats is not None:
                for r_prev in incoming_rels.get(edge.head, []):
                    best_prev = max(best_prev, pair_stats.score(r_prev, edge.rel_id))
            hub_penalty = math.log(out_degree.get(edge.head, 0) + 1.0)
            score = self.config.alpha * sq + self.config.beta * best_prev - gamma * hub_penalty
            edge.score = score
            score_map[edge.index] = score
        return score_map

    def _apply_pruning(
        self,
        edges: List[Edge],
        rel_scores: Dict[int, float],
        top_m: int,
        top_k_rel: int,
    ) -> List[Edge]:
        if not edges:
            return []
        filtered_edges = edges
        if top_k_rel > 0:
            rel_ids = list({edge.rel_id for edge in edges})
            if top_k_rel < len(rel_ids):
                rel_ids_sorted = sorted(
                    rel_ids, key=lambda rid: rel_scores.get(rid, 0.0), reverse=True
                )
                keep_rels = set(rel_ids_sorted[:top_k_rel])
                filtered_edges = [e for e in edges if e.rel_id in keep_rels]

        if top_m <= 0:
            return filtered_edges

        edges_by_head: Dict[Any, List[Edge]] = {}
        for edge in filtered_edges:
            edges_by_head.setdefault(edge.head, []).append(edge)
        kept: List[Edge] = []
        for edge_list in edges_by_head.values():
            edge_list.sort(key=lambda e: (-e.score, e.index))
            kept.extend(edge_list[:top_m])
        kept.sort(key=lambda e: e.index)
        return kept

    def _ensure_topic_coverage(
        self,
        pruned_edges: List[Edge],
        original_edges: List[Edge],
        topic_entities: Optional[Iterable[Any]],
        edge_scores: Dict[int, float],
    ) -> List[Edge]:
        if not topic_entities:
            return pruned_edges
        topic_keys = {self._entity_key(ent) for ent in topic_entities}
        pruned_set = {edge.index for edge in pruned_edges}
        incident_by_topic: Dict[Any, List[Edge]] = {}
        for edge in original_edges:
            if edge.head in topic_keys:
                incident_by_topic.setdefault(edge.head, []).append(edge)
            if edge.tail in topic_keys and edge.tail != edge.head:
                incident_by_topic.setdefault(edge.tail, []).append(edge)

        for topic in topic_keys:
            candidates = incident_by_topic.get(topic)
            if not candidates:
                continue
            has_incident = any(
                edge.head == topic or edge.tail == topic for edge in pruned_edges
            )
            if has_incident:
                continue
            best_edge = max(candidates, key=lambda e: edge_scores.get(e.index, 0.0))
            if best_edge.index not in pruned_set:
                pruned_edges.append(best_edge)
                pruned_set.add(best_edge.index)
        pruned_edges.sort(key=lambda e: e.index)
        return pruned_edges

    def _filter_seed_connected(
        self,
        pruned_edges: List[Edge],
        topic_entities: Optional[Iterable[Any]],
    ) -> List[Edge]:
        if not pruned_edges or not topic_entities:
            return pruned_edges
        topic_keys = {self._entity_key(ent) for ent in topic_entities}
        adj: Dict[Any, set] = {}
        for edge in pruned_edges:
            adj.setdefault(edge.head, set()).add(edge.tail)
            adj.setdefault(edge.tail, set()).add(edge.head)
        seed_keys = [key for key in topic_keys if key in adj]
        if not seed_keys:
            return []
        reachable = set(seed_keys)
        stack = list(seed_keys)
        while stack:
            node = stack.pop()
            for nbr in adj.get(node, ()):
                if nbr in reachable:
                    continue
                reachable.add(nbr)
                stack.append(nbr)
        filtered = [
            edge
            for edge in pruned_edges
            if edge.head in reachable and edge.tail in reachable
        ]
        filtered.sort(key=lambda e: e.index)
        return filtered

    def _min_edges_threshold(self, num_edges: int) -> int:
        if self.config.min_edges <= 0:
            return 0
        if 0 < self.config.min_edges < 1:
            return max(1, int(math.ceil(num_edges * self.config.min_edges)))
        return int(self.config.min_edges)

    def _denoise_stage(
        self,
        question: str,
        edges: List[Edge],
        topic_entities: Optional[Iterable[Any]],
        stage: StageConfig,
        pair_stats: Optional[PairStats],
    ) -> List[Edge]:
        rel_ids = list({edge.rel_id for edge in edges})
        rel_scores = self._score_relations(question, rel_ids)
        edge_scores = self._edge_scores(edges, rel_scores, pair_stats, stage.gamma)
        pruned_edges = self._apply_pruning(edges, rel_scores, stage.top_m, stage.top_k_rel)
        if self.config.ensure_connectivity:
            pruned_edges = self._ensure_topic_coverage(
                pruned_edges, edges, topic_entities, edge_scores
            )
            pruned_edges = self._filter_seed_connected(pruned_edges, topic_entities)
        return pruned_edges

    def denoise(
        self,
        question: str,
        subgraph: Dict[str, Any],
        topic_entities: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        base_tuples = list(subgraph.get("tuples") or [])

        seeds, skipped, skipped_examples = self._normalize_seed_ids(topic_entities)
        if skipped:
            logger.warning(
                "Node scoring: skipped %d unconvertible seeds; examples=%s",
                skipped,
                skipped_examples,
            )
        if not seeds:
            raise RuntimeError("Denoise failed: no valid seed entities.")
        seeds_set = set(seeds)

        node_ctx, base_edges_int = self._build_node_context(base_tuples)

        candidate_nodes = set(seeds_set)
        for u, _, v in base_edges_int:
            candidate_nodes.add(u)
            candidate_nodes.add(v)
        for ent in subgraph.get("entities") or []:
            ent_id = self._to_int_entity_id(ent)
            if ent_id is not None:
                candidate_nodes.add(ent_id)

        node_scores = self._score_nodes(question, candidate_nodes, node_ctx)
        strict_candidates = [
            (eid, score)
            for eid, score in node_scores.items()
            if score > NODE_SCORE_THRESHOLD and eid not in seeds_set
        ]
        strict_candidates.sort(key=lambda x: x[1], reverse=True)

        top_k_nodes = int(self.config.strict.top_k_rel)
        if top_k_nodes < 0:
            top_k_nodes = 0
        strict_nodes = [eid for eid, _ in strict_candidates[:top_k_nodes]]
        selected_nodes = set(strict_nodes) | seeds_set

        logger.info(
            "Node selection: strict_nodes=%d (threshold>%.3f, K=%d), selected_nodes=%d",
            len(strict_nodes),
            NODE_SCORE_THRESHOLD,
            top_k_nodes,
            len(selected_nodes),
        )
        logger.debug(
            "Top strict_nodes (up to 5): %s",
            [(eid, float(score)) for eid, score in strict_candidates[:5]],
        )

        out_tuple_map: Dict[Tuple[int, Any, int], Tuple[int, Any, int]] = {}
        internal_count = 0
        boundary_count = 0
        for u, rel, v in base_edges_int:
            in_u = u in selected_nodes
            in_v = v in selected_nodes
            if not (in_u or in_v):
                continue
            if in_u and in_v:
                internal_count += 1
            else:
                boundary_count += 1
            key = self._tuple_key(u, rel, v)
            out_tuple_map.setdefault(key, (u, rel, v))

        out_tuples_initial = list(out_tuple_map.values())
        logger.info(
            "Node-induced: edges=%d (internal=%d, boundary=%d)",
            len(out_tuples_initial),
            internal_count,
            boundary_count,
        )

        min_edges = self._min_edges_threshold(len(base_edges_int))
        if len(out_tuple_map) < min_edges:
            logger.warning(
                "Sparsity prevention: strict_nodes=%d, selected_nodes=%d, out_edges=%d < min_edges=%d. "
                "Not enforcing min_edges because candidate edges may be insufficient.",
                len(strict_nodes),
                len(selected_nodes),
                len(out_tuple_map),
                min_edges,
            )
        else:
            logger.info(
                "Sparsity prevention: out_edges=%d (min_edges=%d)",
                len(out_tuple_map),
                min_edges,
            )

        endpoints = set()
        for u, _, v in out_tuple_map.values():
            endpoints.add(u)
            endpoints.add(v)

        new_subgraph = dict(subgraph)
        internal_keys = []
        boundary_keys = []
        for k in out_tuple_map.keys():
            sbj, _, obj = k
            if sbj in selected_nodes and obj in selected_nodes:
                internal_keys.append(k)
            else:
                boundary_keys.append(k)
        internal_keys.sort(key=lambda x: (x[0], str(x[1]), x[2]))
        boundary_keys.sort(key=lambda x: (x[0], str(x[1]), x[2]))
        new_subgraph["tuples"] = [out_tuple_map[k] for k in (internal_keys + boundary_keys)]
        new_subgraph["entities"] = sorted(endpoints | selected_nodes)

        new_subgraph = self._ensure_connectivity_augment(
            new_subgraph, base_tuples, topic_entities
        )
        return new_subgraph
