from __future__ import annotations

import json
import logging
import math
import os
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
        self._unknown_rel_map: Dict[str, int] = {}
        self._unknown_rel_desc: Dict[int, str] = {}
        self._next_unknown_rel_id = -1

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
            if "text" in ent:
                return ent["text"]
            if "kb_id" in ent:
                return ent["kb_id"]
        return ent

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
        outgoing_by_head: Dict[Any, List[Edge]] = {}
        for edge in original_edges:
            outgoing_by_head.setdefault(edge.head, []).append(edge)

        for topic in topic_keys:
            if not outgoing_by_head.get(topic):
                continue
            has_outgoing = any(edge.head == topic for edge in pruned_edges)
            if has_outgoing:
                continue
            candidates = outgoing_by_head[topic]
            best_edge = max(candidates, key=lambda e: edge_scores.get(e.index, 0.0))
            if best_edge.index not in pruned_set:
                pruned_edges.append(best_edge)
                pruned_set.add(best_edge.index)
        pruned_edges.sort(key=lambda e: e.index)
        return pruned_edges

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
        return pruned_edges

    def denoise(
        self,
        question: str,
        subgraph: Dict[str, Any],
        topic_entities: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        if not self.config.enable:
            return subgraph
        tuples = subgraph.get("tuples", [])
        if not tuples:
            return subgraph
        edges = self._parse_edges(tuples)
        if not edges:
            return subgraph

        pair_stats = self.global_pair_stats
        if self.config.enable_pair_score and pair_stats is None:
            pair_stats = build_local_pair_stats(edges)

        strict_edges = self._denoise_stage(
            question, edges, topic_entities, self.config.strict, pair_stats
        )
        min_edges = self._min_edges_threshold(len(edges))
        if len(strict_edges) < min_edges:
            logger.debug(
                "Strict denoise too small (%d < %d), falling back to loose.",
                len(strict_edges),
                min_edges,
            )
            loose_edges = self._denoise_stage(
                question, edges, topic_entities, self.config.loose, pair_stats
            )
            kept_edges = loose_edges
        else:
            kept_edges = strict_edges

        new_subgraph = dict(subgraph)
        new_subgraph["tuples"] = [edge.original for edge in kept_edges]
        return new_subgraph
