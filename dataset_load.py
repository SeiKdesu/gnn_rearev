import json
import numpy as np
import re
from tqdm import tqdm
import torch
from collections import Counter
import random
import warnings
import pickle
warnings.filterwarnings("ignore")
from modules.question_encoding.tokenizers import LSTMTokenizer#, BERTTokenizer
from transformers import AutoTokenizer
import time

import os

from subgraph_union_graph import load_union_graph
from subgraph_policy import load_policy_ckpt, expand_sample_subgraph


_PSE_UNION_GRAPH_MEMO = {}
_PSE_POLICY_MEMO = {}
# Shared subgraph (merged) memoization (process-local).
# [shared-subgraph]
_SHARED_SUBGRAPH_MEMO = {}


# [shared-subgraph]
def _coerce_entity_id(x, entity2id):
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        if x in entity2id:
            return int(entity2id[x])
        try:
            return int(x)
        except Exception as e:
            raise KeyError(f"Unknown entity string: {x!r}") from e
    if isinstance(x, dict):
        for key in ("text", "kb_id", "id"):
            if key in x:
                v = x[key]
                if isinstance(v, int):
                    return int(v)
                if isinstance(v, str):
                    if v in entity2id:
                        return int(entity2id[v])
                    try:
                        return int(v)
                    except Exception:
                        pass
        raise KeyError(f"Unknown entity dict keys: {list(x.keys())}")
    raise TypeError(f"Unsupported entity type: {type(x)}")


# [shared-subgraph]
def _coerce_relation_id(x, relation2id):
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        if x in relation2id:
            return int(relation2id[x])
        try:
            return int(x)
        except Exception as e:
            raise KeyError(f"Unknown relation string: {x!r}") from e
    if isinstance(x, dict):
        for key in ("text", "id"):
            if key in x:
                v = x[key]
                if isinstance(v, int):
                    return int(v)
                if isinstance(v, str):
                    if v in relation2id:
                        return int(relation2id[v])
                    try:
                        return int(v)
                    except Exception:
                        pass
        raise KeyError(f"Unknown relation dict keys: {list(x.keys())}")
    raise TypeError(f"Unsupported relation type: {type(x)}")


class BasicDataLoader(object):
    """ 
    Basic Dataloader contains all the functions to read questions and KGs from json files and
    create mappings between global entity ids and local ids that are used during GNN updates.
    """

    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        self.tokenize = tokenize
        self._parse_args(config, word2id, relation2id, entity2id)
        self._init_union_graph_and_policy(config, tokenize)
        self._load_file(config, data_type)
        self._load_data()
        

    def _load_file(self, config, data_type="train"):

        """
        Loads lines (questions + KG subgraphs) from json files.
        """
        
        data_file = config['data_folder'] + data_type + ".json"
        self.data_file = data_file
        print('loading data from', data_file)
        self.data_type = data_type
        self.data = []
        skip_index = set()
        index = 0

        with open(data_file) as f_in:
            for line in tqdm(f_in):
                if index == config['max_train'] and data_type == "train": break  #break if we reach max_question_size
                line = json.loads(line)
                if not getattr(self, "use_shared_subgraph", False):
                    line = self._maybe_expand_subgraph(line)
                else:
                    # In shared-subgraph mode we ignore per-sample subgraphs; drop them to save RAM.
                    # [shared-subgraph]
                    if "subgraph" in line:
                        line["subgraph"] = {"tuples": [], "entities": []}
                
                if len(line['entities']) == 0:
                    skip_index.add(index)
                    continue
                self.data.append(line)
                if not getattr(self, "use_shared_subgraph", False):
                    self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                index += 1

        print("skip", skip_index)
        print('max_facts: ', self.max_facts)
        if getattr(self, "enable_policy_expand", False):
            avg_ent = float(np.mean([len(s.get("subgraph", {}).get("entities", [])) for s in self.data])) if self.data else 0.0
            avg_tpl = float(np.mean([len(s.get("subgraph", {}).get("tuples", [])) for s in self.data])) if self.data else 0.0
            cov = self._answer_coverage(self.data)
            print(
                f"[PSE] split={data_type} avg_entities={avg_ent:.2f} avg_tuples={avg_tpl:.2f} "
                f"answer_coverage={cov*100:.2f}%"
            )
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

    def _load_data(self):

        """
        Creates mappings between global entity ids and local entity ids that are used during GNN updates.
        """

        # Shared-subgraph mode builds everything lazily per-batch and reuses one merged graph.
        # [shared-subgraph]
        if getattr(self, "use_shared_subgraph", False):
            return self._load_data_shared()

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        self._prepare_data()

    # [shared-subgraph]
    def _load_data_shared(self):
        """
        Shared-subgraph mode:
          - Load one merged subgraph from `subgraph.json` once per process.
          - Avoid (num_data x max_local_entity) dense precomputations.
          - Build only per-sample lightweight metadata; build dense arrays per-batch in `get_batch`.
        """
        shared = self._get_shared_subgraph(self.shared_subgraph_path)

        self._shared_entities = shared["entities"]
        self._shared_g2l = shared["g2l"]
        self._shared_base_heads = shared["heads"]
        self._shared_base_rels = shared["rels"]
        self._shared_base_tails = shared["tails"]
        self._shared_num_raw_tuples = shared["num_raw_tuples"]
        self._shared_num_raw_unique_tuples = shared["num_raw_unique_tuples"]
        self._shared_bad_tuples = shared["bad_tuples"]

        self.max_local_entity = len(self._shared_entities)
        self.max_facts = int(len(self._shared_base_heads))
        if self.use_self_loop:
            # Keep semantics consistent with legacy sizing.
            self.max_facts = self.max_facts + self.max_local_entity

        print(
            f"[shared-subgraph] split={self.data_type} path={self.shared_subgraph_path} "
            f"|V|={self.max_local_entity} raw_tuples={self._shared_num_raw_tuples} "
            f"unique_tuples={self._shared_num_raw_unique_tuples} bad_tuples={self._shared_bad_tuples} "
            f"effective_facts={len(self._shared_base_heads)}"
        )

        # Per-sample lightweight caches (object arrays).
        self.question_id = []
        self.answer_lists = np.empty(self.num_data, dtype=object)
        self._shared_seed_local = np.empty(self.num_data, dtype=object)
        self._shared_answer_local = np.empty(self.num_data, dtype=object)

        # Tokenize questions + precompute (small) seed/answer index lists.
        self._prepare_data_shared()

    # [shared-subgraph]
    def _get_shared_subgraph(self, path: str):
        cache_key = (os.path.abspath(path), bool(self.use_inverse_relation))
        if cache_key in _SHARED_SUBGRAPH_MEMO:
            return _SHARED_SUBGRAPH_MEMO[cache_key]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[shared-subgraph] subgraph.json not found: {path!r}\n"
                "Build it (train-only) using:\n"
                "  python3 scripts/build_shared_subgraph.py --input <data_folder>/train.json --output <data_folder>/subgraph.json"
            )

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entities_in = payload.get("entities", []) or []
        tuples_in = payload.get("tuples", []) or []

        # Coerce + unique-preserve entity list.
        entities = []
        seen_ent = set()
        for e in entities_in:
            eid = _coerce_entity_id(e, self.entity2id)
            if eid in seen_ent:
                continue
            seen_ent.add(eid)
            entities.append(eid)
        g2l = {eid: i for i, eid in enumerate(entities)}

        # Coerce + dedup tuples; also ensure endpoints are present in `entities`.
        tuple_seen = set()
        heads = []
        rels = []
        tails = []
        bad_tuples = 0
        num_raw_tuples = 0
        num_raw_unique_tuples = 0
        num_base_rel = len(self.relation2id)

        for tpl in tuples_in:
            if not isinstance(tpl, (list, tuple)) or len(tpl) != 3:
                continue
            num_raw_tuples += 1
            try:
                h = _coerce_entity_id(tpl[0], self.entity2id)
                r = _coerce_relation_id(tpl[1], self.relation2id)
                t = _coerce_entity_id(tpl[2], self.entity2id)
            except Exception:
                bad_tuples += 1
                continue

            key = (h, r, t)
            if key in tuple_seen:
                continue
            tuple_seen.add(key)
            num_raw_unique_tuples += 1

            # Repair in-memory if endpoints are missing from entities.
            # [shared-subgraph]
            if h not in g2l:
                g2l[h] = len(entities)
                entities.append(h)
            if t not in g2l:
                g2l[t] = len(entities)
                entities.append(t)

            h_l = g2l[h]
            t_l = g2l[t]
            heads.append(h_l)
            rels.append(r)
            tails.append(t_l)
            if self.use_inverse_relation:
                if int(r) >= num_base_rel:
                    heads.append(t_l)
                    rels.append(int(r) - num_base_rel)
                    tails.append(h_l)
                else:
                    heads.append(t_l)
                    rels.append(int(r) + num_base_rel)
                    tails.append(h_l)

        out = {
            "entities": entities,
            "g2l": g2l,
            "heads": np.asarray(heads, dtype=np.int64),
            "rels": np.asarray(rels, dtype=np.int64),
            "tails": np.asarray(tails, dtype=np.int64),
            "num_raw_tuples": int(num_raw_tuples),
            "num_raw_unique_tuples": int(num_raw_unique_tuples),
            "bad_tuples": int(bad_tuples),
        }
        _SHARED_SUBGRAPH_MEMO[cache_key] = out
        return out

    # [shared-subgraph]
    def _prepare_data_shared(self):
        """
        Shared-subgraph variant of `_prepare_data`:
          - Tokenizes questions for all samples (same as legacy).
          - Precomputes only lightweight per-sample lists:
              * seed local indices
              * answer local indices + answer global lists
        """
        max_count = 0
        for line in self.data:
            word_list = line["question"].split(" ")
            max_count = max(max_count, len(word_list))

        if self.rel_word_emb:
            self.build_rel_words(self.tokenize)
        else:
            self.rel_texts = None
            self.rel_texts_inv = None
            self.ent_texts = None

        self.max_query_word = max_count

        # build tokenizers
        if self.tokenize == "lstm":
            self.num_word = len(self.word2id)
            self.tokenizer = LSTMTokenizer(self.word2id, self.max_query_word)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)
        else:
            if self.tokenize == "bert":
                tokenizer_name = "bert-base-uncased"
            elif self.tokenize == "roberta":
                tokenizer_name = "roberta-base"
            elif self.tokenize == "sbert":
                tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
            elif self.tokenize == "sbert2":
                tokenizer_name = "sentence-transformers/all-mpnet-base-v2"
            elif self.tokenize == "t5":
                tokenizer_name = "t5-small"
            elif self.tokenize == "simcse":
                tokenizer_name = "princeton-nlp/sup-simcse-bert-base-uncased"
            elif self.tokenize == "relbert":
                tokenizer_name = "pretrained_lms/sr-simbert/"
            else:
                tokenizer_name = "bert-base-uncased"

            self.max_query_word = max_count + 2  # [CLS] + [SEP]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.num_word = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)

        g2l = self._shared_g2l
        for idx, sample in enumerate(tqdm(self.data, desc="[shared-subgraph] preparing samples")):
            self.question_id.append(sample["id"])

            # tokenize question
            if self.tokenize == "lstm":
                self.query_texts[idx] = self.tokenizer.tokenize(sample["question"])
            else:
                tokens = self.tokenizer.encode_plus(
                    text=sample["question"],
                    max_length=self.max_query_word,
                    pad_to_max_length=True,
                    return_attention_mask=False,
                    truncation=True,
                )
                self.query_texts[idx] = np.array(tokens["input_ids"])

            # seed entities (question entities)
            seed_set = set()
            key_ent = "entities_cid" if "entities_cid" in sample else "entities"
            for entity in sample.get(key_ent, []) or []:
                try:
                    ge = _coerce_entity_id(entity, self.entity2id)
                except Exception:
                    continue
                le = g2l.get(ge)
                if le is not None:
                    seed_set.add(int(le))
            self._shared_seed_local[idx] = sorted(seed_set)

            # answers (global list + local list)
            answer_list = []
            answer_local = []
            if "answers_cid" in sample and sample["answers_cid"] is not None:
                for answer in sample["answers_cid"]:
                    try:
                        ae = _coerce_entity_id(answer, self.entity2id)
                    except Exception:
                        continue
                    answer_list.append(ae)
                    le = g2l.get(ae)
                    if le is not None:
                        answer_local.append(int(le))
            else:
                for answer in sample.get("answers", []) or []:
                    if not isinstance(answer, dict):
                        continue
                    keyword = "text" if isinstance(answer.get("kb_id"), int) else "kb_id"
                    try:
                        ae = int(self.entity2id[answer[keyword]])
                    except Exception:
                        continue
                    answer_list.append(ae)
                    le = g2l.get(ae)
                    if le is not None:
                        answer_local.append(int(le))
            self.answer_lists[idx] = answer_list
            self._shared_answer_local[idx] = answer_local

    # [shared-subgraph]
    def _shared_build_dense_batch(self, sample_ids):
        """
        Build dense matrices only for the current batch:
          - candidate_entities: (B, |V|)
          - query_entities: (B, |V|)
          - seed_distribution: (B, |V|)
          - answer_dists: (B, |V|)
        """
        batch_size = len(sample_ids)
        max_local_entity = self.max_local_entity

        # NOTE: This is the only unavoidable duplication in shared-subgraph mode:
        # the model expects per-sample (B, |V|) dense tensors.
        # [shared-subgraph]
        candidate_entities = np.tile(np.asarray(self._shared_entities, dtype=np.int64), (batch_size, 1))
        query_entities = np.zeros((batch_size, max_local_entity), dtype=np.float32)
        seed_distribution = np.zeros((batch_size, max_local_entity), dtype=np.float32)
        answer_dists = np.zeros((batch_size, max_local_entity), dtype=np.float32)

        uniform_val = 1.0 / float(max_local_entity) if max_local_entity > 0 else 0.0
        for bi, sid in enumerate(sample_ids):
            seeds = self._shared_seed_local[sid] or []
            for le in seeds:
                query_entities[bi, le] = 1.0
            if seeds:
                seed_distribution[bi, seeds] = 1.0 / float(len(seeds))
            else:
                seed_distribution[bi, :] = uniform_val

            ans_local = self._shared_answer_local[sid] or []
            for le in ans_local:
                answer_dists[bi, le] = 1.0

        return candidate_entities, query_entities, seed_distribution, answer_dists

    def _parse_args(self, config, word2id, relation2id, entity2id):

        """
        Builds necessary dictionaries and stores arguments.
        """
        self.data_eff = config['data_eff']
        self.data_name = config['name']

        if 'use_inverse_relation' in config:
            self.use_inverse_relation = config['use_inverse_relation']
        else:
            self.use_inverse_relation = False
        if 'use_self_loop' in config:
            self.use_self_loop = config['use_self_loop']
        else:
            self.use_self_loop = False

        self.rel_word_emb = config['relation_word_emb']
        #self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0

        print('building word index ...')
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        self.q_type = config['q_type']
        self.enable_policy_expand = bool(config.get("enable_policy_expand", False))
        # For edge-weight computation toggles in `_build_fact_mat`.
        # [shared-subgraph]
        self.normalized_gnn = bool(config.get("normalized_gnn", False))
        self.norm_rel = bool(config.get("norm_rel", False))

        # Shared subgraph (merged) mode.
        # [shared-subgraph]
        self.use_shared_subgraph = bool(config.get("use_shared_subgraph", False))
        self.shared_subgraph_path = config.get("shared_subgraph_path") or os.path.join(
            config["data_folder"], "subgraph.json"
        )
        if self.use_shared_subgraph and self.enable_policy_expand:
            # PSE expands *per-sample* subgraphs; shared-subgraph mode ignores per-sample subgraphs.
            print("[shared-subgraph] enable_policy_expand=True ignored (shared-subgraph mode).")
            self.enable_policy_expand = False

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    def _init_union_graph_and_policy(self, args, tokenize):
        """
        Initialize train-only union graph + edge policy (CPU by default) for inference-time expansion.
        This MUST NOT use dev/test answers; it loads caches created from train.json only.
        """
        self._pse_union_adj = None
        self._pse_union_meta = None
        self._pse_policy = None

        if not getattr(self, "enable_policy_expand", False):
            return

        data_folder = args["data_folder"]
        union_cache = args.get("union_graph_cache") or os.path.join(data_folder, "cache/union_graph.pkl")
        policy_ckpt = args.get("policy_ckpt") or os.path.join(data_folder, "cache/policy_ckpt.pt")
        policy_device = args.get("policy_device") or "cpu"

        if not os.path.exists(union_cache) or not os.path.exists(policy_ckpt):
            raise RuntimeError(
                "[PSE] enable_policy_expand=True but cache/ckpt missing.\n"
                f"  union_graph_cache: {union_cache} (exists={os.path.exists(union_cache)})\n"
                f"  policy_ckpt: {policy_ckpt} (exists={os.path.exists(policy_ckpt)})\n"
                "Build them from TRAIN ONLY using:\n"
                "  python3 main.py --mode train_policy --data_folder <data_folder> --lm <lstm|bert|...>\n"
            )

        if union_cache in _PSE_UNION_GRAPH_MEMO:
            self._pse_union_adj, self._pse_union_meta = _PSE_UNION_GRAPH_MEMO[union_cache]
        else:
            self._pse_union_adj, self._pse_union_meta = load_union_graph(union_cache)
            _PSE_UNION_GRAPH_MEMO[union_cache] = (self._pse_union_adj, self._pse_union_meta)

        policy_key = (policy_ckpt, policy_device)
        if policy_key in _PSE_POLICY_MEMO:
            self._pse_policy = _PSE_POLICY_MEMO[policy_key]
        else:
            self._pse_policy = load_policy_ckpt(
                policy_ckpt,
                word2id=self.word2id,
                map_location=policy_device,
                override_device=policy_device,
            )
            _PSE_POLICY_MEMO[policy_key] = self._pse_policy

        # Expansion hyperparams (defaults align with spec).
        self._pse_expand_hops = int(args.get("policy_expand_hops", 2))
        self._pse_topk_per_node = int(args.get("policy_topk_per_node", 20))
        self._pse_max_new_edges_per_hop = int(args.get("policy_max_new_edges_per_hop", 200))
        self._pse_max_new_edges_total = int(args.get("policy_max_new_edges_total", 800))
        self._pse_frontier_mode = str(args.get("policy_frontier_mode", "new"))

    def _maybe_expand_subgraph(self, sample):
        if not getattr(self, "enable_policy_expand", False):
            return sample
        return expand_sample_subgraph(
            sample,
            union_adj=self._pse_union_adj,
            policy=self._pse_policy,
            policy_expand_hops=self._pse_expand_hops,
            topk_per_node=self._pse_topk_per_node,
            max_new_edges_per_hop=self._pse_max_new_edges_per_hop,
            max_new_edges_total=self._pse_max_new_edges_total,
            frontier_mode=self._pse_frontier_mode,
        )

    def _answer_coverage(self, samples):
        """
        Utility for sanity reporting only. Expansion never uses answers.
        Coverage: fraction of samples where any answer entity id is in subgraph entities.
        """
        if not samples:
            return 0.0
        hit = 0
        total = 0
        for s in samples:
            total += 1
            ans_ids = set()
            if "answers_cid" in s and s["answers_cid"] is not None:
                ans_ids = set(int(x) for x in s["answers_cid"])
            else:
                for a in s.get("answers", []) or []:
                    if isinstance(a, int):
                        ans_ids.add(int(a))
                        continue
                    if not isinstance(a, dict):
                        continue
                    key = "text" if isinstance(a.get("kb_id"), int) else "kb_id"
                    raw = a.get(key) or a.get("kb_id") or a.get("text")
                    if raw is None:
                        continue
                    if isinstance(raw, int):
                        ans_ids.add(int(raw))
                        continue
                    if raw in self.entity2id:
                        ans_ids.add(int(self.entity2id[raw]))
                        continue
                    try:
                        ans_ids.add(int(raw))
                    except Exception:
                        continue

            ent = set(int(x) for x in s.get("subgraph", {}).get("entities", []) or [])
            if not ent:
                # fallback: endpoints
                for tpl in s.get("subgraph", {}).get("tuples", []) or []:
                    if isinstance(tpl, (list, tuple)) and len(tpl) == 3:
                        ent.add(int(tpl[0]))
                        ent.add(int(tpl[2]))
            if ans_ids and ent.intersection(ans_ids):
                hit += 1
        return hit / max(1, total)

    
    def get_quest(self, training=False):
        q_list = []
        
        sample_ids = self.sample_ids
        for sample_id in sample_ids:
            tp_str = self.decode_text(self.query_texts[sample_id, :])
            # id2word = self.id2word
            # for i in range(self.max_query_word):
            #     if self.query_texts[sample_id, i] in id2word:
            #         tp_str += id2word[self.query_texts[sample_id, i]] + " "
            q_list.append(tp_str)
        return q_list

    def decode_text(self, np_array_x):
        if self.tokenize == 'lstm':
            id2word = self.id2word
            tp_str = ""
            for i in range(self.max_query_word):
                if np_array_x[i] in id2word:
                    tp_str += id2word[np_array_x[i]] + " "
        else:
            tp_str = ""
            words = self.tokenizer.convert_ids_to_tokens(np_array_x)
            for w in words:
                if w not in ['[CLS]', '[SEP]', '[PAD]']:
                    tp_str += w + " "
        return tp_str
    

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        max_count = 0
        for line in self.data:
            word_list = line["question"].split(' ')
            max_count = max(max_count, len(word_list))

        
        if self.rel_word_emb:
            self.build_rel_words(self.tokenize)
        else:
            self.rel_texts = None
            self.rel_texts_inv = None
            self.ent_texts = None



        self.max_query_word = max_count
        #self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        #self.query_texts2 = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)

        #build tokenizers
        if self.tokenize == 'lstm':
            self.num_word = len(self.word2id)
            self.tokenizer = LSTMTokenizer(self.word2id, self.max_query_word)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)
        else:
            if self.tokenize == 'bert':
                tokenizer_name = 'bert-base-uncased'    
            elif self.tokenize  == 'roberta':
                tokenizer_name = 'roberta-base'
            elif self.tokenize  == 'sbert':
                tokenizer_name = 'sentence-transformers/all-MiniLM-L6-v2'
            elif self.tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif self.tokenize  == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif self.tokenize  == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize  == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'

            self.max_query_word = max_count + 2 #for cls token and sep
            #self.tokenizer = AutoTokenizer(self.max_query_word)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.num_word = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) #self.tokenizer.q_tokenizer.encode("[UNK]")[0]
            
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)


        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["id"])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            #print(g2l)
            if len(g2l) == 0:
                #print(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            key_ent = 'entities_cid' if 'entities_cid' in sample else 'entities'
            for j, entity in enumerate(sample[key_ent]):
                # if entity['text'] not in self.entity2id:
                #     continue
                try:
                    if isinstance(entity, dict) and  'text' in entity:
                        global_entity = self.entity2id[entity['text']]
                    else:
                        global_entity = self.entity2id[entity]
                    global_entity = self.entity2id[entity['text']]
                except:
                    global_entity = entity #self.entity2id[entity['text']]

                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)
            
            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)
            for global_entity, local_entity in g2l.items():
                if self.data_name != 'cwq':

                    if local_entity not in tp_set:  # skip entities in question
                    #print(global_entity)
                    #print(local_entity)
                        self.candidate_entities[next_id, local_entity] = global_entity
                elif self.data_name == 'cwq':
                    self.candidate_entities[next_id, local_entity] = global_entity
                # if local_entity != 0:  # skip question node
                #     self.candidate_entities[next_id, local_entity] = global_entity

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                try:
                    if isinstance(sbj, dict) and  'text' in sbj:
                        head = g2l[self.entity2id[sbj['text']]]
                        rel = self.relation2id[rel['text']]
                        tail = g2l[self.entity2id[obj['text']]]
                    else:
                        head = g2l[self.entity2id[sbj]]
                        rel = self.relation2id[rel]
                        tail = g2l[self.entity2id[obj]]
                except:
                    head = g2l[sbj]
                    try:
                        rel = int(rel)
                    except:
                        rel = self.relation2id[rel]
                    tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                self.kb_fact_rels[next_id, i] = rel
                if self.use_inverse_relation:
                    if self.enable_policy_expand and rel >= len(self.relation2id):
                        # Edge already uses inverse relation id (r + |R|); add the base direction.
                        head_list.append(tail)
                        rel_list.append(rel - len(self.relation2id))
                        tail_list.append(head)
                    else:
                        head_list.append(tail)
                        rel_list.append(rel + len(self.relation2id))
                        tail_list.append(head)
                        self.kb_fact_rels[next_id, i] = rel + len(self.relation2id)
                
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            try:
                assert np.sum(self.seed_distribution[next_id]) > 0.0
            except:
                print(next_id, len(tp_set))
                exit(-1)

            #tokenize question
            if self.tokenize == 'lstm':
                self.query_texts[next_id] = self.tokenizer.tokenize(sample['question'])
            else:
                tokens =  self.tokenizer.encode_plus(text=sample['question'], max_length=self.max_query_word, \
                    pad_to_max_length=True, return_attention_mask = False, truncation=True)
                self.query_texts[next_id] = np.array(tokens['input_ids'])


            # construct distribution for answers
            answer_list = []
            if 'answers_cid' in sample:
                for answer in sample['answers_cid']:
                    #keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                    answer_ent = answer
                    answer_list.append(answer_ent)
                    if answer_ent in g2l:
                        self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            else:
                for answer in sample['answers']:
                    keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                    answer_ent = self.entity2id[answer[keyword]]
                    answer_list.append(answer_ent)
                    if answer_ent in g2l:
                        self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            self.answer_lists[next_id] = answer_list

            if not self.data_eff:
                self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1
        num_no_query_ent = 0
        num_one_query_ent = 0
        num_multiple_ent = 0
        for i in range(next_id):
            ct = num_query_entity[i]
            if ct == 1:
                num_one_query_ent += 1
            elif ct == 0:
                num_no_query_ent += 1
            else:
                num_multiple_ent += 1
        print("{} cases in total, {} cases without query entity, {} cases with single query entity,"
              " {} cases with multiple query entities".format(next_id, num_no_query_ent,
                                                              num_one_query_ent, num_multiple_ent))

        
    def build_rel_words(self, tokenize):
        """ 
        Tokenizes relation surface forms.
        """

        max_rel_words = 0
        rel_words = []
        if 'metaqa' in self.data_file:
            for rel in self.relation2id:
                words = rel.split('_')
                max_rel_words = max(len(words), max_rel_words)
                rel_words.append(words)
            #print(rel_words)
        else:
            for rel in self.relation2id:
                rel = rel.strip()
                fields = rel.split('.')
                try:
                    words = fields[-2].split('_') + fields[-1].split('_')
                    max_rel_words = max(len(words), max_rel_words)
                    rel_words.append(words)
                    #print(rel, words)
                except:
                    words = ['UNK']
                    rel_words.append(words)
                    pass
                #words = fields[-2].split('_') + fields[-1].split('_')
            
        self.max_rel_words = max_rel_words
        if tokenize == 'lstm':
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
            for rel_id,tokens in enumerate(rel_words):
                for j, word in enumerate(tokens):
                    if j < self.max_rel_words:
                            if word in self.word2id:
                                self.rel_texts[rel_id, j] = self.word2id[word]
                                self.rel_texts_inv[rel_id, j] = self.word2id[word]
                            else:
                                self.rel_texts[rel_id, j] = len(self.word2id)
                                self.rel_texts_inv[rel_id, j] = len(self.word2id)
        else:
            if tokenize == 'bert':
                tokenizer_name = 'bert-base-uncased'
            elif tokenize == 'roberta':
                tokenizer_name = 'roberta-base'
            elif tokenize == 'sbert':
                tokenizer_name = 'sentence-transformers/all-MiniLM-L6-v2'
            elif tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif tokenize == 't5':
                tokenizer_name = 't5-small'
            elif tokenize  == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            pad_val = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
            
            for rel_id,words in enumerate(rel_words):

                tokens =  tokenizer.encode_plus(text=' '.join(words), max_length=self.max_rel_words, \
                    pad_to_max_length=True, return_attention_mask = False, truncation=True)
                tokens_inv =  tokenizer.encode_plus(text=' '.join(words[::-1]), max_length=self.max_rel_words, \
                    pad_to_max_length=True, return_attention_mask = False, truncation=True)
                self.rel_texts[rel_id] = np.array(tokens['input_ids'])
                self.rel_texts_inv[rel_id] = np.array(tokens_inv['input_ids'])


        
        #print(rel_words)
        #print(len(rel_words), len(self.relation2id))
        assert len(rel_words) == len(self.relation2id)
        #print(self.rel_texts, self.max_rel_words)

    def create_kb_adj_mats(self, sample_id):

        """
        Re-build local adj mats if we have data_eff == True (they are not pre-stored).
        """
        sample = self.data[sample_id]
        g2l = self.global2local_entity_maps[sample_id]
        
        # build connection between question and entities in it
        head_list = []
        rel_list = []
        tail_list = []
        for i, tpl in enumerate(sample['subgraph']['tuples']):
            sbj, rel, obj = tpl
            try:
                if isinstance(sbj, dict) and  'text' in sbj:
                    head = g2l[self.entity2id[sbj['text']]]
                    rel = self.relation2id[rel['text']]
                    tail = g2l[self.entity2id[obj['text']]]
                else:
                    head = g2l[self.entity2id[sbj]]
                    rel = self.relation2id[rel]
                    tail = g2l[self.entity2id[obj]]
            except:
                head = g2l[sbj]
                try:
                    rel = int(rel)
                except:
                    rel = self.relation2id[rel]
                tail = g2l[obj]
            head_list.append(head)
            rel_list.append(rel)
            tail_list.append(tail)
            if self.use_inverse_relation:
                if self.enable_policy_expand and rel >= len(self.relation2id):
                    head_list.append(tail)
                    rel_list.append(rel - len(self.relation2id))
                    tail_list.append(head)
                else:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)

        return np.array(head_list, dtype=int),  np.array(rel_list, dtype=int), np.array(tail_list, dtype=int)

    # [shared-subgraph]
    def _build_fact_mat_shared(self, batch_size: int, fact_dropout: float):
        """
        Build the batched edge list for shared-subgraph mode.

        NOTE: The downstream GNN expects a *flattened* graph for (batch_size * |V|) nodes,
        so we still replicate edge indices per-batch via offsets. We avoid any
        (num_data x |V|) persistent allocations.
        """
        base_heads = self._shared_base_heads
        base_rels = self._shared_base_rels
        base_tails = self._shared_base_tails

        num_fact = int(base_heads.shape[0])
        if num_fact == 0:
            batch_heads = np.array([], dtype=np.int64)
            batch_rels = np.array([], dtype=np.int64)
            batch_tails = np.array([], dtype=np.int64)
            batch_ids = np.array([], dtype=np.int64)
            fact_ids = np.array([], dtype=np.int64)
            weight_list = np.array([], dtype=np.float32)
            weight_rel_list = np.array([], dtype=np.float32)
            return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list

        keep = int(np.floor(num_fact * (1.0 - float(fact_dropout))))
        keep = max(0, min(num_fact, keep))
        if keep == num_fact:
            mask_index = slice(None)
        else:
            mask_index = np.random.permutation(num_fact)[:keep]

        kept_heads = base_heads[mask_index]
        kept_rels = base_rels[mask_index]
        kept_tails = base_tails[mask_index]

        offsets = (np.arange(batch_size, dtype=np.int64) * int(self.max_local_entity))[:, None]
        batch_heads = (kept_heads[None, :] + offsets).reshape(-1)
        batch_tails = (kept_tails[None, :] + offsets).reshape(-1)
        batch_rels = np.tile(kept_rels, batch_size)
        batch_ids = np.repeat(np.arange(batch_size, dtype=np.int64), kept_heads.shape[0])

        if self.use_self_loop:
            ent = np.arange(self.max_local_entity, dtype=np.int64)[None, :] + offsets
            ent = ent.reshape(-1)
            rel = np.array([self.num_kb_relation - 1], dtype=np.int64)
            rel = np.tile(rel, ent.shape[0])
            ids = np.repeat(np.arange(batch_size, dtype=np.int64), self.max_local_entity)
            batch_heads = np.append(batch_heads, ent)
            batch_tails = np.append(batch_tails, ent)
            batch_rels = np.append(batch_rels, rel)
            batch_ids = np.append(batch_ids, ids)

        fact_ids = np.arange(batch_heads.shape[0], dtype=np.int64)

        # Weights are used only when normalized_gnn/norm_rel are enabled.
        if not self.normalized_gnn:
            weight_list = np.ones(batch_heads.shape[0], dtype=np.float32)
        else:
            head_count = Counter(batch_heads.tolist())
            weight_list = np.asarray([1.0 / head_count[int(h)] for h in batch_heads], dtype=np.float32)

        if not self.norm_rel:
            weight_rel_list = np.ones(batch_heads.shape[0], dtype=np.float32)
        else:
            head_rels_batch = list(zip(batch_heads.tolist(), batch_rels.tolist()))
            head_rels_count = Counter(head_rels_batch)
            weight_rel_list = np.asarray(
                [1.0 / head_rels_count[(int(h), int(r))] for (h, r) in head_rels_batch],
                dtype=np.float32,
            )

        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list

    
    def _build_fact_mat(self, sample_ids, fact_dropout):
        """
        Creates local adj mats that contain entities, relations, and structure.
        """
        # [shared-subgraph]
        if getattr(self, "use_shared_subgraph", False):
            return self._build_fact_mat_shared(len(sample_ids), fact_dropout=fact_dropout)
        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        #print(sample_ids)
        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            if self.data_eff:
                head_list, rel_list, tail_list = self.create_kb_adj_mats(sample_id) #kb_adj_mats[sample_id]
            else:
                (head_list, rel_list, tail_list) = self.kb_adj_mats[sample_id]
            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]

            real_head_list = head_list[mask_index] + index_bias
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                ent_array = np.array(range(num_ent_now), dtype=int) + index_bias
                rel_array = np.array([self.num_kb_relation - 1] * num_ent_now, dtype=int)
                batch_heads = np.append(batch_heads, ent_array)
                batch_tails = np.append(batch_tails, ent_array)
                batch_rels = np.append(batch_rels, rel_array)
                batch_ids = np.append(batch_ids, np.full(num_ent_now, i, dtype=int))
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_rels_ids = zip(batch_heads, batch_rels)
        head_count = Counter(batch_heads)
        # tail_count = Counter(batch_tails)
        weight_list = [1.0 / head_count[head] for head in batch_heads]

        
        head_rels_batch = list(zip(batch_heads, batch_rels))
        #print(head_rels_batch)
        head_rels_count = Counter(head_rels_batch)
        weight_rel_list = [1.0 / head_rels_count[(h,r)] for (h,r) in head_rels_batch]

        #print(head_rels_count)

        # tail_count = Counter(batch_tails)

        # entity2fact_index = torch.LongTensor([batch_heads, fact_ids])
        # entity2fact_val = torch.FloatTensor(weight_list)
        # entity2fact_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
        #     [len(sample_ids) * self.max_local_entity, len(batch_heads)]))
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list


    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            if 'entities_cid' in sample:
                self._add_entity_to_map(self.entity2id, sample['entities_cid'], g2l)
            else:
                self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            #self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps



    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        #print(entities)
        #print(entity2id)
        for entity_global_id in entities:
            try:
                if isinstance(entity_global_id, dict) and 'text' in entity_global_id:
                    ent = entity2id[entity_global_id['text']]
                else:
                    ent = entity2id[entity_global_id]
                if ent not in g2l:
                    g2l[ent] = len(g2l)
            except:
                if entity_global_id not in g2l:
                    g2l[entity_global_id] = len(g2l)

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = self.query_texts[sample_ids]
        else:
            raise NotImplementedError
        
        return q_input

    



class SingleDataLoader(BasicDataLoader):
    """
    Single Dataloader creates training/eval batches during KGQA.
    """
    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, tokenize, data_type)
        
    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None

        # [shared-subgraph]
        if getattr(self, "use_shared_subgraph", False):
            local_entity, query_entities, seed_dist, answer_dist = self._shared_build_dense_batch(sample_ids)
            q_input = self.deal_q_type(q_type)
            kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
            if iteration == 0:
                num_fact_base = int(self._shared_base_heads.shape[0])
                keep = int(np.floor(num_fact_base * (1.0 - float(fact_dropout))))
                print(
                    f"[shared-subgraph] split={self.data_type} fact_dropout={fact_dropout} "
                    f"keep_facts_per_sample={keep}/{num_fact_base} batch_size={len(sample_ids)}"
                )
            if test:
                return local_entity, query_entities, kb_adj_mats, q_input, seed_dist, true_batch_id, answer_dist, self.answer_lists[sample_ids]
            return local_entity, query_entities, kb_adj_mats, q_input, seed_dist, true_batch_id, answer_dist

        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
        
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mats, \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mats, \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def load_dict_int(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[int(word)] = int(word)
    return word2id

def load_data(config, tokenize):

    """
    Creates train/val/test dataloaders (seperately).
    """
    if 'sr-cwq' in config['data_folder']:
        entity2id = load_dict_int(config['data_folder'] + config['entity2id'])
    else:
        entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    
    if config["is_eval"]:
        train_data = None
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = test_data.num_word
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = train_data.num_word
    relation_texts = test_data.rel_texts
    relation_texts_inv = test_data.rel_texts_inv
    entities_texts = None
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data, #test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_word": num_word,
        "rel_texts": relation_texts,
        "rel_texts_inv": relation_texts_inv,
        "ent_texts": entities_texts
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    #args = get_config()
    load_data(args)
