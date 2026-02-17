import json
import numpy as np
import re
from tqdm import tqdm
import torch
from collections import Counter
import random
import warnings
import pickle
import sqlite3
warnings.filterwarnings("ignore")
from modules.question_encoding.tokenizers import LSTMTokenizer#, BERTTokenizer
from transformers import AutoTokenizer
import time

import os


def _scan_json_value_end(s: str, start: int) -> int:
    """
    Return end position (exclusive) of the JSON value starting at `start`.
    Works for objects/arrays/strings/numbers/true/false/null.
    """
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

    # number / true / false / null: scan until comma or object end
    i = start
    while i < n and s[i] not in ",}":
        i += 1
    return i


def _strip_top_level_key_json_line(line: str, key: str) -> str:
    """
    Remove a top-level key from a JSON object line without fully parsing (fast path).
    Used to skip parsing huge 'subgraph' fields when we will replace them anyway.
    """
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

                # trailing comma (only remove if key is first)
                t = val_end
                while t < n and line[t].isspace():
                    t += 1
                has_trailing_comma = t < n and line[t] == ","

                # preceding comma (remove if key is not first)
                p = i - 1
                while p >= 0 and line[p].isspace():
                    p -= 1
                has_preceding_comma = p >= 0 and line[p] == ","

                if has_preceding_comma:
                    remove_start = p
                    remove_end = val_end  # keep trailing comma if any
                else:
                    remove_start = i
                    remove_end = (t + 1) if has_trailing_comma else val_end

                return line[:remove_start] + line[remove_end:]

            in_str = True
            i += 1
            continue

        i += 1

    return line


class BasicDataLoader(object):
    """ 
    Basic Dataloader contains all the functions to read questions and KGs from json files and
    create mappings between global entity ids and local ids that are used during GNN updates.
    """

    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        # Keep config for streaming mode / split-specific settings.
        self.config = config
        self.tokenize = tokenize
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)
        self._load_data()
        

    def _load_file(self, config, data_type="train"):

        """
        Loads lines (questions + KG subgraphs) from json files.
        """

        override_key = {
            "train": "data_file_train",
            "dev": "data_file_dev",
            "test": "data_file_test",
        }.get(data_type)
        override_path = config.get(override_key) if override_key else None
        if override_path:
            data_file = override_path if os.path.isabs(override_path) else os.path.join(config["data_folder"], override_path)
        else:
            data_file = os.path.join(config["data_folder"], f"{data_type}.json")

        self.data_file = data_file
        print('loading data from', data_file)
        self.data_type = data_type

        if getattr(self, "stream_data", False):
            if self.data_eff:
                print("WARNING: --stream_data forces --data_eff False (subgraphs are not kept in memory).")
                self.data_eff = False
            self._scan_file_stats(config, data_type, data_file)
            self.data = None
            return

        self.data = []
        skip_index = set()
        index = 0

        if getattr(self, "use_merged_subgraph", False):
            self._set_merged_subgraph_db_path_for_split(config, data_type)
            self._open_merged_subgraph_db()

        with open(data_file) as f_in:
            for raw_line in tqdm(f_in):
                if index == config['max_train'] and data_type == "train": break  #break if we reach max_question_size
                if getattr(self, "use_merged_subgraph", False):
                    raw_line = _strip_top_level_key_json_line(raw_line, "subgraph")
                line = json.loads(raw_line)

                if len(line['entities']) == 0:
                    skip_index.add(index)
                    continue

                if getattr(self, "use_merged_subgraph", False):
                    seed_key = 'entities_cid' if 'entities_cid' in line else 'entities'
                    seed_entities = [self._to_global_entity_id(e) for e in line.get(seed_key, [])]
                    seed_entities = [e for e in seed_entities if isinstance(e, int)]
                    line['subgraph'] = self._extract_subgraph_from_merged(seed_entities)

                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                index += 1

        print("skip", skip_index)
        print('max_facts: ', self.max_facts)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

    def _scan_file_stats(self, config, data_type: str, data_file: str) -> None:
        """
        Low-RAM pre-scan: count examples and compute max sizes without keeping JSON objects.
        """
        max_train = int(config.get("max_train", 200000))
        kept = 0
        max_query_words = 0
        max_local_entity = 0
        max_facts = 0

        # If we will overwrite subgraphs from merged DB, use caps for allocation.
        use_merged = getattr(self, "use_merged_subgraph", False)
        if use_merged:
            max_local_entity = int(getattr(self, "merged_subgraph_max_entities", 800))
            max_facts = 2 * int(getattr(self, "merged_subgraph_max_tuples", 4000))

        with open(data_file) as f_in:
            for raw_line in tqdm(f_in):
                if kept == max_train and data_type == "train":
                    break
                if use_merged:
                    raw_line = _strip_top_level_key_json_line(raw_line, "subgraph")
                obj = json.loads(raw_line)

                if len(obj.get("entities", [])) == 0:
                    continue

                q = obj.get("question", "")
                max_query_words = max(max_query_words, len(q.split(" ")))

                if not use_merged:
                    g2l = dict()
                    seed_key = "entities_cid" if "entities_cid" in obj else "entities"
                    self._add_entity_to_map(self.entity2id, obj.get(seed_key, []) or [], g2l)
                    sg = obj.get("subgraph") or {}
                    self._add_entity_to_map(self.entity2id, sg.get("entities") or [], g2l)
                    max_local_entity = max(max_local_entity, len(g2l))
                    max_facts = max(max_facts, 2 * len(sg.get("tuples") or []))

                kept += 1

        self.num_data = kept
        self.batches = np.arange(self.num_data)
        self.max_facts = max_facts
        self._stream_max_query_words = max_query_words
        self.max_local_entity = max_local_entity

        print(f"[stream_data] num_data={self.num_data} max_local_entity={self.max_local_entity} max_facts={self.max_facts}")

    def _load_data(self):

        """
        Creates mappings between global entity ids and local entity ids that are used during GNN updates.
        """

        if getattr(self, "stream_data", False):
            self._load_data_streaming()
            return

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full(
            (self.num_data, self.max_local_entity),
            len(self.entity2id),
            dtype=np.int32,
        )
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=np.float32)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=np.float32)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=np.float32)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        self._prepare_data()

    def _load_data_streaming(self) -> None:
        """
        Low-RAM loader: stream JSONL twice and build numpy arrays without keeping full JSON objects.
        Requires data_eff == False (enforced in _load_file).
        """
        config = self.config

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        # Use pre-scanned max words
        max_count = int(getattr(self, "_stream_max_query_words", 0))

        if self.rel_word_emb:
            self.build_rel_words(self.tokenize)
        else:
            self.rel_texts = None
            self.rel_texts_inv = None
            self.ent_texts = None

        self.max_query_word = max_count
        if self.tokenize == 'lstm':
            self.num_word = len(self.word2id)
            self.tokenizer = LSTMTokenizer(self.word2id, self.max_query_word)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=np.int32)
        else:
            if self.tokenize == 'bert':
                tokenizer_name = 'bert-base-uncased'
            elif self.tokenize == 'roberta':
                tokenizer_name = 'roberta-base'
            elif self.tokenize == 'sbert':
                tokenizer_name = 'sentence-transformers/all-MiniLM-L6-v2'
            elif self.tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif self.tokenize == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif self.tokenize == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'
            else:
                tokenizer_name = 'bert-base-uncased'

            self.max_query_word = max_count + 2
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.num_word = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=np.int32)

        self.question_id = []
        self.candidate_entities = np.full(
            (self.num_data, self.max_local_entity),
            len(self.entity2id),
            dtype=np.int32,
        )
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=np.float32)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=np.float32)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=np.float32)
        self.answer_lists = np.empty(self.num_data, dtype=object)
        self.local_entity_counts = np.zeros(self.num_data, dtype=np.int32)

        node_dtype = np.uint16 if self.max_local_entity <= np.iinfo(np.uint16).max else np.int32
        rel_dtype = np.uint16 if self.num_kb_relation <= np.iinfo(np.uint16).max else np.int32

        # Second pass: fill arrays
        max_train = int(config.get("max_train", 200000))
        data_type = getattr(self, "data_type", "train")
        use_merged = getattr(self, "use_merged_subgraph", False)
        if use_merged:
            # ensure correct split DB is set
            self._set_merged_subgraph_db_path_for_split(config, data_type)
            self._open_merged_subgraph_db()

        next_id = 0
        num_query_entity = {}
        with open(self.data_file) as f_in:
            for raw_line in tqdm(f_in):
                if next_id == max_train and data_type == "train":
                    break
                if use_merged:
                    raw_line = _strip_top_level_key_json_line(raw_line, "subgraph")
                sample = json.loads(raw_line)

                if len(sample.get("entities", [])) == 0:
                    continue

                if use_merged:
                    seed_key = 'entities_cid' if 'entities_cid' in sample else 'entities'
                    seed_entities = [self._to_global_entity_id(e) for e in sample.get(seed_key, [])]
                    seed_entities = [e for e in seed_entities if isinstance(e, int)]
                    sample['subgraph'] = self._extract_subgraph_from_merged(seed_entities)

                self.question_id.append(sample.get("id"))

                # build g2l map for this sample
                g2l = {}
                seed_key = 'entities_cid' if 'entities_cid' in sample else 'entities'
                self._add_entity_to_map(self.entity2id, sample.get(seed_key, []) or [], g2l)
                self._add_entity_to_map(self.entity2id, (sample.get('subgraph') or {}).get('entities') or [], g2l)

                self.local_entity_counts[next_id] = len(g2l)
                tp_set = set()

                for entity in sample.get(seed_key, []) or []:
                    global_entity = self._to_global_entity_id(entity)
                    if global_entity not in g2l:
                        continue
                    local_ent = g2l[global_entity]
                    self.query_entities[next_id, local_ent] = 1.0
                    tp_set.add(local_ent)

                num_query_entity[next_id] = len(tp_set)

                for global_entity, local_entity in g2l.items():
                    if self.data_name != 'cwq':
                        if local_entity not in tp_set:
                            self.candidate_entities[next_id, local_entity] = global_entity
                    else:
                        self.candidate_entities[next_id, local_entity] = global_entity

                # relations in local KB
                head_list = []
                rel_list = []
                tail_list = []
                for tpl in (sample.get('subgraph') or {}).get('tuples') or []:
                    sbj, rel, obj = tpl
                    try:
                        head = g2l[int(sbj)]
                        tail = g2l[int(obj)]
                    except Exception:
                        continue
                    try:
                        rel_id = int(rel)
                    except Exception:
                        try:
                            rel_id = self.relation2id[rel]
                        except Exception:
                            continue
                    head_list.append(head)
                    rel_list.append(rel_id)
                    tail_list.append(tail)
                    if self.use_inverse_relation:
                        head_list.append(tail)
                        rel_list.append(rel_id + len(self.relation2id))
                        tail_list.append(head)

                if len(tp_set) > 0:
                    for local_ent in tp_set:
                        self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
                else:
                    if len(g2l) > 0:
                        self.seed_distribution[next_id, : len(g2l)] = 1.0 / len(g2l)

                # tokenize question
                if self.tokenize == 'lstm':
                    self.query_texts[next_id] = self.tokenizer.tokenize(sample.get('question', ''))
                else:
                    tokens = self.tokenizer.encode_plus(
                        text=sample.get('question', ''),
                        max_length=self.max_query_word,
                        pad_to_max_length=True,
                        return_attention_mask=False,
                        truncation=True,
                    )
                    self.query_texts[next_id] = np.array(tokens['input_ids'], dtype=np.int32)

                # construct distribution for answers
                answer_list = []
                if 'answers_cid' in sample:
                    for answer_ent in sample.get('answers_cid') or []:
                        answer_list.append(answer_ent)
                        if answer_ent in g2l:
                            self.answer_dists[next_id, g2l[answer_ent]] = 1.0
                else:
                    for answer in sample.get('answers') or []:
                        keyword = 'text' if type(answer.get('kb_id')) == int else 'kb_id'
                        if keyword not in answer:
                            continue
                        if answer[keyword] not in self.entity2id:
                            continue
                        answer_ent = self.entity2id[answer[keyword]]
                        answer_list.append(answer_ent)
                        if answer_ent in g2l:
                            self.answer_dists[next_id, g2l[answer_ent]] = 1.0

                self.answer_lists[next_id] = answer_list

                # store adjacency for fast batches (use compact dtypes)
                self.kb_adj_mats[next_id] = (
                    np.asarray(head_list, dtype=node_dtype),
                    np.asarray(rel_list, dtype=rel_dtype),
                    np.asarray(tail_list, dtype=node_dtype),
                )

                next_id += 1

        # In case we stopped early (max_train), trim arrays.
        self.num_data = next_id
        self.batches = np.arange(self.num_data)
        self.candidate_entities = self.candidate_entities[: self.num_data]
        self.query_entities = self.query_entities[: self.num_data]
        self.seed_distribution = self.seed_distribution[: self.num_data]
        self.answer_dists = self.answer_dists[: self.num_data]
        self.answer_lists = self.answer_lists[: self.num_data]
        self.query_texts = self.query_texts[: self.num_data]
        self.kb_adj_mats = self.kb_adj_mats[: self.num_data]
        self.local_entity_counts = self.local_entity_counts[: self.num_data]

        num_no_query_ent = sum(1 for v in num_query_entity.values() if v == 0)
        num_one_query_ent = sum(1 for v in num_query_entity.values() if v == 1)
        num_multiple_ent = sum(1 for v in num_query_entity.values() if v > 1)
        print(
            "{} cases in total, {} cases without query entity, {} cases with single query entity,"
            " {} cases with multiple query entities".format(
                self.num_data, num_no_query_ent, num_one_query_ent, num_multiple_ent
            )
        )

    def _parse_args(self, config, word2id, relation2id, entity2id):

        """
        Builds necessary dictionaries and stores arguments.
        """
        self.data_eff = config['data_eff']
        self.data_name = config['name']
        self.stream_data = config.get('stream_data', False)
        self.use_merged_subgraph = config.get('use_merged_subgraph', False)
        if self.use_merged_subgraph:
            db_arg = config.get('merged_subgraph_db', 'subgraph_merge.sqlite')
            if os.path.isabs(db_arg):
                self.merged_subgraph_db_path = db_arg
            else:
                self.merged_subgraph_db_path = os.path.join(config['data_folder'], db_arg)
            self.merged_subgraph_hops = int(config.get('merged_subgraph_hops', 2))
            self.merged_subgraph_max_entities = int(config.get('merged_subgraph_max_entities', 800))
            self.merged_subgraph_max_tuples = int(config.get('merged_subgraph_max_tuples', 4000))
            self.merged_subgraph_sql_limit = int(config.get('merged_subgraph_sql_limit', 20000))
            self._merged_db_conn = None

        if 'use_inverse_relation' in config:
            self.use_inverse_relation = config['use_inverse_relation']
        else:
            self.use_inverse_relation = False
        if 'use_self_loop' in config:
            self.use_self_loop = config['use_self_loop']
        else:
            self.use_self_loop = False

        self.rel_word_emb = config['relation_word_emb']
        self.normalized_gnn = config.get('normalized_gnn', False)
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

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    
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
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)
                
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
                head_list.append(tail)
                rel_list.append(rel + len(self.relation2id))
                tail_list.append(head)

        return np.array(head_list, dtype=int),  np.array(rel_list, dtype=int), np.array(tail_list, dtype=int)

    
    def _build_fact_mat(self, sample_ids, fact_dropout):
        """
        Creates local adj mats that contain entities, relations, and structure.
        """
        heads_parts = []
        rels_parts = []
        tails_parts = []
        ids_parts = []

        max_local_entity = self.max_local_entity
        num_relation = self.num_kb_relation

        # Fast path: if no dropout, avoid costly permutations.
        use_dropout = fact_dropout is not None and fact_dropout > 0.0

        for i, sample_id in enumerate(sample_ids):
            index_bias = i * max_local_entity
            if self.data_eff:
                head_arr, rel_arr, tail_arr = self.create_kb_adj_mats(sample_id)
            else:
                head_arr, rel_arr, tail_arr = self.kb_adj_mats[sample_id]

            if use_dropout and len(head_arr) > 0:
                num_fact = len(head_arr)
                num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
                if num_keep_fact < num_fact:
                    mask_index = np.random.choice(num_fact, size=num_keep_fact, replace=False)
                    head_arr = head_arr[mask_index]
                    rel_arr = rel_arr[mask_index]
                    tail_arr = tail_arr[mask_index]

            if len(head_arr) > 0:
                # Stored adjacencies may use compact dtypes (e.g., uint16). Upcast before adding index_bias.
                head_arr = np.asarray(head_arr, dtype=np.int32)
                rel_arr = np.asarray(rel_arr, dtype=np.int32)
                tail_arr = np.asarray(tail_arr, dtype=np.int32)
                heads_parts.append(head_arr + index_bias)
                rels_parts.append(rel_arr)
                tails_parts.append(tail_arr + index_bias)
                ids_parts.append(np.full(len(head_arr), i, dtype=int))

            if self.use_self_loop:
                if hasattr(self, "local_entity_counts") and self.local_entity_counts is not None:
                    num_ent_now = int(self.local_entity_counts[sample_id])
                else:
                    num_ent_now = len(self.global2local_entity_maps[sample_id])
                if num_ent_now > 0:
                    ent_array = np.arange(num_ent_now, dtype=int) + index_bias
                    rel_array = np.full(num_ent_now, num_relation - 1, dtype=int)
                    heads_parts.append(ent_array)
                    rels_parts.append(rel_array)
                    tails_parts.append(ent_array)
                    ids_parts.append(np.full(num_ent_now, i, dtype=int))

        if heads_parts:
            batch_heads = np.concatenate(heads_parts).astype(int, copy=False)
            batch_rels = np.concatenate(rels_parts).astype(int, copy=False)
            batch_tails = np.concatenate(tails_parts).astype(int, copy=False)
            batch_ids = np.concatenate(ids_parts).astype(int, copy=False)
        else:
            batch_heads = np.array([], dtype=int)
            batch_rels = np.array([], dtype=int)
            batch_tails = np.array([], dtype=int)
            batch_ids = np.array([], dtype=int)

        fact_ids = np.arange(len(batch_heads), dtype=int)

        if getattr(self, "normalized_gnn", False) and len(batch_heads) > 0:
            max_nodes = len(sample_ids) * max_local_entity
            head_counts = np.bincount(batch_heads, minlength=max_nodes).astype(np.float32)
            weight_list = (1.0 / head_counts[batch_heads]).astype(np.float32)
        else:
            weight_list = None

        # Not used by current models (kept for interface compatibility)
        weight_rel_list = None

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

    def _to_global_entity_id(self, entity_val):
        """
        Convert various entity representations in the dataset into the global integer ID.
        CWQ typically stores entity IDs as ints already; other datasets may store strings.
        """
        try:
            if isinstance(entity_val, dict) and 'text' in entity_val:
                return self.entity2id[entity_val['text']]
            return self.entity2id[entity_val]
        except Exception:
            try:
                return int(entity_val)
            except Exception:
                return entity_val

    def _open_merged_subgraph_db(self):
        if getattr(self, "_merged_db_conn", None) is not None:
            return
        if not os.path.exists(self.merged_subgraph_db_path):
            raise FileNotFoundError(
                f"merged_subgraph_db not found: {self.merged_subgraph_db_path} "
                f"(run merge_cwq_subgraphs.py first, or pass --merged_subgraph_db)"
            )

        conn = sqlite3.connect(self.merged_subgraph_db_path)
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
        conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
        conn.execute("CREATE INDEX IF NOT EXISTS tuples_by_o ON tuples(o, r, s);")
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS frontier (id INTEGER PRIMARY KEY) WITHOUT ROWID;")
        conn.commit()
        self._merged_db_conn = conn

    def _set_merged_subgraph_db_path_for_split(self, config, data_type: str) -> None:
        """
        Allow different merged subgraph DBs per split (train/dev/test).
        """
        key_map = {
            "train": "merged_subgraph_db_train",
            "dev": "merged_subgraph_db_dev",
            "test": "merged_subgraph_db_test",
        }
        override_key = key_map.get(data_type)
        db_arg = None
        if override_key is not None:
            db_arg = config.get(override_key, None)
        if not db_arg:
            db_arg = config.get("merged_subgraph_db", "subgraph_merge.sqlite")

        if os.path.isabs(db_arg):
            self.merged_subgraph_db_path = db_arg
        else:
            self.merged_subgraph_db_path = os.path.join(config["data_folder"], db_arg)

    def _extract_subgraph_from_merged(self, seed_entities):
        """
        Build a per-sample subgraph by k-hop expansion over the merged subgraph DB.
        Uses caps to keep local graphs bounded.
        """
        if not seed_entities:
            return {"entities": [], "tuples": []}
        self._open_merged_subgraph_db()
        conn = self._merged_db_conn

        max_entities = self.merged_subgraph_max_entities
        max_tuples = self.merged_subgraph_max_tuples
        hops = self.merged_subgraph_hops
        sql_limit = self.merged_subgraph_sql_limit

        visited = set(seed_entities)
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

            # Fetch edges incident to the current frontier (both outgoing and incoming).
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

        # Ensure determinism
        return {
            "entities": list(visited),
            "tuples": list(tuples),
        }

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
