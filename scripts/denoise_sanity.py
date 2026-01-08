import argparse
import json
import os

from dataset_load import load_dict, load_dict_int
from modules.denoise_subgraph import SubgraphDenoiser
from parsing import bool_flag


def entity_key(ent):
    if isinstance(ent, dict):
        if "text" in ent:
            return ent["text"]
        if "kb_id" in ent:
            return ent["kb_id"]
    return ent


def tuple_entity_keys(tuples):
    keys = set()
    for sbj, _, obj in tuples:
        keys.add(entity_key(sbj))
        keys.add(entity_key(obj))
    return keys


def count_nodes(tuples):
    nodes = set()
    for sbj, _, obj in tuples:
        nodes.add(entity_key(sbj))
        nodes.add(entity_key(obj))
    return len(nodes)


def extract_answer_keys(sample):
    answers = []
    if "answers_cid" in sample:
        for ans in sample.get("answers_cid", []):
            answers.append(entity_key(ans))
    else:
        for ans in sample.get("answers", []):
            if isinstance(ans, dict):
                if "kb_id" in ans:
                    answers.append(ans["kb_id"])
                elif "text" in ans:
                    answers.append(ans["text"])
            else:
                answers.append(ans)
    return set(answers)


def answer_in_tuples(answer_keys, tuples):
    if not answer_keys:
        return False
    tuple_nodes = tuple_entity_keys(tuples)
    return bool(answer_keys & tuple_nodes)


def build_undirected_adj(tuples):
    adj = {}
    for sbj, _, obj in tuples:
        s_key = entity_key(sbj)
        o_key = entity_key(obj)
        adj.setdefault(s_key, set()).add(o_key)
        adj.setdefault(o_key, set()).add(s_key)
    return adj


def reachable_within_hops(adj, start_nodes, target_nodes, max_hops):
    if not start_nodes or not target_nodes:
        return False
    if start_nodes & target_nodes:
        return True
    visited = set(start_nodes)
    frontier = set(start_nodes)
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for nbr in adj.get(node, set()):
                if nbr in visited:
                    continue
                if nbr in target_nodes:
                    return True
                visited.add(nbr)
                next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    return False


def load_relation2id(data_folder, relation2id_file):
    path = os.path.join(data_folder, relation2id_file)
    if "sr-cwq" in data_folder:
        return load_dict_int(path)
    return load_dict(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data/webqsp/", type=str)
    parser.add_argument("--split", default="dev", type=str, choices=["train", "dev", "test"])
    parser.add_argument("--sample_idx", default=0, type=int)
    parser.add_argument("--relation2id", default="relations.txt", type=str)

    parser.add_argument("--enable_denoise", default=True, type=bool_flag)
    parser.add_argument("--encoder_type", default="tfidf", type=str)
    parser.add_argument("--encoder_model_name", default=None, type=str)
    parser.add_argument("--relation_desc_path", default=None, type=str)
    parser.add_argument("--topM_strict", default=10, type=int)
    parser.add_argument("--topK_strict", default=50, type=int)
    parser.add_argument("--gamma_strict", default=0.2, type=float)
    parser.add_argument("--topM_loose", default=25, type=int)
    parser.add_argument("--topK_loose", default=150, type=int)
    parser.add_argument("--gamma_loose", default=0.05, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.2, type=float)
    parser.add_argument("--min_edges", default=50, type=float)
    parser.add_argument("--enable_pair_score", default=False, type=bool_flag)
    parser.add_argument("--pair_stats_path", default=None, type=str)
    parser.add_argument("--ensure_connectivity", default=True, type=bool_flag)
    parser.add_argument("--denoise_sim_threshold", default=0.2, type=float)
    parser.add_argument("--denoise_sim_threshold_min", default=-1.0, type=float)
    parser.add_argument("--denoise_sim_threshold_step", default=0.05, type=float)
    parser.add_argument("--denoise_max_hops", default=4, type=int)

    args = parser.parse_args()

    sample_path = os.path.join(args.data_folder, f"{args.split}.json")
    sample = None
    with open(sample_path, "r", encoding="utf-8") as f_in:
        for idx, line in enumerate(f_in):
            if idx == args.sample_idx:
                sample = json.loads(line)
                break
    if sample is None:
        raise ValueError("sample_idx {} out of range for {}".format(args.sample_idx, sample_path))

    relation2id = load_relation2id(args.data_folder, args.relation2id)
    denoiser = SubgraphDenoiser(relation2id, vars(args))

    original_tuples = sample["subgraph"]["tuples"]
    original_entities = sample["subgraph"].get("entities")
    original_entity_count = (
        len(original_entities) if original_entities is not None else count_nodes(original_tuples)
    )
    answers_for_denoiser = sample.get("answers_cid") or sample.get("answers") or []
    denoised = denoiser.denoise(
        sample["question"],
        sample["subgraph"],
        topic_entities=sample.get("entities_cid", sample.get("entities", [])),
        answer_entities=answers_for_denoiser,
    )
    denoised_tuples = denoised["tuples"]
    denoised_entities = denoised.get("entities")
    denoised_entity_count = (
        len(denoised_entities) if denoised_entities is not None else count_nodes(denoised_tuples)
    )

    answer_keys = extract_answer_keys(sample)
    original_answer_in = answer_in_tuples(answer_keys, original_tuples)
    denoised_answer_in = answer_in_tuples(answer_keys, denoised_tuples)

    topic_entities = sample.get("entities_cid", sample.get("entities", []))
    topic_keys = {entity_key(ent) for ent in topic_entities}
    original_adj = build_undirected_adj(original_tuples)
    denoised_adj = build_undirected_adj(denoised_tuples)
    original_reachable = reachable_within_hops(original_adj, topic_keys, answer_keys, 3)
    denoised_reachable = reachable_within_hops(denoised_adj, topic_keys, answer_keys, 3)

    print("Question:", sample.get("question", ""))
    print("Original edges:", len(original_tuples), "entities:", original_entity_count)
    print("Denoised edges:", len(denoised_tuples), "entities:", denoised_entity_count)
    if original_tuples:
        reduction = 100.0 * (1.0 - (len(denoised_tuples) / len(original_tuples)))
        print("Edge reduction: {:.1f}%".format(reduction))
    if original_entity_count:
        ent_reduction = 100.0 * (1.0 - (denoised_entity_count / original_entity_count))
        print("Entity reduction: {:.1f}%".format(ent_reduction))
    print("Answer in tuples (original):", original_answer_in)
    print("Answer in tuples (denoised):", denoised_answer_in)
    print("Answer reachable within 3 hops (original, undirected):", original_reachable)
    print("Answer reachable within 3 hops (denoised, undirected):", denoised_reachable)


if __name__ == "__main__":
    main()
