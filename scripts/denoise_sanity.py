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


def count_nodes(tuples):
    nodes = set()
    for sbj, _, obj in tuples:
        nodes.add(entity_key(sbj))
        nodes.add(entity_key(obj))
    return len(nodes)


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
    original_nodes = count_nodes(original_tuples)
    denoised = denoiser.denoise(
        sample["question"],
        sample["subgraph"],
        topic_entities=sample.get("entities_cid", sample.get("entities", [])),
    )
    denoised_tuples = denoised["tuples"]
    denoised_nodes = count_nodes(denoised_tuples)

    print("Question:", sample.get("question", ""))
    print("Original edges:", len(original_tuples), "nodes:", original_nodes)
    print("Denoised edges:", len(denoised_tuples), "nodes:", denoised_nodes)
    if original_tuples:
        reduction = 100.0 * (1.0 - (len(denoised_tuples) / len(original_tuples)))
        print("Edge reduction: {:.1f}%".format(reduction))


if __name__ == "__main__":
    main()
