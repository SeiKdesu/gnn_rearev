import argparse
import sys


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_shared_args(parser, *, suppress_defaults: bool = False):
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument('--name', default='webqsp' if default is None else default, type=str)
    parser.add_argument('--data_folder', default='data/webqsp/' if default is None else default, type=str)
    parser.add_argument('--max_train', default=200000 if default is None else default, type=int)
    parser.add_argument(
        '--data_file_train',
        default=default,
        type=str,
        help='Override train file (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--data_file_dev',
        default=default,
        type=str,
        help='Override dev file (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--data_file_test',
        default=default,
        type=str,
        help='Override test file (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--switch_epoch',
        default=default,
        type=int,
        help='Epoch index to start using *_switch files (0-based). Example: 20 switches after 20 epochs (starts at epoch 21).',
    )
    parser.add_argument(
        '--data_file_train_switch',
        default=default,
        type=str,
        help='Train file to use after switch_epoch (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--data_file_dev_switch',
        default=default,
        type=str,
        help='Dev file to use after switch_epoch (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--data_file_test_switch',
        default=default,
        type=str,
        help='Test file to use after switch_epoch (relative to --data_folder unless absolute)',
    )

    # embeddings
    parser.add_argument('--word2id', default='vocab.txt' if default is None else default, type=str)
    parser.add_argument('--relation2id', default='relations.txt' if default is None else default, type=str)
    parser.add_argument('--entity2id', default='entities.txt' if default is None else default, type=str)
    parser.add_argument('--char2id', default='chars.txt' if default is None else default, type=str)
    parser.add_argument('--entity_emb_file', default=default, type=str)
    parser.add_argument('--relation_emb_file', default=default, type=str)
    parser.add_argument('--relation_word_emb', default=True if default is None else default, type=bool_flag)
    parser.add_argument('--word_emb_file', default='word_emb.npy' if default is None else default, type=str)
    parser.add_argument('--rel_word_ids', default='rel_word_idx.npy' if default is None else default, type=str)
    parser.add_argument('--kge_frozen', default=0 if default is None else default, type=int)
    parser.add_argument(
        '--lm',
        default='lstm' if default is None else default,
        type=str,
        choices=['lstm', 'bert', 'roberta', 'sbert', 't5', 'sbert2', 'dbert', 'simcse', 'relbert'],
    )
    parser.add_argument('--lm_frozen', default=1 if default is None else default, type=int)

    # dimensions, layers, dropout
    parser.add_argument('--entity_dim', default=50 if default is None else default, type=int)
    parser.add_argument('--kg_dim', default=100 if default is None else default, type=int)
    parser.add_argument('--word_dim', default=300 if default is None else default, type=int)
    parser.add_argument('--lm_dropout', default=0.3 if default is None else default, type=float)
    parser.add_argument('--linear_dropout', default=0.2 if default is None else default, type=float)

    # optimization
    parser.add_argument('--num_epoch', default=100 if default is None else default, type=int)
    parser.add_argument('--warmup_epoch', default=0 if default is None else default, type=int)
    parser.add_argument('--fact_scale', default=3 if default is None else default, type=int)
    parser.add_argument('--eval_every', default=2 if default is None else default, type=int)
    parser.add_argument('--batch_size', default=4 if default is None else default, type=int)
    parser.add_argument(
        '--batch_size_switch',
        default=default,
        type=int,
        help='Batch size to use after switch_epoch (if set).',
    )
    parser.add_argument('--gradient_clip', default=1.0 if default is None else default, type=float)
    parser.add_argument('--lr', default=0.0005 if default is None else default, type=float)
    parser.add_argument('--decay_rate', default=0.0 if default is None else default, type=float)
    parser.add_argument('--seed', default=19960626 if default is None else default, type=int)
    parser.add_argument('--lr_schedule', action='store_true', default=False if default is None else default)
    parser.add_argument('--label_smooth', default=0.1 if default is None else default, type=float)
    parser.add_argument('--fact_drop', default=0 if default is None else default, type=float)
    #parser.add_argument('--encode_type', action='store_true')

    # model options

    parser.add_argument('--is_eval', action='store_true', default=False if default is None else default)
    parser.add_argument('--checkpoint_dir', default='checkpoint/pretrain/' if default is None else default, type=str)
    parser.add_argument('--log_level', type=str, default='info' if default is None else default)
    parser.add_argument('--experiment_name', default='' if default is None else default, type=str)
    parser.add_argument('--load_experiment', default=default, type=str)
    parser.add_argument('--load_ckpt_file', default=default, type=str)
    parser.add_argument('--eps', default=0.95 if default is None else default, type=float) # threshold for f1
    parser.add_argument('--test_batch_size', default=20 if default is None else default, type=int)
    parser.add_argument(
        '--test_batch_size_switch',
        default=default,
        type=int,
        help='Test batch size to use after switch_epoch (if set).',
    )
    parser.add_argument('--q_type', default='seq' if default is None else default, type=str)
    parser.add_argument(
        '--dump_wrong',
        default=False if default is None else default,
        type=bool_flag,
        help='During evaluation, write per-hop debug info for wrong predictions to checkpoint_dir.',
    )
    parser.add_argument(
        '--dump_wrong_topk',
        default=20 if default is None else default,
        type=int,
        help='Top-k candidates to include in wrong-case debug output.',
    )

    # Optional: override per-sample subgraphs using a merged subgraph DB (CWQ)
    parser.add_argument('--use_merged_subgraph', default=False if default is None else default, type=bool_flag)
    parser.add_argument(
        '--merged_subgraph_db',
        default='subgraph_merge.sqlite' if default is None else default,
        type=str,
        help='SQLite DB created by merge_cwq_subgraphs.py (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--merged_subgraph_db_train',
        default=default,
        type=str,
        help='Optional: train split DB (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--merged_subgraph_db_dev',
        default=default,
        type=str,
        help='Optional: dev split DB (relative to --data_folder unless absolute)',
    )
    parser.add_argument(
        '--merged_subgraph_db_test',
        default=default,
        type=str,
        help='Optional: test split DB (relative to --data_folder unless absolute)',
    )
    parser.add_argument('--merged_subgraph_hops', default=2 if default is None else default, type=int)
    parser.add_argument('--merged_subgraph_max_entities', default=800 if default is None else default, type=int)
    parser.add_argument('--merged_subgraph_max_tuples', default=4000 if default is None else default, type=int)
    parser.add_argument('--merged_subgraph_sql_limit', default=20000 if default is None else default, type=int)
    parser.add_argument(
        '--stream_data',
        default=False if default is None else default,
        type=bool_flag,
        help='Low-RAM mode: do not keep full JSON objects in memory; stream and build tensors/arrays',
    )



def add_parse_args(parser):
    
    # Allow shared args (e.g., --data_file_train) to appear either before or after the subcommand.
    # Defaults live on the top-level parser; subparsers suppress defaults to avoid overwriting
    # values parsed at the top-level.
    add_shared_args(parser, suppress_defaults=False)

    subparsers = parser.add_subparsers(help='Reason KGQA model', dest='model')
    if hasattr(subparsers, "required"):
        subparsers.required = True

    parser_rearev = subparsers.add_parser("ReaRev")
    create_parser_rearev(parser_rearev)

    parser_nsm = subparsers.add_parser("NSM")
    create_parser_nsm(parser_nsm)

    parser_graftnet = subparsers.add_parser("GraftNet")
    create_parser_graftnet(parser_graftnet)

    parser_nutrea = subparsers.add_parser("NuTrea")
    #create_parser_nutrea(parser_nutrea)
    create_parser_rearev(parser_nutrea)


def create_parser_rearev(parser):

    parser.add_argument('--model_name', default='ReaRev', type=str, choices=['ReaRev'])
    parser.add_argument('--alg', default='bfs', type=str)
    parser.add_argument('--num_iter', default=2, type=int)
    parser.add_argument('--num_ins', default=3, type=int)
    parser.add_argument('--num_gnn', default=6, type=int)
    parser.add_argument('--loss_type', default='kl', type=str)
    parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    parser.add_argument('--normalized_gnn', default=False, type=bool_flag)
    parser.add_argument('--norm_rel', action='store_true')
    parser.add_argument('--data_eff', action='store_true')
    parser.add_argument('--pos_emb', action='store_true')
    parser.add_argument('--use_beta_crossattn', default=False, type=bool_flag)
    parser.add_argument('--beta_num_heads', default=8, type=int)
    parser.add_argument('--beta_lambda', default=1.0, type=float)
    parser.add_argument('--beta_dropout', default=0.0, type=float)
    parser.add_argument('--beta_fusion_hidden', default=None, type=int)
    parser.add_argument('--debug_beta_topk', default=0, type=int)
    add_shared_args(parser, suppress_defaults=True)


def create_parser_nsm(parser):
    parser.add_argument('--model_name', default='NSM', type=str, choices=['NSM'])
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--reason_kb', default=False, type=bool_flag)
    parser.add_argument('--loss_type', default='kl', type=str)
    parser.add_argument('--lambda_constrain', default=0.0, type=float)
    parser.add_argument('--lambda_back', default=0.0, type=float)
    parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    parser.add_argument('--use_inverse_relation', action='store_true')
    parser.add_argument('--norm_rel', action='store_true')
    parser.add_argument('--normalized_gnn', default=False, type=bool_flag)
    parser.add_argument('--data_eff', action='store_true')
    add_shared_args(parser, suppress_defaults=True)

def create_parser_graftnet(parser):
    parser.add_argument('--model_name', default='GraftNet', type=str, choices=['GraftNet'])
    parser.add_argument('--pagerank_lambda', default=0.8, type=float)
    parser.add_argument('--loss_type', default='bce', type=str)
    parser.add_argument('--num_layer', default=3, type=int)
    parser.add_argument('--use_inverse_relation', action='store_true')
    parser.add_argument('--norm_rel', action='store_true')
    parser.add_argument('--normalized_gnn', default=False, type=bool_flag)
    parser.add_argument('--data_eff', action='store_true')
    #parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    add_shared_args(parser, suppress_defaults=True)
