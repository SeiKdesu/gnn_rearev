import argparse

from utils import create_logger
import torch
import numpy as np
import os
import time
#from Models.ReaRev.rearev import
from train_model import Trainer_KBQA
from parsing import add_parse_args
import wandb


def _get_mode(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode", default="train", type=str, choices=["train", "train_policy"])
    ns, _ = p.parse_known_args(argv)
    return ns.mode


def _run_train_policy(argv):
    from train_subgraph_policy import build_policy_arg_parser, train_policy_from_args

    parser = build_policy_arg_parser()
    args = parser.parse_args(argv)
    train_policy_from_args(args)



def main():
    mode = _get_mode(None)
    if mode == "train_policy":
        _run_train_policy(None)
        return

    parser = argparse.ArgumentParser()
    add_parse_args(parser)
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    wandb.init(
        project="GNN-RAG-KBQA",
        name=args.experiment_name,
        config=vars(args),
        mode="online",  # set to "offline" if you want local-only logs
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.experiment_name == None:
        timestamp = str(int(time.time()))
        args.experiment_name = "{}-{}-{}".format(
            args.dataset,
            args.model_name,
            timestamp,
        )

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)
        wandb.alert(
            title="Evaluation Finished ✅",
            text=f"Evaluation of {args.load_experiment} is complete.",
        )
    wandb.finish()


if __name__ == '__main__':
    main()
