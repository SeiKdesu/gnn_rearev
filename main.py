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



parser = argparse.ArgumentParser()
add_parse_args(parser)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
wandb.init(
    project='GNN-RAG-KBQA',
    name=args.experiment_name,
    config=vars(args),
    mode="online",  # "offline" にするとネット接続不要でローカルログのみ
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


def main():
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
