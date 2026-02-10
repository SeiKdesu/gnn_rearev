import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from typing import Optional, Tuple

from parsing import add_parse_args
from train_model import Trainer_KBQA
from utils import create_logger


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.environ.get(name, None)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _is_torchrun() -> bool:
    world_size = _env_int("WORLD_SIZE", 1) or 1
    local_rank = os.environ.get("LOCAL_RANK", None)
    return world_size > 1 and local_rank is not None


def _setup_distributed_env(args: argparse.Namespace) -> Tuple[bool, int, int, int]:
    world_size = _env_int("WORLD_SIZE", 1) or 1
    if world_size <= 1:
        return False, 0, 1, 0

    rank = _env_int("RANK", 0) or 0
    local_rank = _env_int("LOCAL_RANK", 0) or 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return True, rank, world_size, local_rank


def _setup_distributed_spawn(args: argparse.Namespace, local_rank: int) -> Tuple[bool, int, int, int]:
    world_size = int(args.num_gpus)
    rank = int(local_rank)

    os.environ.setdefault("MASTER_ADDR", str(args.master_addr))
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return True, rank, world_size, local_rank


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_experiment_name(args: argparse.Namespace) -> None:
    if getattr(args, "experiment_name", None):
        return
    timestamp = str(int(time.time()))
    # Keep legacy-ish format, but avoid referencing non-existent args.dataset.
    args.experiment_name = f"{args.name}-{args.model_name}-{timestamp}"


def _run(args: argparse.Namespace, *, rank: int, world_size: int, local_rank: int, distributed: bool) -> None:
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.distributed = distributed
    args.use_cuda = torch.cuda.is_available()
    args.device = f"cuda:{local_rank}" if args.use_cuda else "cpu"

    # Different seed per rank for dropout etc. (data sharding is handled deterministically in Trainer)
    _seed_everything(int(args.seed) + int(rank))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _ensure_experiment_name(args)

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    logger = create_logger(args) if rank == 0 else None

    if rank == 0:
        wandb.init(
            project="GNN-RAG-KBQA",
            name=args.experiment_name,
            config=vars(args),
            mode="online",
        )

    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)

    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        if rank == 0:
            assert args.load_experiment is not None
            ckpt_path = (
                os.path.join(args.checkpoint_dir, args.load_experiment)
                if args.load_experiment is not None
                else None
            )
            trainer.evaluate_single(ckpt_path)
            wandb.alert(
                title="Evaluation Finished ✅",
                text=f"Evaluation of {args.load_experiment} is complete.",
            )

    if distributed:
        dist.barrier()

    if rank == 0:
        wandb.finish()


def _worker(local_rank: int, args: argparse.Namespace) -> None:
    # torchrun case: ranks are provided via env
    if _is_torchrun():
        distributed, rank, world_size, local_rank = _setup_distributed_env(args)
        _run(args, rank=rank, world_size=world_size, local_rank=local_rank, distributed=distributed)
        if distributed:
            dist.destroy_process_group()
        return

    # Spawn-based multi-GPU (single node)
    if int(args.num_gpus) > 1:
        distributed, rank, world_size, local_rank = _setup_distributed_spawn(args, local_rank)
        _run(args, rank=rank, world_size=world_size, local_rank=local_rank, distributed=distributed)
        dist.destroy_process_group()
        return

    # Single process
    _run(args, rank=0, world_size=1, local_rank=0, distributed=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_parse_args(parser)
    args = parser.parse_args()

    # If launched via torchrun, do not spawn again.
    if _is_torchrun():
        _worker(_env_int("LOCAL_RANK", 0) or 0, args)
        return

    if int(args.num_gpus) > 1:
        mp.spawn(_worker, nprocs=int(args.num_gpus), args=(args,))
        return

    _worker(0, args)


if __name__ == "__main__":
    main()
