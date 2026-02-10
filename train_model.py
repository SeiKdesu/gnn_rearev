from utils import create_logger
import time
import numpy as np
import os, math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim

from tqdm import tqdm
import wandb
tqdm.monitor_iterval = 0


# from dataset_load_paths import load_data
from dataset_load import load_data
from dataset_load_graft import load_data_graft
from models.ReaRev.rearev import ReaRev

# from models.NSM.nsm import NSM
# from models.GraftNet.graftnet import GraftNet
from evaluate import Evaluator


class Trainer_KBQA(object):
    def __init__(self, args, model_name, logger=None):
        # print('Trainer here')
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.best_h1b = 0.0
        self.best_f1b = 0.0
        self.eps = args["eps"]
        self.warmup_epoch = args["warmup_epoch"]
        self.learning_rate = self.args["lr"]
        self.test_batch_size = args["test_batch_size"]
        self.distributed = bool(args.get("distributed", False))
        self.rank = int(args.get("rank", 0))
        self.world_size = int(args.get("world_size", 1))
        self.local_rank = int(args.get("local_rank", 0))
        self.is_main_process = self.rank == 0

        device_str = args.get("device", None)
        if device_str is not None:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if args["use_cuda"] else "cpu")

        if self.distributed and not dist.is_initialized():
            raise RuntimeError(
                "Distributed training requested but torch.distributed is not initialized. "
                "Launch with `torchrun --nproc_per_node=2 ...` or `python main.py ... --num_gpus 2`."
            )
        self.reset_time = 0
        self.load_data(args, args["lm"])

        if "decay_rate" in args:
            self.decay_rate = args["decay_rate"]
        else:
            self.decay_rate = 0.98

        if model_name == "ReaRev":
            self.model = ReaRev(
                self.args, len(self.entity2id), self.num_kb_relation, self.num_word
            )
        # elif model_name == "NSM":
        #     self.model = NSM(
        #         self.args, len(self.entity2id), self.num_kb_relation, self.num_word
        #     )
        # elif model_name == "GraftNet":
        #     self.model = GraftNet(
        #         self.args, len(self.entity2id), self.num_kb_relation, self.num_word
        #     )
        # elif model_name == "NuTrea":
        #     self.model = NuTrea(
        #         self.args, len(self.entity2id), self.num_kb_relation, self.num_word
        #     )

        if self.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")
        
        self.model.to(self.device)

        if self.distributed:
            if self.device.type == "cuda":
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                )
            else:
                self.model = DDP(self.model, find_unused_parameters=True)

        self.evaluator = Evaluator(
            args=args,
            model=self.model,
            entity2id=self.entity2id,
            relation2id=self.relation2id,
            device=self.device,
        )
        self.load_pretrain()
        self._refresh_relation_features()
        self.optim_def()

        self.num_relation = self.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.num_word = len(self.word2id)

        print(
            "Entity: {}, Relation: {}, Word: {}".format(
                self.num_entity, self.num_relation, self.num_word
            )
        )

        for k, v in args.items():
            if k.endswith("dim"):
                setattr(self, k, v)
            if k.endswith("emb_file") or k.endswith("kge_file"):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args["data_folder"] + v)

    def optim_def(self):

        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim_model = optim.Adam(trainable, lr=self.learning_rate)
        if self.decay_rate > 0:
            self.scheduler = ExponentialLR(self.optim_model, self.decay_rate)

    def load_data(self, args, tokenize):
        if args["model_name"] == "GraftNet":
            dataset = load_data_graft(args, tokenize)
        else:
            dataset = load_data(args, tokenize)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]
        self.num_word = dataset["num_word"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.rel_texts = dataset["rel_texts"]
        self.rel_texts_inv = dataset["rel_texts_inv"]

    def load_pretrain(self):
        args = self.args
        if args["load_experiment"] is not None:
            ckpt_path = os.path.join(args["checkpoint_dir"], args["load_experiment"])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def _refresh_relation_features(self):
        if not self.args.get("relation_word_emb", False):
            return
        was_training = self.model.training
        self.model.encode_rel_texts(self.rel_texts, self.rel_texts_inv)
        if was_training:
            self.model.train()

    def evaluate(self, data, test_batch_size=20, write_info=False):
        self._refresh_relation_features()
        return self.evaluator.evaluate(data, test_batch_size, write_info)

    def train(self, start_epoch, end_epoch):
        # self.load_pretrain()
        eval_every = self.args["eval_every"]
        # eval_acc = inference(self.model, self.valid_data, self.entity2id, self.args)
        # self.evaluate(self.test_data, self.test_batch_size)
        if self.is_main_process:
            print("Start Training------------------")
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, train_h1, train_f1 = self.train_epoch(epoch)

            if self.decay_rate > 0:
                self.scheduler.step()

            if self.is_main_process and self.logger is not None:
                self.logger.info(
                    "Epoch: {}, loss : {:.4f}, time: {}".format(
                        epoch + 1, loss, time.time() - st
                    )
                )
                self.logger.info(
                    "Training h1 : {:.4f}, f1 : {:.4f}".format(train_h1, train_f1)
                )

            if self.is_main_process and wandb.run is not None:
                wandb.log(
                    {
                        "Epoch": epoch + 1,
                        "Train Loss": float(loss),
                        "Train H1": float(train_h1),
                        "Train F1": float(train_f1),
                        "Learning Rate": float(self.optim_model.param_groups[0]["lr"]),
                    }
                )

            if self.is_main_process:
                self.save_ckpt(f"{epoch}")

            if (epoch + 1) % eval_every == 0:
                if self.distributed:
                    dist.barrier()
                if self.is_main_process:
                    eval_f1, eval_h1, eval_em = self.evaluate(
                        self.valid_data, self.test_batch_size
                    )

                    if self.logger is not None:
                        self.logger.info(
                            "EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(
                                eval_f1, eval_h1, eval_em
                            )
                        )
               
                # eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size)
                # self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                do_test = False

                if self.is_main_process:
                    if epoch > self.warmup_epoch:
                        if eval_h1 > self.best_h1:
                            self.best_h1 = eval_h1
                            self.save_ckpt("h1")
                            if self.logger is not None:
                                self.logger.info("BEST EVAL H1: {:.4f}".format(eval_h1))
                            do_test = True
                        if eval_f1 > self.best_f1:
                            self.best_f1 = eval_f1
                            self.save_ckpt("f1")
                            if self.logger is not None:
                                self.logger.info("BEST EVAL F1: {:.4f}".format(eval_f1))
                            do_test = True

                    eval_f1, eval_h1, eval_em = self.evaluate(
                        self.test_data, self.test_batch_size
                    )
                    if self.logger is not None:
                        self.logger.info(
                            "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(
                                eval_f1, eval_h1, eval_em
                            )
                        )
                    if wandb.run is not None:
                        wandb.log(
                            {
                                "Epoch": epoch + 1,
                                "Val F1": float(eval_f1),
                                "Val H1": float(eval_h1),
                                "Val EM": float(eval_em),
                            }
                        )
                if self.distributed:
                    dist.barrier()
                # if do_test:
                #     eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size)
                #     self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

                # if eval_h1 > self.best_h1:
                #     self.best_h1 = eval_h1
                #     self.save_ckpt("h1")
                # if eval_f1 > self.best_f1:
                #     self.best_f1 = eval_f1
                #     self.save_ckpt("f1")
                # self.reset_time = 0
                # else:
                #     self.logger.info('No improvement after one evaluation iter.')
                #     self.reset_time += 1
                # if self.reset_time >= 5:
                #     self.logger.info('No improvement after 5 evaluation. Early Stopping.')
                #     break
        if self.is_main_process:
            self.save_ckpt("final")
            if self.logger is not None:
                self.logger.info("Train Done! Evaluate on testset with saved model")
            print("End Training------------------")
            self.evaluate_best()
        if self.distributed:
            dist.barrier()

    def evaluate_best(self):
        for reason in ["h1", "f1", "final"]:
            filename = os.path.join(
                self.args["checkpoint_dir"],
                "{}-{}.ckpt".format(self.args["experiment_name"], reason),
            )
            if not os.path.exists(filename):
                print(f"⚠️ Checkpoint not found: {filename}, skipping...")
                continue

            self.load_ckpt(filename)
            eval_f1, eval_h1, eval_em = self.evaluate(
                self.test_data, self.test_batch_size, write_info=False
            )
            self.logger.info(f"Best {reason} evaluation")
            self.logger.info(
                "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em)
            )
        # filename = os.path.join(
        #     self.args["checkpoint_dir"],
        #     "{}-h1.ckpt".format(self.args["experiment_name"]),
        # )
        # self.load_ckpt(filename)
        # eval_f1, eval_h1, eval_em = self.evaluate(
        #     self.test_data, self.test_batch_size, write_info=False
        # )
        # self.logger.info("Best h1 evaluation")
        # self.logger.info(
        #     "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em)
        # )

        # filename = os.path.join(
        #     self.args["checkpoint_dir"],
        #     "{}-f1.ckpt".format(self.args["experiment_name"]),
        # )
        # self.load_ckpt(filename)
        # eval_f1, eval_h1, eval_em = self.evaluate(
        #     self.test_data, self.test_batch_size, write_info=False
        # )
        # self.logger.info("Best f1 evaluation")
        # self.logger.info(
        #     "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em)
        # )

        # filename = os.path.join(
        #     self.args["checkpoint_dir"],
        #     "{}-final.ckpt".format(self.args["experiment_name"]),
        # )
        # self.load_ckpt(filename)
        # eval_f1, eval_h1, eval_em = self.evaluate(
        #     self.test_data, self.test_batch_size, write_info=False
        # )
        # self.logger.info("Final evaluation")
        # self.logger.info(
        #     "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em)
        # )

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        eval_f1, eval_hits, eval_ems = self.evaluate(
            self.valid_data, self.test_batch_size, write_info=False
        )
        self.logger.info(
            "EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(
                eval_f1, eval_hits, eval_ems
            )
        )
        test_f1, test_hits, test_ems = self.evaluate(
            self.test_data, self.test_batch_size, write_info=True
        )
        self.logger.info(
            "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(
                test_f1, test_hits, test_ems
            )
        )

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _set_train_batches(self, epoch: int) -> int:
        """
        Deterministic shuffling + (optional) DDP sharding.
        Returns number of samples assigned to this rank.
        """
        num_data = int(self.train_data.num_data)
        rng = np.random.RandomState(int(self.args.get("seed", 0)) + int(epoch))
        indices = np.arange(num_data, dtype=np.int64)
        rng.shuffle(indices)

        if not self.distributed or self.world_size <= 1:
            self.train_data.batches = indices
            return int(len(indices))

        world_size = int(self.world_size)
        total_size = int(math.ceil(num_data / world_size) * world_size)
        if total_size > num_data:
            pad = indices[: (total_size - num_data)]
            indices = np.concatenate([indices, pad], axis=0)

        rank_indices = indices[int(self.rank) : total_size : world_size]
        self.train_data.batches = rank_indices
        return int(len(rank_indices))

    def train_epoch(self, epoch: int):
        self.model.train()
        num_samples = self._set_train_batches(epoch)

        loss_sum = 0.0
        sample_count = 0.0
        h1_sum = 0.0
        f1_sum = 0.0
        metric_count = 0.0

        num_iter = int(math.ceil(num_samples / self.args["batch_size"])) if num_samples > 0 else 0

        for iteration in tqdm(range(num_iter), disable=not self.is_main_process):
            batch = self.train_data.get_batch(
                iteration, self.args["batch_size"], self.args["fact_drop"]
            )

            self.optim_model.zero_grad()
            loss, _, _, tp_list = self.model(batch, training=True)
            h1_list, f1_list = tp_list

            batch_sz = int(batch[0].shape[0])
            loss_sum += float(loss.item()) * batch_sz
            sample_count += float(batch_sz)
            h1_sum += float(np.sum(h1_list))
            f1_sum += float(np.sum(f1_list))
            metric_count += float(len(h1_list))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for name, param in self.model.named_parameters()],
                self.args["gradient_clip"],
            )
            self.optim_model.step()

        if self.distributed:
            stats = torch.tensor(
                [loss_sum, sample_count, h1_sum, metric_count, f1_sum, metric_count],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            loss_sum, sample_count, h1_sum, metric_count, f1_sum, _ = stats.tolist()

        loss_mean = (loss_sum / sample_count) if sample_count > 0 else 0.0
        h1_mean = (h1_sum / metric_count) if metric_count > 0 else 0.0
        f1_mean = (f1_sum / metric_count) if metric_count > 0 else 0.0

        extras = [0, 0]
        return loss_mean, extras, h1_mean, f1_mean

    def save_ckpt(self, reason="h1"):
        model = self._unwrap_model()
        checkpoint = {"model_state_dict": model.state_dict()}
        model_name = os.path.join(
            self.args["checkpoint_dir"],
            "{}-{}.ckpt".format(self.args["experiment_name"], reason),
        )
        torch.save(checkpoint, model_name)
        print("Best %s, save model as %s" % (reason, model_name))

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename, map_location="cpu")
        sd = checkpoint["model_state_dict"]

        model = self._unwrap_model()
        model_sd = model.state_dict()

        # DDP/DataParallel checkpoints may prefix keys with "module."
        if any(k.startswith("module.") for k in sd.keys()) and not any(
            k.startswith("module.") for k in model_sd.keys()
        ):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        elif (not any(k.startswith("module.") for k in sd.keys())) and any(
            k.startswith("module.") for k in model_sd.keys()
        ):
            sd = {"module." + k: v for k, v in sd.items()}

        # 1) モデルに存在するキーだけ残す
        filtered = {k: v for k, v in sd.items() if k in model_sd}

        # 2) shape不一致も除外（あると strict=True で落ちる）
        shape_mismatch = []
        for k in list(filtered.keys()):
            if filtered[k].shape != model_sd[k].shape:
                shape_mismatch.append((k, tuple(filtered[k].shape), tuple(model_sd[k].shape)))
                filtered.pop(k)

        # 3) strict=True を通すために「モデルが要求する全キーが揃ってるか」確認
        missing = [k for k in model_sd.keys() if k not in filtered]
        unexpected = [k for k in sd.keys() if k not in model_sd]

        if unexpected:
            print("[INFO] Dropped unexpected keys (not in current model):", unexpected[:20], "..." if len(unexpected) > 20 else "")
        if shape_mismatch:
            print("[WARN] Dropped shape-mismatch keys:", shape_mismatch[:10], "..." if len(shape_mismatch) > 10 else "")

        if missing:
            # これが出るなら「今のモデルに必要な重みが ckpt に無い」ので完全復元は不可
            raise RuntimeError(f"Missing keys for strict=True (ckpt lacks these): {missing[:30]}{'...' if len(missing)>30 else ''}")

        model.load_state_dict(filtered, strict=True)
        with torch.no_grad():
            max_abs_diff = 0.0
            max_key = None
            for k, v in model.state_dict().items():
                diff = (v.cpu() - filtered[k]).abs().max().item()
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    max_key = k
        print(f"[VERIFY] max_abs_diff={max_abs_diff:.3e} at {max_key}")
        assert max_abs_diff == 0.0, "Loaded weights do not exactly match checkpoint!"
        model.to(self.device)
        self._refresh_relation_features()
        print("Loaded checkpoint with strict=True (after filtering).")
