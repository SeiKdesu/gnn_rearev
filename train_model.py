from utils import create_logger
import time
import numpy as np
import os, math

import torch
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
        self.device = torch.device("cuda" if args["use_cuda"] else "cpu")
        self.reset_time = 0
        self._data_switched = False
        self._data_switch_epoch = args.get("switch_epoch", None)
        self._data_switch_plan = self._build_data_switch_plan(args)
        self._data_switch_idx = 0
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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        
        self.model.to(self.device)
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

    def _log_info(self, msg: str) -> None:
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def _parse_csv_list(self, value, cast_func=None):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items = list(value)
        else:
            s = str(value).strip()
            if not s:
                return []
            items = [v.strip() for v in s.split(",") if v.strip()]
        if cast_func is None:
            return items
        out = []
        for v in items:
            try:
                out.append(cast_func(v))
            except Exception:
                self._log_info(f"[data switch] failed to parse value={v!r}; skipping.")
        return out

    def _build_data_switch_plan(self, args):
        plan = []

        epochs = self._parse_csv_list(args.get("switch_epochs"), int)
        train_files = self._parse_csv_list(args.get("data_file_train_switches"))
        dev_files = self._parse_csv_list(args.get("data_file_dev_switches"))
        test_files = self._parse_csv_list(args.get("data_file_test_switches"))
        bs_list = self._parse_csv_list(args.get("batch_size_switches"), int)
        tbs_list = self._parse_csv_list(args.get("test_batch_size_switches"), int)

        if epochs:
            last_ep = -1
            for i, ep in enumerate(epochs):
                if ep < last_ep:
                    self._log_info("[data switch] switch_epochs not sorted; using given order.")
                last_ep = ep
                step = {
                    "epoch": ep,
                    "train": train_files[i] if i < len(train_files) else None,
                    "dev": dev_files[i] if i < len(dev_files) else None,
                    "test": test_files[i] if i < len(test_files) else None,
                    "batch_size": bs_list[i] if i < len(bs_list) else None,
                    "test_batch_size": tbs_list[i] if i < len(tbs_list) else None,
                }
                if any(v is not None for k, v in step.items() if k != "epoch"):
                    plan.append(step)
            return plan

        # Backward compatibility: single switch_epoch
        switch_epoch = args.get("switch_epoch", None)
        if switch_epoch is not None:
            step = {
                "epoch": switch_epoch,
                "train": args.get("data_file_train_switch"),
                "dev": args.get("data_file_dev_switch"),
                "test": args.get("data_file_test_switch"),
                "batch_size": args.get("batch_size_switch"),
                "test_batch_size": args.get("test_batch_size_switch"),
            }
            if any(v is not None for k, v in step.items() if k != "epoch"):
                plan.append(step)
        return plan

    def _switch_dataset_if_needed(self, epoch: int) -> None:
        if not self._data_switch_plan:
            return
        while self._data_switch_idx < len(self._data_switch_plan):
            step = self._data_switch_plan[self._data_switch_idx]
            switch_epoch = step.get("epoch", None)
            if switch_epoch is None:
                self._data_switch_idx += 1
                continue
            try:
                switch_epoch = int(switch_epoch)
            except Exception:
                self._log_info(f"[data switch] invalid switch_epoch={switch_epoch!r}; skipping.")
                self._data_switch_idx += 1
                continue
            if switch_epoch < 0:
                self._data_switch_idx += 1
                continue
            if epoch < switch_epoch:
                return

            provided = []
            for split in ("train", "dev", "test"):
                val = step.get(split)
                if val is not None:
                    self.args[f"data_file_{split}"] = val
                    provided.append((split, val))

            bs_switch = step.get("batch_size")
            tbs_switch = step.get("test_batch_size")
            if not provided and bs_switch is None and tbs_switch is None:
                self._log_info(f"[data switch] epoch {epoch + 1}: no changes provided; skipping.")
                self._data_switch_idx += 1
                continue

            prev_counts = (self.num_entity, self.num_kb_relation, self.num_word)
            old_train, old_valid, old_test = self.train_data, self.valid_data, self.test_data
            self.load_data(self.args, self.args["lm"])
            del old_train, old_valid, old_test

            new_counts = (self.num_entity, self.num_kb_relation, self.num_word)
            if new_counts[:2] != prev_counts[:2]:
                raise RuntimeError(
                    "Switching datasets changed (num_entity, num_relation). "
                    f"before={prev_counts[:2]}, after={new_counts[:2]}. "
                    "This is not supported without rebuilding the model."
                )
            lm_name = self.args.get("lm", "lstm")
            if lm_name == "lstm" and new_counts[2] != prev_counts[2]:
                raise RuntimeError(
                    "Switching datasets changed num_word under LSTM. "
                    f"before={prev_counts[2]}, after={new_counts[2]}. "
                    "This is not supported without rebuilding the model."
                )
            if lm_name != "lstm" and new_counts[2] != prev_counts[2]:
                self._log_info(
                    f"[data switch] num_word changed {prev_counts[2]} -> {new_counts[2]} (lm={lm_name}); "
                    "ignored for non-LSTM."
                )

            self.evaluator = Evaluator(
                args=self.args,
                model=self.model,
                entity2id=self.entity2id,
                relation2id=self.relation2id,
                device=self.device,
            )
            self._refresh_relation_features()

            if bs_switch is not None:
                try:
                    bs_switch = int(bs_switch)
                    if bs_switch > 0 and bs_switch != self.args["batch_size"]:
                        old_bs = self.args["batch_size"]
                        self.args["batch_size"] = bs_switch
                        self._log_info(f"[data switch] batch_size changed {old_bs} -> {bs_switch}")
                except Exception:
                    self._log_info(f"[data switch] invalid batch_size_switch={bs_switch!r}; skipping.")
            if tbs_switch is not None:
                try:
                    tbs_switch = int(tbs_switch)
                    if tbs_switch > 0 and tbs_switch != self.test_batch_size:
                        old_tbs = self.test_batch_size
                        self.test_batch_size = tbs_switch
                        self.args["test_batch_size"] = tbs_switch
                        self._log_info(f"[data switch] test_batch_size changed {old_tbs} -> {tbs_switch}")
                except Exception:
                    self._log_info(f"[data switch] invalid test_batch_size_switch={tbs_switch!r}; skipping.")

            provided_str = ", ".join([f"{s}={v}" for s, v in provided]) if provided else "no file changes"
            self._log_info(
                f"[data switch] epoch {epoch + 1}: switched data files -> {provided_str}"
            )

            self._data_switch_idx += 1

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
        print("Start Training------------------")
        for epoch in range(start_epoch, end_epoch + 1):
            self._switch_dataset_if_needed(epoch)
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()

            if self.decay_rate > 0:
                self.scheduler.step()

            self.logger.info(
                "Epoch: {}, loss : {:.4f}, time: {}".format(
                    epoch + 1, loss, time.time() - st
                )
            )
            self.logger.info(
                "Training h1 : {:.4f}, f1 : {:.4f}".format(
                    np.mean(h1_list_all), np.mean(f1_list_all)
                )
            )
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": loss,
                "Train H1": np.mean(h1_list_all),
                "Train F1": np.mean(f1_list_all),
                "Learning Rate": self.optim_model.param_groups[0]['lr']
            })

            self.save_ckpt(f"{epoch}")
            if (epoch + 1) % eval_every == 0:
                eval_f1, eval_h1, eval_em = self.evaluate(
                    self.valid_data, self.test_batch_size
                )
                
                self.logger.info(
                    "EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(
                        eval_f1, eval_h1, eval_em
                    )
                )
               
                # eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size)
                # self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                do_test = False

                if epoch > self.warmup_epoch:
                    if eval_h1 > self.best_h1:
                        self.best_h1 = eval_h1
                        self.save_ckpt("h1")
                        self.logger.info("BEST EVAL H1: {:.4f}".format(eval_h1))
                        do_test = True
                    if eval_f1 > self.best_f1:
                        self.best_f1 = eval_f1
                        self.save_ckpt("f1")
                        self.logger.info("BEST EVAL F1: {:.4f}".format(eval_f1))
                        do_test = True

                eval_f1, eval_h1, eval_em = self.evaluate(
                    self.test_data, self.test_batch_size
                )
                self.logger.info(
                    "TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(
                        eval_f1, eval_h1, eval_em
                    )
                )
                wandb.log({
                    "Epoch": epoch + 1,
                    "Val F1": eval_f1,
                    "Val H1": eval_h1,
                    "Val EM": eval_em
                })
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
        self.save_ckpt("final")
        self.logger.info("Train Done! Evaluate on testset with saved model")
        print("End Training------------------")
        self.evaluate_best()

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

    def train_epoch(self):
        self.model.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        actor_losses = []
        ent_losses = []
        num_epoch = math.ceil(self.train_data.num_data / self.args["batch_size"])
        h1_list_all = []
        f1_list_all = []
        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(
                iteration, self.args["batch_size"], self.args["fact_drop"]
            )

            self.optim_model.zero_grad()
            loss, _, _, tp_list = self.model(batch, training=True)
            # if tp_list is not None:
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for name, param in self.model.named_parameters()],
                self.args["gradient_clip"],
            )
            self.optim_model.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        model = self.model
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

        model = self.model
        model_sd = model.state_dict()

        # (任意) DDP/DataParallelの "module." が付いてる場合に剥がす
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

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
