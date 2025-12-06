import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSubgraphSampler(nn.Module):
    def __init__(self, entity_dim, rel_dim, temperature=1.0):
        super(DifferentiableSubgraphSampler, self).__init__()
        # 論文 Eq(2): Fact Representation Fi = [hi || ri || ti]
        # 入力次元 = Head(entity_dim) + Relation(rel_dim) + Tail(entity_dim)
        self.fact_dim = entity_dim * 2 + rel_dim

        # 論文 Eq(3) & Eq(7): 選択確率の計算用レイヤー
        # p(tau_i) = sigma(W * Fi + b)
        self.fact_scorer = nn.Linear(self.fact_dim, 1)
        self.temperature = temperature

    def get_fact_representations(self, head_emb, rel_emb, tail_emb):
        """
        論文 Eq(2): ファクト表現 Fi を構築
        """
        # head, relation, tail の埋め込みを結合 (Concatenate)
        fact_reps = torch.cat([head_emb, rel_emb, tail_emb], dim=-1)
        return fact_reps

    def forward(self, head_emb, rel_emb, tail_emb, is_eval=False):
        """
        Forward Pass: ファクト表現構築 -> スコアリング -> サンプリング
        """
        # 1. ファクト表現の作成 (Eq 2)
        fact_reps = self.get_fact_representations(head_emb, rel_emb, tail_emb)

        # 2. 選択確率の計算 (p_i)
        # shape: (num_facts, 1)
        logits = self.fact_scorer(fact_reps)
        probs = torch.sigmoid(logits)

        # 推論時(Inference): 単純に確率を返す (論文 Section 4.4参照: Top-kや閾値で選定するため)
        if is_eval:
            return probs, None

        # 3. Gumbel-Softmaxによる微分可能なサンプリング (Eq 7, 8)
        # 「選ぶ(Select)」か「選ばない(Reject)」の2クラス分類として扱います。

        eps = 1e-10
        # log(p) と log(1-p) を用意
        log_p = torch.log(probs + eps)           # Selectのロジット
        log_not_p = torch.log(1 - probs + eps)   # Rejectのロジット

        # (Num_facts, 2) のロジットを作成 [Select, Reject]
        gumbel_logits = torch.cat([log_p, log_not_p], dim=-1)

        # Gumbel-Softmaxを実行
        # hard=True: Forwardではone-hot (0 or 1)、Backwardでは勾配が流れる
        z_soft = F.gumbel_softmax(gumbel_logits, tau=self.temperature, hard=True, dim=1)

        # z_soft[:, 0] が "Select (1)" に対応するマスク
        selection_mask = z_soft[:, 0].unsqueeze(-1) # (Num_facts, 1)

        return probs, selection_mask