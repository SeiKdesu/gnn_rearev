
import math
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn


from .base_gnn import BaseGNNLayer

VERY_NEG_NUMBER = -100000000000


class BetaCrossAttention(nn.Module):
    """
    Multi-head cross-attention from instruction -> nodes.
    Keeps per-head attention weights and fuses into node-wise beta logits.
    """
    def __init__(self, embed_dim, num_heads, instr_len, fusion_hidden=None, dropout=0.0):
        super(BetaCrossAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads for BetaCrossAttention: "
                f"embed_dim={embed_dim}, num_heads={num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.instr_len = instr_len
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        in_dim = num_heads * instr_len
        if fusion_hidden is None or fusion_hidden <= 0:
            fusion_hidden = in_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def _manual_attn_weights(self, query, key, key_padding_mask=None):
        """
        Manual per-head attention weights (B, H, K, N) using self.attn parameters.
        """
        # query: (B, K, D), key: (B, N, D)
        # This follows torch.nn.MultiheadAttention projections.
        if hasattr(self.attn, "_qkv_same_embed_dim") and self.attn._qkv_same_embed_dim:
            w = self.attn.in_proj_weight
            b = self.attn.in_proj_bias
            e = self.embed_dim
            q = F.linear(query, w[:e], b[:e])
            k = F.linear(key, w[e:2 * e], b[e:2 * e])
        else:
            q = F.linear(query, self.attn.q_proj_weight, self.attn.in_proj_bias[: self.embed_dim])
            k = F.linear(key, self.attn.k_proj_weight, self.attn.in_proj_bias[self.embed_dim : 2 * self.embed_dim])

        bsz, tgt_len, _ = q.size()
        src_len = k.size(1)
        head_dim = self.embed_dim // self.num_heads
        q = q.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)  # B, H, K, D
        k = k.view(bsz, src_len, self.num_heads, head_dim).transpose(1, 2)  # B, H, N, D
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], VERY_NEG_NUMBER)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
        return attn_weights

    def forward(self, node_emb, instr_emb, node_mask=None):
        """
        node_emb: (B, N, D)
        instr_emb: (B, K, D)
        node_mask: (B, N) bool/float (1 for valid)
        """
        if instr_emb.size(1) != self.instr_len:
            raise ValueError(
                f"instr_len mismatch: expected {self.instr_len}, got {instr_emb.size(1)}"
            )
        key_padding_mask = None
        if node_mask is not None:
            key_padding_mask = ~node_mask.bool()
        try:
            _, attn_w = self.attn(
                instr_emb,
                node_emb,
                node_emb,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        except TypeError:
            attn_w = self._manual_attn_weights(instr_emb, node_emb, key_padding_mask=key_padding_mask)

        if attn_w.dim() == 3:
            # Head-averaged weights; fallback to per-head computation.
            attn_w = self._manual_attn_weights(instr_emb, node_emb, key_padding_mask=key_padding_mask)

        # attn_w: (B, H, K, N)
        bsz, num_heads, tgt_len, src_len = attn_w.size()
        # (B, N, H*K)
        x = attn_w.permute(0, 3, 1, 2).contiguous().view(bsz, src_len, num_heads * tgt_len)
        beta_logits = self.fusion(x).squeeze(-1)
        if node_mask is not None:
            beta_logits = beta_logits.masked_fill(~node_mask.bool(), VERY_NEG_NUMBER)
        return beta_logits

class ReasonGNNLayer(BaseGNNLayer):
    """
    GNN Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg):
        super(ReasonGNNLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        
        self.use_posemb = args['pos_emb']
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        # beta cross-attention (optional)
        self.use_beta_crossattn = args.get('use_beta_crossattn', False)
        self.beta_lambda = args.get('beta_lambda', 1.0)
        self.beta_num_heads = args.get('beta_num_heads', 8)
        self.beta_dropout = args.get('beta_dropout', 0.0)
        self.beta_fusion_hidden = args.get('beta_fusion_hidden', None)
        self.debug_beta_topk = args.get('debug_beta_topk', 0)
        if self.use_beta_crossattn:
            self.beta_crossattn = BetaCrossAttention(
                embed_dim=entity_dim,
                num_heads=self.beta_num_heads,
                instr_len=self.num_ins,
                fusion_hidden=self.beta_fusion_hidden,
                dropout=self.beta_dropout,
            )
        else:
            self.beta_crossattn = None
        for i in range(self.num_gnn):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*(self.num_ins)*entity_dim + entity_dim, out_features=entity_dim))

            if self.use_posemb:
                self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
                self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))
        self.lin_m =  nn.Linear(in_features=(self.num_ins)*entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.local_entity = local_entity
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities
       

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
        """
        Aggregates neighbor representations
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb is not None:
            pe = pos_emb(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel)+pe) * fact_query)
        else :
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        fact_val = fact_val * fact_prior
        
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb_inv is not None:
            pe = pos_emb_inv(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel)+pe) * fact_query)
        else :
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
        

        fact_val = fact_val * fact_prior

        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def combine(self,emb):
        """
        Combines instruction-specific representations.
        """
        local_emb = torch.cat(emb, dim=-1)
        local_emb = F.relu(self.lin_m(local_emb))

        score_func = self.score_func
        
        score_tp = score_func(self.linear_drop(local_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        return current_dist, local_emb

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        """
        Compute next probabilistic vectors and current node representations.
        """
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        # score_func = getattr(self, 'score_func' + str(step))
        score_func = self.score_func
        neighbor_reps = []
        
        if self.use_posemb :
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else :
            pos_emb, pos_emb_inv = None, None

        for j in range(relational_ins.size(1)):
            # we do the same procedure for existing and inverse relations
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb)
            neighbor_reps.append(neighbor_rep)

            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:,j,:], rel_linear, pos_emb_inv)
            neighbor_reps.append(neighbor_rep)

        neighbor_reps = torch.cat(neighbor_reps, dim=2)
        
        
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        #print(next_local_entity_emb.size())
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.use_beta_crossattn and self.beta_crossattn is not None:
            beta_logits = self.beta_crossattn(self.local_entity_emb, relational_ins, self.local_entity_mask)
            score_tp = score_tp + self.beta_lambda * beta_logits
            if self.debug_beta_topk and self.debug_beta_topk > 0:
                self._log_beta_topk(beta_logits, step)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        
        
        return current_dist, self.local_entity_emb 

    def _log_beta_topk(self, beta_logits, step):
        logger = logging.getLogger(__name__)
        k = min(self.debug_beta_topk, beta_logits.size(1))
        if k <= 0:
            return
        with torch.no_grad():
            top_vals, top_idx = torch.topk(beta_logits, k=k, dim=1)
            # Log per batch element; keep concise.
            batch_size = beta_logits.size(0)
            for b in range(batch_size):
                node_ids = self.local_entity[b, top_idx[b]].tolist()
                vals = top_vals[b].tolist()
                pairs = ", ".join([f"{nid}:{val:.4f}" for nid, val in zip(node_ids, vals)])
                logger.info("beta_topk step=%s batch=%s -> %s", step, b, pairs)
