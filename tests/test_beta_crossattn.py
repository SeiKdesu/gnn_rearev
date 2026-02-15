import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from modules.kg_reasoning.reasongnn import BetaCrossAttention, VERY_NEG_NUMBER


def main():
    torch.manual_seed(0)
    bsz, num_nodes, dim = 2, 5, 16
    num_heads = 4
    instr_len = 3
    node_emb = torch.randn(bsz, num_nodes, dim)
    instr_emb = torch.randn(bsz, instr_len, dim)
    node_mask = torch.ones(bsz, num_nodes, dtype=torch.bool)
    node_mask[:, -1] = False

    beta = BetaCrossAttention(
        embed_dim=dim,
        num_heads=num_heads,
        instr_len=instr_len,
        fusion_hidden=None,
        dropout=0.0,
    )
    beta_logits = beta(node_emb, instr_emb, node_mask)

    assert beta_logits.shape == (bsz, num_nodes)
    masked_vals = beta_logits[:, -1]
    assert torch.all(masked_vals <= VERY_NEG_NUMBER / 10), "Masked nodes should be very negative."
    print("BetaCrossAttention sanity check passed.")


if __name__ == "__main__":
    main()
