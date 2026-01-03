WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python denoise_sanity.py \
  --data_folder data/CWQ_freebase/ --split train --sample_idx 0 --enable_denoise true \
  --min_edges 0 \
  --topK_strict 10 --topM_strict 5 --gamma_strict 0.9 \
  --topK_loose  10 --topM_loose  8 --gamma_loose  0.5
