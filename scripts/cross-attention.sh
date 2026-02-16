WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2  --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name original_freebaseQA_dim_50 --data_folder data/CWQ/ --warmup_epoch 80 --use_beta_crossattn true \
  --beta_num_heads 5 \
  --beta_lambda 1.0 \
  --beta_dropout 0.1 
