
###ReaRev+SBERT training
# python main.py ReaRev --is_eval --load_experiment relbert-full_cwq-rearev-final.ckpt --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \
# --experiment_name prn_cwq-rearev-sbert --data_folder data/CWQ/ --num_epoch 100 --warmup_epoch 80

###ReaRev+LMSR training
# python main.py ReaRev  --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \
# --experiment_name prn_cwq-rearev-lmsr  --data_folder data/CWQ/ --num_epoch 100 #--warmup_epoch 80



###Evaluate CWQ
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --data_folder data/CWQ/ --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True --load_experiment ReaRev_CWQ.ckpt --is_eval --name cwq


# python main.py ReaRev --entity_dim 100 --num_epoch 20 --batch_size 4 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_100 --data_folder data/CWQ/ --warmup_epoch 80

WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2  --lm sbert --num_iter 2 --num_ins 3   --name cwq --experiment_name cross_attn_cwq --data_folder data/CWQ/ --warmup_epoch 80 

# python main.py ReaRev --entity_dim 150 --num_epoch 20 --batch_size 6 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_150 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 200 --num_epoch 20 --batch_size 4 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_200 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 250 --num_epoch 20 --batch_size 3 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_250 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 300 --num_epoch 20 --batch_size 1 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_300 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 350 --num_epoch 20 --batch_size 1 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_350 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 400 --num_epoch 20 --batch_size 1 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_400 --data_folder data/CWQ/ --warmup_epoch 80