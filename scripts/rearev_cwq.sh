
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

# python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 4 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice --data_folder data/CWQ/ --warmup_epoch 80  --data_eff --is_eval --load_experiment dim_50_twice-0.ckpt
python main.py ReaRev --entity_dim 50 --num_epoch 100  --batch_size 4 --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --data_folder data/SimpleQA/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.2 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_20 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.3 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_30 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.4 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_40 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.5 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_50 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.6 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_60 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.7 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_70 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.8 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_80 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff
# python main.py ReaRev --entity_dim 50 --num_epoch 100 --refinement_threshold 0.9 --batch_size 4 --load_ckpt_file dim_50-35.ckpt --load_experiment dim_50-35.ckpt --is_eval --eval_every 1 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_50_twice_load_ckpt_90 --data_folder data/CWQ/ --warmup_epoch 80  --data_eff

# python main.py ReaRev --entity_dim 150 --num_epoch 20 --batch_size 6 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_150 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 200 --num_epoch 20 --batch_size 4 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_200 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 250 --num_epoch 20 --batch_size 3 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3   --name cwq --experiment_name dim_250 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 300 --num_epoch 20 --batch_size 1 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_300 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 350 --num_epoch 20 --batch_size 1 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_350 --data_folder data/CWQ/ --warmup_epoch 80

# python main.py ReaRev --entity_dim 400 --num_epoch 20 --batch_size 1 --eval_every 2 --lm relbert --num_iter 2 --num_ins 3  --name cwq --experiment_name dim_400 --data_folder data/CWQ/ --warmup_epoch 80