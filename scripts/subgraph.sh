WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python main.py ReaRev \
  --name cwq --data_folder data/CWQ/ --lm relbert \
  --experiment_name cwq_khop \
  --switch_epoch 50 \
  --data_file_train_switch train_khop.json \
  --data_file_dev_switch dev_khop.json \
  --data_file_test_switch test_khop.json \
  --stream_data true \
  --beta_num_heads 5 \
  --beta_lambda 1.0 \
  --beta_dropout 0.1 \
  --debug_beta_topk 5 \
  --experiment_name cross-atten-change_data

