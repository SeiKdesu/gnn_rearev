WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python main.py ReaRev \
  --name cwq --data_folder data/CWQ/ --lm relbert \
  --experiment_name cwq_khop \
  --switch_epoch 20 \
  --data_file_train_switch train-subgraph.json \
  --data_file_dev_switch dev-subgraph.json \
  --data_file_test_switch test-subgraph.json \
  --stream_data true
