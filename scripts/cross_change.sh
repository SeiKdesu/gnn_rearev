WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python main.py ReaRev \
  --name cwq --data_folder data/CWQ/ --lm relbert \
  --switch_epoch 20 \
  --data_file_train_switch train_switch.json \
  --data_file_dev_switch dev_switch.json \
  --data_file_test_switch test_switch.json \
  --batch_size_switch 2 \
  --test_batch_size_switch 10
