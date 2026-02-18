WANDB_API_KEY="66a8159cb49efe62675286b7216aecb7360a03a3" python main.py ReaRev \
  --name cwq --data_folder data/CWQ/ --lm relbert \
  --num_epoch 80 \
  --switch_epochs 1,2 \
  --data_file_train_switches train_2hop.json,train_3hop.json \
  --data_file_dev_switches dev_2hop.json,dev_3hop.json \
  --data_file_test_switches test_2hop.json,test_3hop.json
