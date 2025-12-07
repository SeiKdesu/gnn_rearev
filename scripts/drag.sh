# 事前学習
python main.py ReaRev \
  --data_folder data/webqsp/ \
  --experiment_name pretrain-rearev-2 \
  --num_epoch 100 \
  --batch_size 8 \
  --lr 5e-5 \


# ファインチューニング
# python main.py DReaRev \
#   --data_folder data/webqsp/ \
#   --experiment_name drag-joint \
#   --num_epoch 18 \
#   --batch_size 4 \
#   --lr 5e-5 \
#   --drag_temperature 0.5 \
#   --lambda_sel 1.0 \
#   --selector_sparsity_weight 0.1 \
#   --selector_sparsity_target 0.2 \
#   --selector_entropy_weight 0.01 \
#   --load_experiment pretrain-rearev-final.ckpt
