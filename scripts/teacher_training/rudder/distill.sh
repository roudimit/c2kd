CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 20 \
  --lr 1e-4 \
  --config configs/example/27_rudder_distill.yaml \
  --stage baseline-translate-train
  
  
