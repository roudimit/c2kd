CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 30 \
  --lr 1e-4 \
  --config configs/example/26_vatex_distill.yaml \
  --stage baseline-translate-train
  
  
