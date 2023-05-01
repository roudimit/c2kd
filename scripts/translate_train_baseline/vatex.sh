CUDA_VISIBLE_DEVICES=2 python train.py \
  --epochs 30 \
  --lr 1e-4 \
  --config configs/example/26_vatex_labse.yaml \
  --stage baseline-translate-train