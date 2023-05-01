CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 20 \
  --lr 1e-4 \
  --config configs/example/15_msrvtt_clip_multilingual.yaml \
  --stage baseline-translate-train