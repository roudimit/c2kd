CUDA_VISIBLE_DEVICES=3 python train.py \
  --epochs 10 \
  --lr 1e-4 \
  --config configs/example/19_youcook_multilingual_s3d.yaml \
  --stage baseline-translate-train