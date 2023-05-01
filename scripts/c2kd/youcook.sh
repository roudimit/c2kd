CUDA_VISIBLE_DEVICES=3 python train.py \
  --epochs 10 \
  --lr 1e-4 \
  --config configs/example/25_yc_s3d_teachers.yaml \
  --teacher_sim_pool min \
  --balance 0.1 \
  --distill_temp 0.1 \
  --teachers arch_teacher_xlm,arch_teacher_mbert,arch_teacher_distill \
  --stage c2kd