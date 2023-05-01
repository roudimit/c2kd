CUDA_VISIBLE_DEVICES=2 python train.py \
  --epochs 30 \
  --lr 1e-4 \
  --config configs/example/26_vatex_teachers.yaml \
  --teacher_sim_pool max \
  --balance 0.1 \
  --distill_temp 0.1 \
  --teachers arch_teacher_xlm,arch_teacher_mbert,arch_teacher_distill \
  --stage c2kd