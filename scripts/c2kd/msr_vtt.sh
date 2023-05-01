CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 20 \
  --lr 1e-4 \
  --config configs/example/23_msrvtt_teachers.yaml \
  --teacher_sim_pool min \
  --balance 0.5 \
  --distill_temp 0.1 \
  --teachers arch_teacher_xlm,arch_teacher_mbert,arch_teacher_distill \
  --stage c2kd