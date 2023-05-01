CUDA_VISIBLE_DEVICES=1 python train.py \
  --epochs 20 \
  --lr 5e-5 \
  --config configs/example/27_rudder_teachers.yaml \
  --teacher_sim_pool mean \
  --balance 0.1 \
  --distill_temp 0.1 \
  --teachers arch_teacher_xlm,arch_teacher_mbert,arch_teacher_distill \
  --stage c2kd