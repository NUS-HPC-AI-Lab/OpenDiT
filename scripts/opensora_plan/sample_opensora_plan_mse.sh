CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_mse.py --config configs/opensora_plan/sample_65f_skip.yaml > log_sample_opensora_plan_mse.sh.txt 2>&1
