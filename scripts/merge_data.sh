python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/baseline-verl-grpo/test-2k-no-entctl--DeepSeek-R1-Distill-Qwen-1.5B-deepscaler-NODE2/global_step_350/actor \
    --target_dir checkpoints/baseline-verl-grpo/test-2k-no-entctl--DeepSeek-R1-Distill-Qwen-1.5B-deepscaler-NODE2/global_step_350