set -x



# export VERL_LOGGING_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0


data_path=/nfsdata/yhe/verl/data/qwen-math/deepscaler/train.parquet
save_path=/nfsdata/yhe/verl/analysis_data/train_all_generation_math_instruct.parquet
model_path=/nfsdata/yhe/models/Qwen2.5-Math-1.5B-Instruct


python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=16 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.name=vllm \
    rollout.temperature=1.0 \
    rollout.calculate_log_probs=True \
    rollout.prompt_length=1536 \
    rollout.response_length=2560 \
    +rollout.logprobs=10 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8