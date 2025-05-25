CUDA_VISIBLE_DEVICES=0 python eval.py \
    --checkpt_path ./logs/pretrain/latest.pt \
    --config small \
    --timesteps 1024 \
    --temperature 0.8 \
    --bsz 8 \
    --device cuda \
    --num_samples 32 \
    --eval_entropy \
    --eval_perplexity \
    # --torch_compile \
