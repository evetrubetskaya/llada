accelerate launch train.py \
    --logdir ./logs \
    --name pretrain-large \
    --config large \
    --mode pretrain \
    --bsz 4 \
    --lr 2e-4 \
    --n_epochs 100 \
    --torch_compile \
    --dtype bf16 \
    # --continue_training \
    # --reset_opt
