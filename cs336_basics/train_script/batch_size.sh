#!/bin/bash

# Hyperparameters from assignment
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
ROPE_THETA=10000

MAX_ITERS=20000

# Optimizer hyperparameters
WEIGHT_DECAY=0.3
BETA1=0.9
BETA2=0.999
WARMUP_ITERS=2000
MAX_GRAD_NORM=1.0

# Data paths
TRAIN_DATA="data/TinyStoriesV2-GPT4-train_tokens.npy"
VAL_DATA="data/TinyStoriesV2-GPT4-valid_tokens.npy"

# Logging and checkpointing
SAVE_INTERVAL=20000
LOG_INTERVAL=10
EVAL_INTERVAL=500
EVAL_ITERS=100

# WandB logging
USE_WANDB=true
WANDB_PROJECT="transformer-lm"

# Device
DEVICE="cuda:0"

# Batch sizes to try
BATCH_SIZES=(1 4 16 64 128 256 512)

# Base learning rate at batch_size=64
LR_BASE=3e-3
BS_BASE=64


# Loop through batch sizes
for BS in "${BATCH_SIZES[@]}"; do
    # Linear scaling rule: lr ∝ batch_size，如果batch_size扩大k倍，那么最佳学习率也扩大k倍
    LR=$(python3 -c "print(f\"{${LR_BASE} * $BS / $BS_BASE:.2e}\")")
    
    echo "=========================================="
    echo "Batch Size: $BS, Learning Rate: $LR"
    echo "=========================================="
    
    CHECKPOINT_DIR="checkpoints/tinystories_17m_bs_${BS}_lr_${LR}"
    mkdir -p $CHECKPOINT_DIR
    
    WANDB_RUN_NAME="tinystories_17m_bs_${BS}_lr_${LR}"
    
    python -m cs336_basics.train \
        --vocab_size $VOCAB_SIZE \
        --context_length $CONTEXT_LENGTH \
        --d_model $D_MODEL \
        --num_layers $NUM_LAYERS \
        --num_heads $NUM_HEADS \
        --d_ff $D_FF \
        --rope_theta $ROPE_THETA \
        --batch_size $BS \
        --max_iters $MAX_ITERS \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --beta1 $BETA1 \
        --beta2 $BETA2 \
        --warmup_iters $WARMUP_ITERS \
        --max_grad_norm $MAX_GRAD_NORM \
        --train_data $TRAIN_DATA \
        --val_data $VAL_DATA \
        --checkpoint_dir $CHECKPOINT_DIR \
        --save_interval $SAVE_INTERVAL \
        --log_interval $LOG_INTERVAL \
        --eval_interval $EVAL_INTERVAL \
        --eval_iters $EVAL_ITERS \
        --device $DEVICE \
        $( [ "$USE_WANDB" = true ] && echo "--use_wandb" ) \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $WANDB_RUN_NAME
    
    echo "Finished batch_size=$BS, lr=$LR"
    echo ""
done