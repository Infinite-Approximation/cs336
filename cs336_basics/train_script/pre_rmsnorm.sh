#!/bin/bash

# 这个脚本用来测试去掉RMSNorm后的模型表现，以验证RMSNorm的效果
# Hyperparameters from assignment
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
ROPE_THETA=10000

# Batch size and steps calculated to process ~327M tokens
BATCH_SIZE=64
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
WANDB_PROJECT="transformer-lm-ablation"

# Device
DEVICE="cuda:0"

# 3e-3是最优的学习率，测试降低学习率是否能提高稳定性
LEARNING_RATES=(1e-3 3e-3)

# Loop through learning rates
for LR in "${LEARNING_RATES[@]}"; do
    echo "=========================================="
    echo "Starting training with learning rate: $LR"
    echo "=========================================="
    
    # Create checkpoint directory for this learning rate
    CHECKPOINT_DIR="checkpoints/tinystories_17m_lr_${LR}_no_rmsnorm"
    mkdir -p $CHECKPOINT_DIR
    
    # Set WandB run name
    WANDB_RUN_NAME="tinystories_17m_lr_${LR}_no_rmsnorm"
    
    # Run training
    python -m cs336_basics.train \
        --vocab_size $VOCAB_SIZE \
        --context_length $CONTEXT_LENGTH \
        --d_model $D_MODEL \
        --num_layers $NUM_LAYERS \
        --num_heads $NUM_HEADS \
        --d_ff $D_FF \
        --rope_theta $ROPE_THETA \
        --batch_size $BATCH_SIZE \
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
        --wandb_run_name $WANDB_RUN_NAME \
        --no_pre_rmsnorm
    
    echo "Finished training with learning rate: $LR"
    echo ""
done

echo "All learning rate experiments completed!"