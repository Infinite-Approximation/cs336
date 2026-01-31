TRAIN_DATA="data/train_tokens.npy"
VAL_DATA="data/train_tokens.npy"

# Run training with CPU-optimized settings
python -m cs336_basics.train \
    --vocab_size 10000 \
    --context_length 128 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 4 \
    --d_ff 1024 \
    --batch_size 32 \
    --max_iters 2000 \
    --warmup_iters 100 \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --device cpu \
    --log_interval 10 \
    --eval_interval 100 \
    --save_interval 500