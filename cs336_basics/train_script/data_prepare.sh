python -m cs336_basics.data \
    --vocab cs336_basics/tokenizer/res/tinystories_vocab.json \
    --merges cs336_basics/tokenizer/res/tinystories_merges.txt \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/TinyStoriesV2-GPT4-train_tokens.npy

python -m cs336_basics.data \
    --vocab cs336_basics/tokenizer/res/tinystories_vocab.json \
    --merges cs336_basics/tokenizer/res/tinystories_merges.txt \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output data/TinyStoriesV2-GPT4-valid_tokens.npy