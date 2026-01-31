#!/bin/bash

# 压缩项目为 tar.gz，排除不必要的大文件和目录

PROJECT_NAME="assignment1-basics"
OUTPUT_FILE="../${PROJECT_NAME}.tar.gz"

echo "Creating submission archive: ${OUTPUT_FILE}"
echo "Excluding: .venv, .git, data/, checkpoints/*.pt, __pycache__, *.pyc"

# 使用 tar 排除模式
tar -czf "${OUTPUT_FILE}" \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='data/*' \
    --exclude='checkpoints/*.pt' \
    --exclude='checkpoints/**/*.pt' \
    --exclude='__pycache__' \
    --exclude='**/__pycache__' \
    --exclude='.claude' \
    --exclude='.ruff_cache' \
    --exclude='assets' \
    --exclude='wandb' \
    --exclude='*.pyc' \
    --exclude='**/*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.egg-info' \
    --exclude='.DS_Store' \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Archive created successfully: ${OUTPUT_FILE}"
    echo ""
    ls -lh "${OUTPUT_FILE}"
    echo ""
    echo "Total files in archive:"
    tar -tzf "${OUTPUT_FILE}" | wc -l
else
    echo ""
    echo "✗ Failed to create archive"
    exit 1
fi