# CS336 作业 1 任务日志

这个文件用来追踪 Transformer 各个组件的实现进度。我们将严格按照这个计划一步步执行。

## 1. 项目结构设置
- [x] 创建 `cs336_basics/model` 目录

## 2. 基础组件实现
这些是构建 Transformer 所需的基础模块。
- [ ] **任务 2.1**: 创建 `cs336_basics/model/linear.py`
    - 实现 `linear` 函数或类
    - 目标: 通过 `test_linear` 测试
- [ ] **任务 2.2**: 创建 `cs336_basics/model/embedding.py`
    - 实现 `Embedding` 层逻辑
    - 目标: 通过 `test_embedding` 测试
- [ ] **任务 2.3**: 创建 `cs336_basics/model/activation.py`
    - 实现 `SiLU` 激活函数
    - 目标: 通过 `test_silu_matches_pytorch` 测试
- [ ] **任务 2.4**: 创建 `cs336_basics/model/normalization.py`
    - 实现 `RMSNorm` 归一化
    - 目标: 通过 `test_rmsnorm` 测试
- [ ] **任务 2.5**: 创建 `cs336_basics/model/utils.py` (文件名待定)
    - 实现 `Softmax`, `CrossEntropy`
    - 目标: 通过相关的工具函数测试

## 3. 注意力机制
实现核心的 Attention 逻辑。
- [ ] **任务 3.1**: 创建 `cs336_basics/model/attention.py`
    - 实现 `scaled_dot_product_attention`
    - 实现 `MultiHeadSelfAttention` (先不带 RoPE)
    - 目标: 通过 `test_scaled_dot_product_attention`, `test_multihead_self_attention` 测试
- [ ] **任务 3.2**: 创建 `cs336_basics/model/rope.py`
    - 实现旋转位置编码 (Rotary Positional Embeddings)
    - 目标: 通过 `test_rope` 测试
- [ ] **任务 3.3**: 更新 `cs336_basics/model/attention.py`
    - 实现 `MultiHeadSelfAttentionWithRoPE`
    - 目标: 通过 `test_multihead_self_attention_with_rope` 测试

## 4. 前馈网络 (Feed Forward Networks)
- [ ] **任务 4.1**: 创建 `cs336_basics/model/ffn.py`
    - 实现 `SwiGLU`
    - 目标: 通过 `test_swiglu` 测试

## 5. Transformer 架构
组装所有组件。
- [ ] **任务 5.1**: 创建 `cs336_basics/model/transformer.py`
    - 实现 `TransformerBlock`
    - 实现 `TransformerLM`
    - 目标: 通过 `test_transformer_block`, `test_transformer_lm` 测试

## 6. 训练工具
- [ ] **任务 6.1**: 创建 `cs336_basics/optimizer.py` 或类似文件
    - 实现 `AdamW` 优化器
    - 实现学习率调度 (Learning Rate Schedule)
    - 实现梯度裁剪 (Gradient Clipping)
    - 实现 Checkpointing (保存/加载)

## 7. 集成
- [ ] **任务 7.1**: 更新 `tests/adapters.py`，从我们新创建的文件中导入函数，替换掉原本的 NotImplementedError。
