# 2 Byte-Pair Encoding (BPE) Tokenizer

## 2.1 The Unicode Standard

(a) What Unicode character does `chr(0)` return?
`'\x00'`

(b) How does this character’s string representation `(__repr__())` differ from its printed representa-tion?
repr会显示 `'\x00'`，但是printed representation是空的。

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
使用print打印这个是空的，不适用print会调用它的 `__repr__()` 方法来显示。

## 2.2 Unicode Encodings

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
Deliverable: A one-to-two sentence response.

UTF-8的code unit只有1个字节，但是UTF-16的code unit有2个字节，UTF-32的code unit有4个字节。这就导致
对于一段话，使用UTF-16编码可能会比UTF-8编码长很多。这也说明UTF-8编码对于句子的压缩程度较高。

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```Python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Deliverable: An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.

有些字符的utf-8编码是多个字节的，必须一起解码才可以，比如 "中"。

(c) Give a two byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation.

`'\xe4\xb8'`, 因为这是 "中" 的utf-8编码的前面两个byte，不是一个完整的byte sequence。

## 2.5 Experimenting with BPE Tokenizer Training

### BPE Training on TinyStories

使用的是 `TinyStoriesV2-GPT4-valid.txt` 数据集。
(a) 训练耗时 0.52 min，占用内存 0.04 GB。最长的 token 是 `b' accomplishemnt'`。这很合理，因为这个单词有很多常见组合。

(b) `max_freq_pair = max(pair_counts.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))[0]` 这段代码是最耗时间的，因为它会遍历一遍 `pair_counts` 变量，然后在遍历 `count` 最大的那些 `pair`，比较他们的字典序。
可以使用最小堆来优化，优化完之后训练耗时 0.02 min，占用内存 0.05 GB.

### BPE Training on OpenWebText

使用的是 `owt_valid.txt` 数据集。
(a) 最长的 token 是 `b'--------------------------------`。这很合理，因为这个单词经常在互联网上出现。

(b) 因为 TinyStoriesV2 的数据集用词偏幼稚化，所以它的压缩率会更大一点，而 OpenWebText 的词语很广泛，所以压缩率会小一点。

## 2.7 Experiments

(a) 从 TinyStories 和 OpenWebText 中抽取 10 个文档。使用你之前训练的 TinyStories 和 OpenWebText 分词器（词汇量大小分别为 10K 和 32K），将这些抽样的文档编码为整数 ID。每个分词器的压缩率（字节/token）是多少？

TinyStories (10K vocab) tokenizer on TinyStories docs: 4.0097 bytes/token
OpenWebText (32K vocab) tokenizer on OpenWebText docs: 4.5082 bytes/token

(b) 如果你用 TinyStories 分词器对 OpenWebText 样本进行分词，会发生什么？比较压缩率和/或定性描述发生的情况。

TinyStories (10K vocab) tokenizer on OpenWebText docs: 3.3692 bytes/token
那肯定压缩率会下降，因为数据分布不一样，TinyStories tokenizer是在 TinyStories 上进行训练的，但是在对于OpenWebText的一些bytes就没有压缩很多。

(c) 估算你的分词器的吞吐量（例如，字节/秒）。对 Pile 数据集（825GB 文本）进行分词需要多长时间？

Throughput: 2431396.58 bytes/sec
Time to tokenize Pile (825GB): 101.20 hours

(d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练和开发数据集编码为整数 token ID 序列。我们要稍后使用这些数据来训练我们的语言模型。我们建议将 token ID 序列化为 uint16 数据类型的 NumPy 数组。为什么 uint16 是一个合适的选择？

uint16 的范围（0-65535）足以容纳 10k 和 32k 的词汇表 ID

# 3 Transformer Language Model Architecture

## 3.6 The Full Transformer LM

**(a) 考虑 GPT-2 XL，其配置如下：**

- **vocab_size (词表大小):** 50,257
- **context_length (上下文长度):** 1,024
- **num_layers (层数):** 48
- **d_model (模型维度):** 1,600
- **num_heads (注意力头数):** 25
- **d_ff (前馈网络维度):** 6,400

假设我们使用此配置构建模型。该模型有多少个可训练参数？假设每个参数都使用单精度浮点数表示，仅加载该模型需要多少内存？

embedding层：50,257 _ 1600 = 80,411,200
一个TransformerBlock的参数：
rmsnorm: 1600 _ 2 = 3,200
attention: 1600 _ 1600 _ 4 = 10,240,000
mlp: 1600 _ 6400 _ 3 = 30,720,000
单层总计：3,200 + 10,240,000 + 30,720,000 = 40,963,200
那么TransformerBLocks的总参数量 = 40,963,200 _ 48 = 1,966,233,600
final norm: 1600
output_embed层： 1600 _ 50,257 = 80,411,200

总参数量：80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 = 2,127,057,600
内存占用（单精度浮点数，4字节/参数）：2,127,057,600 \* 4 bytes = 8,508,230,400 bytes ≈ 7.92 GB

**(b) 识别完成 GPT-2 XL 形状模型的单次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs（浮点运算次数）？假设输入序列包含 `context_length` 个 token。**

记 `context_length = L`, `d_model = d`, `d_ff = d_ff`, `vocab_size = V`, `num_layers = N`。

我们只统计**矩阵乘法**（matmul / GEMM）的 FLOPs，使用标准近似：
若 $A\in\mathbb{R}^{m\times k}$ 与 $B\in\mathbb{R}^{k\times n}$ 相乘，则 FLOPs $\approx 2mkn$（乘加各算 1 次）。

**单个 TransformerBlock 的矩阵乘法：**

1. **QKV 投影**（合并为一个大矩阵）：

- $(L\times d)\cdot(d\times 3d)$，FLOPs：$2\,L\,d\,(3d)=6Ld^2$

2. **Attention 打分**（按多头拆分不改变总 FLOPs）：

- $QK^T$：$(L\times d_k)\cdot(d_k\times L)$，所有头合计 FLOPs：$2L^2d$

3. **Attention 加权求和**：

- $AV$：$(L\times L)\cdot(L\times d_k)$，所有头合计 FLOPs：$2L^2d$

4. **Attention 输出投影**：

- $(L\times d)\cdot(d\times d)$，FLOPs：$2Ld^2$

5. **MLP（SwiGLU，3 个线性层）**：

- gate：$(L\times d)\cdot(d\times d_{ff})$，FLOPs：$2Ld\,d_{ff}$
- up：$(L\times d)\cdot(d\times d_{ff})$，FLOPs：$2Ld\,d_{ff}$
- down：$(L\times d_{ff})\cdot(d_{ff}\times d)$，FLOPs：$2Ld\,d_{ff}$

所以单层 TransformerBlock 的 matmul FLOPs 为：

$$
6Ld^2 + (2L^2d + 2L^2d) + 2Ld^2 + (2Ld\,d_{ff}+2Ld\,d_{ff}+2Ld\,d_{ff})
$$

$$
= 8Ld^2 + 4L^2d + 6Ld\,d_{ff}
$$

**所有 TransformerBlocks：** $N\,(8Ld^2 + 4L^2d + 6Ld\,d_{ff})$

**最后的 vocab 投影（output embedding / lm head）：**

- $(L\times d)\cdot(d\times V)$，FLOPs：$2LdV$

**GPT-2 XL 数值代入（$L=1024, d=1600, d_{ff}=6400, V=50257, N=48$）：**

- 单个 block：$8Ld^2 + 4L^2d + 6Ld\,d_{ff} = 90,596,966,400$ FLOPs
- 48 个 blocks：$4,348,654,387,200$ FLOPs
- lm head：$2LdV = 164,682,137,600$ FLOPs

**总 matmul FLOPs（单次前向传播，batch=1，长度 L）：**

$$
4,348,654,387,200 + 164,682,137,600 = 4,513,336,524,800\ \text{FLOPs} \approx 4.51\times 10^{12}
$$

（注：未计入 softmax、mask、RoPE 旋转、RMSNorm、激活/逐元素乘加等非矩阵乘法开销。）

**(c) 根据上述分析，模型的哪些部分需要的 FLOPs 最多？**

Attention层需要的FLOPs: N _ (8 _ L _ d^2 + 4 _ L^2 _ d), 占用 29.44%
MLP层需要的FLOPs: N _ 6 _ L _ d _ d_ff, 占用 66.91%
LM Head需要的FLOPs: 2 _ L _ d _ v, 占用 3.65%

**(d) 对 GPT-2 small（12 层，768 `d_model`，12 头）、GPT-2 medium（24 层，1024 `d_model`，16 头）和 GPT-2 large（36 层，1280 `d_model`，20 头）重复上述分析。随着模型规模的增加，Transformer 语言模型的哪些部分占总 FLOPs 的比例会相应增加或减少？**

| Model        | Total FLOPs | Attn % | MLP %  | Head % |
| :----------- | :---------- | :----- | :----- | :----- |
| GPT-2 Small  | 3.50e+11    | 27.64% | 49.75% | 22.61% |
| GPT-2 Medium | 1.03e+12    | 29.93% | 59.87% | 10.20% |
| GPT-2 Large  | 2.26e+12    | 29.96% | 64.20% | 5.84%  |
| GPT-2 XL     | 4.51e+12    | 29.44% | 66.91% | 3.65%  |

随着模型的规模增加，Transformer 语言模型的 MLP (Feed Forward 层) 占用的 FLOPs 比例会相应增加，而 LM Head 占用的 FLOPs 比例会逐渐减少。

**(e) 将 GPT-2 XL 的上下文长度增加到 16,384。单次前向传播的总 FLOPs 如何变化？模型组件的 FLOPs 相对贡献如何变化？**

| Model                  | Total FLOPs | Attn % | MLP %  | Head % |
| :--------------------- | :---------- | :----- | :----- | :----- |
| GPT-2 XL (Context 1k)  | 4.51e+12    | 29.44% | 66.91% | 3.65%  |
| GPT-2 XL (Context 16k) | 1.50e+14    | 65.92% | 32.32% | 1.76%  |

注意力模块占用的 FLOPs 会急剧增加。这是因为 QKV 投影和 MLP 的计算量随序列长度 $L$ 线性增长 ($O(L)$)，而 Attention 打分和加权部分的计算复杂度随序列长度呈平方增长 ($O(L^2)$)。当 $L$ 从 1,024 增加到 16,384 时，这部分 $O(L^2)$ 的开销开始占据主导地位。

# 4 Training a Transformer LM

## 4.2 The SGD Optimizer

### 4.2.1 Implementing SGD in PyTorch

lr = 1e1的时候loss下降很缓慢，
lr = 1e2的时候loss下降很快，
lr = 1e3的时候loss就直接上升了。

## 4.3 AdamW

### AdamW 资源核算（adamwAccounting）

让我们计算使用 AdamW 运行训练需要多少内存和计算资源。假设我们对所有张量都使用 float32。

**(a) 运行 AdamW 需要多少峰值内存？**

根据参数（parameters）、激活（activations）、梯度（gradients）和优化器状态（optimizer state）的内存使用情况分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表达你的答案。假设 `d_ff = 4 × d_model`。

为简化起见，在计算激活的内存使用时，只考虑以下组件：

- Transformer block
  - RMSNorm(s)
  - 多头自注意力子层：QKV 投影、Q^T 矩阵乘法、softmax、值的加权求和、输出投影
  - 位置编码前馈：W₁ 矩阵乘法、SiLU、W₂ 矩阵乘法
- 最终 RMSNorm
- 输出嵌入
- logits 上的交叉熵

记：V = vocab_size, L = context_length, N = num_layers, d = d_model, H = num_heads, B = batch_size

**Parameters (参数):**

$$
P = 2Vd + N(2d + 16d^2) + d
$$

其中 $16d^2 = 4d^2$ (attention) + $12d^2$ (MLP，因为 $d_{ff} = 4d$)

**Activations (激活):**

每个 Transformer Block 包含：

- RMSNorm (第一个): $BLd$
- QKV 投影: $3BLd$
- Attention scores: $BHL^2$
- Softmax: $BHL^2$
- 加权求和输出: $BLd$ (因为 $H \times d_k = d$)
- 输出投影: $BLd$
- RMSNorm (第二个): $BLd$
- W₁ 矩阵乘法: $4BLd$
- SiLU: $4BLd$
- W₂ 输出: $BLd$

单个 Block 小计: $16BLd + 2BHL^2$

N 个 Blocks: $BN(16Ld + 2HL^2)$

Block 外部组件：

- 最终 RMSNorm: $BLd$
- 输出嵌入 (logits): $BLV$
- Cross-entropy softmax: $BLV$

$$
A = BN(16Ld + 2HL^2) + BLd + 2BLV
$$

**Gradients (梯度):**

$$
G = P = 2Vd + N(2d + 16d^2) + d
$$

**Optimizer State (优化器状态):**

$$
O = 2P = 2(2Vd + N(2d + 16d^2) + d)
$$

**总内存 (float32，4 bytes/值):**

$$
\text{Total Memory (bytes)} = 4(P + A + G + O) = 4(4P + A)
$$

$$
= 4[4(2Vd + N(2d + 16d^2) + d) + BN(16Ld + 2HL^2) + BLd + 2BLV]
$$

简化为：

$$
= 16(2Vd + N(2d + 16d^2) + d) + 4BN(16Ld + 2HL^2) + 4BLd + 8BLV
$$

**(b) 将你的答案实例化为 GPT-2 XL 形状的模型，以获得仅依赖于 `batch_size` 的表达式。在 80GB 内存内，你能使用的最大批次大小是多少？**

代入 GPT-2 XL 参数：$V = 50257, L = 1024, N = 48, d = 1600, H = 25$

**Parameters:**

$$
P = 2 \times 50257 \times 1600 + 48 \times (2 \times 1600 + 16 \times 1600^2) + 1600
$$

$$
= 160,822,400 + 48 \times 40,963,200 + 1,600 = 2,127,057,600
$$

内存: $2,127,057,600 \times 4 = 8,508,230,400$ bytes $\approx 8.51$ GB

**Activations (per batch):**

$$
A = B \times 48 \times (16 \times 1024 \times 1600 + 2 \times 25 \times 1024^2) + B \times 1024 \times 1600 + 2B \times 1024 \times 50257
$$

$$
= B \times 48 \times 78,643,200 + B \times 1,638,400 + B \times 102,926,336
$$

$$
= B \times 3,879,438,336
$$

内存: $B \times 3,879,438,336 \times 4 = B \times 15,517,753,344$ bytes $\approx B \times 15.52$ GB

**Gradients:** $8.51$ GB (同 Parameters)

**Optimizer State:** $2 \times 8.51 = 17.02$ GB

**总内存:**

$$
\text{Total} = B \times 15.52 + (8.51 + 8.51 + 17.02) = B \times 15.52 + 34.04 \text{ GB}
$$

**最大 batch size (80GB 限制):**

$$
B \times 15.52 + 34.04 \leq 80
$$

$$
B \leq \frac{80 - 34.04}{15.52} = \frac{45.96}{15.52} \approx 2.96
$$

因此 $B_{\max} = 2$

**交付物：** 内存 = $15.52 \times \text{batch\_size} + 34.04$ GB，最大批次大小为 **2**。

**(c) 运行一步 AdamW 需要多少 FLOPs？**

一步 AdamW 训练包含三个部分：

**1. 前向传播 (Forward pass):**
从 3.6(b) 我们知道单样本前向传播的矩阵乘法 FLOPs 为：

$$
F_{\text{forward}} = N(8Ld^2 + 4L^2d + 6Ld \cdot d_{ff}) + 2LdV
$$

对于 batch size = B：

$$
\text{Forward FLOPs} = B \cdot F_{\text{forward}}
$$

**2. 反向传播 (Backward pass):**
根据 [Kaplan et al., 2020] 和 [Hoffmann et al., 2022]，反向传播的 FLOPs 是前向传播的两倍：

$$
\text{Backward FLOPs} = 2B \cdot F_{\text{forward}}
$$

**3. 优化器更新 (AdamW optimizer):**
对每个参数 $\theta$，AdamW 需要更新：

- $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
- $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
- 偏差修正和参数更新

每个参数约需 13 个浮点运算，总计：

$$
\text{Optimizer FLOPs} = 13P = 13(2Vd + N(2d + 16d^2) + d)
$$

**总 FLOPs (一步训练):**

$$
\text{Total} = 3B \cdot F_{\text{forward}} + 13P
$$

$$
= 3B[N(8Ld^2 + 4L^2d + 6Ld \cdot d_{ff}) + 2LdV] + 13(2Vd + N(2d + 16d^2) + d)
$$

因为 $3B \cdot F_{\text{forward}} \gg 13P$（前向+反向的计算量远大于优化器更新），通常可以简化为：

$$
\text{Total} \approx 3B[N(8Ld^2 + 4L^2d + 6Ld \cdot d_{ff}) + 2LdV]
$$

**交付物：** $\text{FLOPs} = 3B[N(8Ld^2 + 4L^2d + 6Ld_{ff}d) + 2LdV] + 13P$，其中优化器项 $13P$ 通常可忽略。

**(d) 模型 FLOPs 利用率（MFU）定义为观察到的吞吐量（每秒 token 数）相对于硬件理论峰值 FLOP 吞吐量的比率 [Chowdhery et al., 2022]。**

一块 NVIDIA A100 GPU 的理论峰值为 19.5 teraFLOP/s（用于 float32 操作）。假设你能够达到 50% 的 MFU，在单个 A100 上训练 GPT-2 XL 400K 步、批次大小为 1024 需要多长时间？根据 [Kaplan et al., 2020] 和 [Hoffmann et al., 2022]，假设反向传播的 FLOPs 是前向传播的两倍。

**计算过程：**

从 3.6(b)，单次前向传播（batch=1，L=1024）的 FLOPs：

$$
F_{\text{forward}} = 4.513 \times 10^{12} \text{ FLOPs}
$$

对于 batch*size = 1024，一步训练包含前向和反向传播：

$$
\text{FLOPs per step} = 3 \times B \times F_{\text{forward}} = 3 \times 1024 \times 4.513 \times 10^{12}
$$

$$
= 1.386 \times 10^{16} \text{ FLOPs}
$$

训练 400,000 步的总 FLOPs：

$$
\text{Total FLOPs} = 400,000 \times 1.386 \times 10^{16} = 5.544 \times 10^{21} \text{ FLOPs}
$$

实际吞吐量（50% MFU）：

$$
\text{Actual throughput} = 19.5 \times 10^{12} \times 0.5 = 9.75 \times 10^{12} \text{ FLOP/s}
$$

训练时间：

$$
\text{Time} = \frac{5.544 \times 10^{21}}{9.75 \times 10^{12}} = 5.686 \times 10^8 \text{ 秒}
$$

转换为天：

$$
\frac{5.686 \times 10^8}{86,400} \approx 6,581 \text{ 天} \approx 18 \text{ 年}
$$

**交付物：** 约 **6,581 天**（约 18 年）。注意：这个结果说明批次大小 1024 对于单个 A100 是不现实的（从 (b) 我们知道最大批次大小只有 2），实际训练需要使用分布式训练或梯度累积。

# 7 Experiments

## 7.2 TinyStories
### LR
(a) 
[here](https://wandb.ai/2310572998jin/transformer-lm?nw=nwuser2310572998jin) 可以看到关于lr和loss之间的关系，只需要看后缀是 _lr\_{lr} 的图即可。
lr=3e-3的时候，模型的val loss最小，为1.37。

(b)
lr=1e-2的时候，训练的loss就会上下起伏很大。
目前认为使得训练发散的lr和最佳lr之间的关系是： 使得训练发散的lr/最佳lr = 3.33，但是为了得到更加精准的答案，需要利用3e-3和1e-2之间的学习率再去训练模型，找到更加精准的学习率。

### Batch Size
[here](https://wandb.ai/2310572998jin/transformer-lm?nw=nwuser2310572998jin) 可以看到关于batch size和loss之间的关系。只需要看含有bs的图即可。
一般来说，batch size越大那么训练效果越好。而batch size和lr之间的关系满足Linear scaling rule: lr ∝ batch_size，如果batch_size相对于BS_BASE扩大k倍，那么最佳学习率相对于LR_BASE(在BS_BASE下的最佳学习率)也扩大k倍。

### Generate Text
**Prompt**:
Once upon a time

**Generated Text**:
Once upon a time, there was a strong and independent boy named Tim. He liked to play with his toy pipe. One day, Tim went to the park to play with his pipe.

While playing, a big wind came and took Tim's pipe away. Tim was sad and began to cry. He didn't want to lose his favorite toy. So, he ran to find it.

Tim looked and looked for his pipe. He found it under a big tree. He was so happy! But then, a big bird saw Tim's pipe and wanted it. The bird took Tim's pipe and flew away. Tim cried, and he went home without his favorite toy.

## 7.3 Ablations and architecture modification
关于消融实验相关的 val loss 可以在 [here](https://wandb.ai/2310572998jin/transformer-lm-ablation?nw=nwuser2310572998jin) 中看到。

### rmsnorm
- 没有rmsnorm

将模型中的pre rmsnorm去掉，这样模型就没有rmsnorm了。这样在lr=3e-3时训练得到的train loss和val loss很不稳定，甚至有些时候直接让loss=nan了。
将lr缩小可以让训练更加稳定。

-post rmsnorm

效果相比于pre norm，val loss相差0.01。

### position embeddings
去掉 RoPE 之后模型训练不稳定。

### SiLU
使用SiLU和使用SwiGLU差别不大，只是前者效果差一些，最终的val loss差了0.06。
  
