# NanoChat 项目地图

> 最小化全栈 ChatGPT 克隆项目完全解析

## 一、项目概述

NanoChat 是一个教育性质的 ChatGPT 克隆项目，包含从分词器训练到 Web 服务的完整流程：

```
数据 → 分词器 → 预训练 → 中间训练 → SFT微调 → RLHF → Web服务
```

**核心特点：**
- ~6700 行 Python 代码 + ~500 行 Rust 代码
- GPT-2 风格架构 + 现代优化（RoPE、GQA、RMSNorm）
- 支持分布式训练（DDP）
- 内置计算器工具调用能力

---

## 二、目录结构总览

```
nanochat/
├── nanochat/          # 核心库（14个模块）
│   ├── gpt.py         # ⭐ GPT模型定义
│   ├── engine.py      # ⭐ 推理引擎 + KV缓存
│   ├── tokenizer.py   # ⭐ BPE分词器
│   ├── dataloader.py  # 数据加载器
│   ├── dataset.py     # 数据集管理
│   ├── common.py      # 通用工具
│   ├── adamw.py       # AdamW优化器
│   ├── muon.py        # Muon优化器
│   ├── configurator.py # 配置解析
│   ├── checkpoint_manager.py # 检查点管理
│   ├── core_eval.py   # 核心评估
│   ├── loss_eval.py   # 损失评估
│   ├── execution.py   # 代码执行
│   ├── report.py      # 报告生成
│   └── ui.html        # Web界面
│
├── scripts/           # 训练/推理脚本（11个）
│   ├── tok_train.py   # 分词器训练
│   ├── tok_eval.py    # 分词器评估
│   ├── base_train.py  # ⭐ 基座模型预训练
│   ├── base_eval.py   # 基座模型评估
│   ├── base_loss.py   # 损失计算
│   ├── mid_train.py   # 中间训练
│   ├── chat_sft.py    # ⭐ SFT微调
│   ├── chat_rl.py     # RLHF训练
│   ├── chat_eval.py   # 聊天评估
│   ├── chat_cli.py    # 命令行聊天
│   └── chat_web.py    # ⭐ Web服务
│
├── tasks/             # 评估任务（8个）
│   ├── common.py      # 任务基类
│   ├── arc.py         # ARC推理
│   ├── mmlu.py        # MMLU知识
│   ├── gsm8k.py       # 数学推理
│   ├── humaneval.py   # 代码生成
│   ├── smoltalk.py    # 对话数据
│   ├── spellingbee.py # 拼写任务
│   └── customjson.py  # 自定义JSON
│
├── rustbpe/           # Rust BPE分词器
│   ├── src/lib.rs     # BPE训练核心
│   └── Cargo.toml     # Rust配置
│
├── tests/             # 测试
└── pyproject.toml     # 项目配置
```

---

## 三、核心模块详解

### 3.1 GPT模型 (`gpt.py`) ⭐最重要

```
GPT
├── transformer
│   ├── wte (Embedding)      # 词嵌入层
│   └── h (ModuleList)       # N个Block
│       └── Block
│           ├── attn (CausalSelfAttention)
│           │   ├── c_q (Linear)    # Query投影
│           │   ├── c_k (Linear)    # Key投影
│           │   ├── c_v (Linear)    # Value投影
│           │   └── c_proj (Linear) # 输出投影
│           └── mlp (MLP)
│               ├── c_fc (Linear)   # 4x扩展
│               └── c_proj (Linear) # 投影回
└── lm_head (Linear)         # 语言模型头
```

**关键创新：**
| 特性 | 传统GPT-2 | NanoChat |
|------|----------|----------|
| 位置编码 | 可学习绝对位置 | RoPE旋转位置 |
| 归一化 | LayerNorm | RMSNorm(无参数) |
| 激活函数 | GELU | ReLU² |
| 注意力 | MHA | GQA支持 |
| 权重 | tied embedding | untied |

**前向传播流程：**
```python
def forward(idx, targets=None, kv_cache=None):
    # 1. 词嵌入
    x = wte(idx)           # (B, T) -> (B, T, C)
    x = norm(x)            # RMSNorm

    # 2. 获取旋转位置编码
    cos_sin = (cos[:, T0:T0+T], sin[:, T0:T0+T])

    # 3. 通过N个Block
    for block in h:
        x = block(x, cos_sin, kv_cache)
    x = norm(x)

    # 4. 语言模型头
    logits = lm_head(x)    # (B, T, C) -> (B, T, vocab)
    logits = 15 * tanh(logits / 15)  # Softcap

    # 5. 计算损失或返回logits
    if targets:
        return cross_entropy(logits, targets)
    return logits
```

### 3.2 推理引擎 (`engine.py`)

```python
class KVCache:
    """KV缓存加速推理"""
    # 形状: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
    def insert_kv(layer_idx, k, v):
        # 将新的k,v插入缓存，返回完整视图

class Engine:
    def generate(tokens, num_samples, max_tokens, temperature, top_k):
        # 1. Prefill: 处理输入提示词
        kv_cache_prefill = KVCache(batch_size=1, ...)
        logits = model.forward(ids, kv_cache=kv_cache_prefill)

        # 2. 复制KV缓存用于多样本生成
        kv_cache_decode = KVCache(batch_size=num_samples, ...)
        kv_cache_decode.prefill(kv_cache_prefill)

        # 3. 解码循环
        while not done:
            logits = model.forward(ids, kv_cache=kv_cache_decode)
            next_token = sample_next_token(logits, temperature, top_k)

            # 4. 工具调用检测 (计算器)
            if token == python_start:
                in_python_block = True
            elif token == python_end:
                result = use_calculator(expr)
                forced_tokens = [output_start, *result_tokens, output_end]

            yield token_column, token_masks
```

### 3.3 分词器 (`tokenizer.py`)

**两种实现：**
1. `HuggingFaceTokenizer` - 用于训练和调试
2. `RustBPETokenizer` - 用于高效推理（基于tiktoken）

**特殊Token：**
```python
SPECIAL_TOKENS = [
    "<|bos|>",            # 文档开始
    "<|user_start|>",     # 用户消息开始
    "<|user_end|>",
    "<|assistant_start|>", # 助手消息开始
    "<|assistant_end|>",
    "<|python_start|>",   # Python工具调用
    "<|python_end|>",
    "<|output_start|>",   # 工具输出
    "<|output_end|>",
]
```

**对话渲染示例：**
```
<|bos|><|user_start|>你好<|user_end|><|assistant_start|>你好！<|assistant_end|>
```

---

## 四、训练流程

### 4.1 预训练 (`base_train.py`)

```
┌─────────────────────────────────────────────────────────┐
│                    预训练流程                            │
├─────────────────────────────────────────────────────────┤
│  数据: FineWeb-Edu 100B (Parquet格式)                    │
│  ↓                                                       │
│  tokenizing_distributed_data_loader                      │
│  ↓                                                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 训练循环:                                        │    │
│  │  for step in range(num_iterations):             │    │
│  │    # 梯度累积                                    │    │
│  │    for micro_step in range(grad_accum_steps):   │    │
│  │      loss = model(x, y)                         │    │
│  │      loss.backward()                            │    │
│  │    # 梯度裁剪                                    │    │
│  │    clip_grad_norm_(params, grad_clip)           │    │
│  │    # 更新参数                                    │    │
│  │    adamw_optimizer.step()  # embedding/lm_head  │    │
│  │    muon_optimizer.step()   # 其他Linear层        │    │
│  └─────────────────────────────────────────────────┘    │
│  ↓                                                       │
│  输出: base_checkpoints/d{depth}/                        │
└─────────────────────────────────────────────────────────┘
```

**优化器配置：**
- **AdamW**: embedding (lr=0.2) + lm_head (lr=0.004)
- **Muon**: 所有Linear层 (lr=0.02, momentum=0.95)

### 4.2 SFT微调 (`chat_sft.py`)

```
┌─────────────────────────────────────────────────────────┐
│                    SFT微调流程                           │
├─────────────────────────────────────────────────────────┤
│  数据混合:                                               │
│  - ARC-Easy (2.3K) + ARC-Challenge (1.1K)              │
│  - GSM8K (8K)                                          │
│  - SmolTalk (10K)                                      │
│  - Identity对话 (1K)                                    │
│  - Spelling任务 (0.6K)                                  │
│  总计: ~23K条                                           │
│  ↓                                                       │
│  render_conversation() → (ids, mask)                    │
│  ↓                                                       │
│  只对 mask=1 的token计算损失（助手回复部分）              │
│  ↓                                                       │
│  输出: chatsft_checkpoints/d{depth}/                    │
└─────────────────────────────────────────────────────────┘
```

---

## 五、推理服务 (`chat_web.py`)

```
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Web服务                          │
├─────────────────────────────────────────────────────────┤
│  WorkerPool (多GPU并行)                                  │
│  ├── Worker 0 (GPU:0) ─ Engine + Model                  │
│  ├── Worker 1 (GPU:1) ─ Engine + Model                  │
│  └── ...                                                │
│                                                          │
│  端点:                                                   │
│  GET  /              → Chat UI (ui.html)                │
│  POST /chat/completions → 流式聊天                      │
│  GET  /health        → 健康检查                          │
│  GET  /stats         → 统计信息                          │
│                                                          │
│  防滥用:                                                 │
│  - 最大500条消息/请求                                    │
│  - 最大8000字符/消息                                     │
│  - 最大32000字符/对话                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 六、关键数据结构

### 6.1 GPTConfig
```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024   # 上下文长度
    vocab_size: int = 50304    # 词表大小
    n_layer: int = 12          # Transformer层数
    n_head: int = 6            # Query头数
    n_kv_head: int = 6         # KV头数(GQA)
    n_embd: int = 768          # 隐藏维度
```

### 6.2 模型深度与参数的关系
```python
# 从depth推导其他参数
num_layers = depth
model_dim = depth * 64        # 宽高比64
num_heads = max(1, ceil(model_dim / 128))  # head_dim=128
```

### 6.3 对话格式
```json
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "今天天气如何？"}
  ]
}
```

---

## 七、学习路线图（建议顺序）

```
┌────────────────────────────────────────────────────────┐
│  第1阶段: 理解基础                                       │
│  ├── 1. common.py      (工具函数)                       │
│  ├── 2. configurator.py (配置解析)                      │
│  └── 3. tokenizer.py   (BPE分词器)                      │
├────────────────────────────────────────────────────────┤
│  第2阶段: 核心模型                                       │
│  ├── 4. gpt.py         (GPT架构) ⭐                     │
│  │     - norm() RMSNorm                                │
│  │     - apply_rotary_emb() 旋转位置编码                │
│  │     - CausalSelfAttention 因果注意力                 │
│  │     - MLP 前馈网络                                   │
│  │     - Block Transformer块                           │
│  │     - GPT 完整模型                                   │
│  └── 5. engine.py      (推理引擎) ⭐                    │
│        - KVCache                                        │
│        - sample_next_token()                           │
│        - Engine.generate()                             │
├────────────────────────────────────────────────────────┤
│  第3阶段: 数据处理                                       │
│  ├── 6. dataset.py     (数据集)                         │
│  └── 7. dataloader.py  (数据加载)                       │
├────────────────────────────────────────────────────────┤
│  第4阶段: 训练流程                                       │
│  ├── 8. adamw.py + muon.py (优化器)                     │
│  ├── 9. base_train.py  (预训练脚本) ⭐                  │
│  └── 10. chat_sft.py   (SFT微调脚本) ⭐                 │
├────────────────────────────────────────────────────────┤
│  第5阶段: 服务部署                                       │
│  ├── 11. chat_cli.py   (命令行)                         │
│  └── 12. chat_web.py   (Web服务) ⭐                     │
├────────────────────────────────────────────────────────┤
│  第6阶段: 进阶                                           │
│  ├── 13. rustbpe/      (Rust分词器)                     │
│  ├── 14. chat_rl.py    (RLHF)                          │
│  └── 15. tasks/        (评估任务)                       │
└────────────────────────────────────────────────────────┘
```

---

## 八、核心代码片段

### 8.1 RMSNorm (无参数)
```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

### 8.2 RoPE旋转位置编码
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

### 8.3 ReLU² 激活
```python
def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()  # ReLU²
    x = self.c_proj(x)
    return x
```

### 8.4 Logits Softcap
```python
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

---

## 九、依赖关系图

```
pyproject.toml 依赖:
├── torch >= 2.8.0        # 深度学习框架
├── tiktoken >= 0.11.0    # 高效分词
├── tokenizers >= 0.22.0  # HuggingFace分词器
├── fastapi >= 0.117.1    # Web框架
├── uvicorn >= 0.36.0     # ASGI服务器
├── wandb >= 0.21.3       # 实验追踪
├── datasets >= 4.0.0     # 数据集加载
└── rustbpe (本地)        # Rust BPE训练
```

---

## 十、运行命令速查

```bash
# 分词器训练
python -m scripts.tok_train

# 预训练（单卡）
python -m scripts.base_train --depth=4

# 预训练（8卡DDP）
torchrun --nproc_per_node=8 -m scripts.base_train

# SFT微调
python -m scripts.chat_sft

# 命令行聊天
python -m scripts.chat_cli

# Web服务
python -m scripts.chat_web --num-gpus 4 --port 8000
```

---

**祝学习顺利！** 按照路线图一步步实现，你将深入理解 LLM 的完整训练和推理流程。
