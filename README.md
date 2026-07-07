# nanochat

从零复现大语言模型完整训练流程——从预训练（Pretrain）到监督微调（SFT），再到手写 KV Cache 推理加速和 Web 对话界面部署。本项目以学习为目的，搞清楚现代 LLM 训练全流程的每个细节，将理论与代码一一对应。

## 🚀 核心特性

- **全流程复现**：独立完成预训练(pretrain)、监督微调(SFT)、推理与Web部署的完整链路
- **分布式预训练**：基于 DDP（Distributed Data Parallel）在 8×A100-80GB 上完成分布式预训练，MFU 达 45.86%
- **现代 GPT 架构**：
  - 旋转位置编码（RoPE）
  - QK Norm（对 Q/K 做 RMSNorm）
  - MLP 使用 ReLU² 激活函数
  - 词嵌入与 lm_head 不共享权重（untied）
  - 滑动窗口注意力模式（SSSL 模式）
- **高效训练优化**：BOS对齐+Best-Fit数据打包（序列利用率100%无padding）、Flash Attention 3、torch.compile、MuonAdamW混合优化器、bf16混合精度
- **推理优化**：手动实现 KV Cache 优化自回归生成效率
- **Web 界面**：基于 FastAPI 的简单聊天网页

## 📊 项目详情

### 训练流程

我完整复现了大语言模型训练的两个核心阶段：

1. **预训练（Pretrain）阶段**
   - 数据集：OpenWebText
   - 目标：学习自然语言的统计分布和基本语义结构，获得通用语言理解能力（语言建模/补全能力）
   - 训练方式：8×A100-80GB 分布式训练（DDP），最大序列长度2048，total batch size 524,288 tokens
   - 训练结果：预训练验证集 bpb（bits per byte）降至 **0.7433**，训练总耗时约 **35 小时**，MFU（Model FLOPs Utilization）达 **45.86%**

2. **监督微调（SFT）阶段**
   - 数据集：SmolTalk 对话数据集 + 自定义 Identity 数据集（共约5万条对话）
   - 目标：在预训练模型基础上，让模型学会instruction-following（遵循指令）能力，能像AI助手一样对话
   - Identity数据集作用：让模型形成稳定的自我身份认知，能回答"你是谁"等问题
   - 训练结果：SFT跑了一晚上（约8-9小时），loss降到约0.2，人工测试能良好进行单轮对话

### Pretrain 与 SFT 的核心区别

两个阶段都是自回归语言建模（输入序列左移一位作为标签，交叉熵损失），关键区别在于：
- **数据格式**：Pretrain用普通连续文本；SFT把用户输入和assistant回复拼接成对话序列
- **Loss计算**：Pretrain对序列中每一个token都计算loss；SFT只对assistant回复部分计算loss（用户提问部分mask掉不算loss）
- **能力目标**：Pretrain学通用语言分布/补全能力；SFT学遵循指令对话的能力

### 为什么需要SFT？
预训练模型只会补全文本，但不一定能按照人类指令回答问题。SFT通过对话数据训练，让模型学会理解指令并给出符合期望的回答。

### 硬件与训练配置

| 配置项 | 数值 |
|--------|------|
| 预训练平台 | 8×NVIDIA A100-80GB（DDP分布式训练） |
| SFT 平台 | 单卡/多卡均可 |
| 模型参数量 | 约 1.68B |
| 模型层数 | 26层 |
| 词向量维度 | 1664 |
| 注意力头数 | 13（head_dim=128） |
| 词表大小 | 32,768（2^15） |
| 最大序列长度 | 2048 |
| 注意力模式 | SSSL（滑动窗口交替模式） |
| 精度 | bf16 混合精度 |
| 预训练 MFU | 45.86% |
| 预训练验证 bpb | 0.7433 |
| 预训练时长 | 约 35 小时 |
| 预训练峰值显存 | 约 74GB/卡 |
| SFT 最终 loss | 约 0.2 |

### 模型验证方法

- 预训练阶段：监控验证集 bpb（bits per byte）持续下降，最低达 0.7433
- SFT 阶段：观察训练 loss 持续下降至 ~0.2
- 人工测试 prompt，验证模型能够较好地完成单轮对话任务
- 项目达到了初始目的：完整理解从预训练到SFT到推理部署的全链路，把理论细节和代码一一对应

### 架构与关键技术

- **整体架构**：Decoder-only Transformer（GPT系列自回归语言模型架构）
- **RoPE旋转位置编码**：通过对query和key向量做旋转变换融入位置信息，相比传统可学习位置嵌入（如GPT-2），优势是不需要训练、不引入额外参数
- **QK Norm**：在 Q/K 投影后加 RMSNorm，有助于训练稳定性，允许更大学习率
- **MuonAdamW混合优化器**：矩阵参数用Muon正交化更新（Newton-Schulz迭代），embedding/标量参数用AdamW
- **手写KV Cache**：自回归生成时缓存之前token的K/V，避免重复计算，分为prefill阶段（处理prompt）和decode阶段（逐token生成），复杂度从O(T²)降到O(T)
- **BOS对齐+Best-Fit数据打包**：从BOS token开始最优装箱打包，序列利用率100%无padding浪费
- **Flash Attention 3**：在Hopper架构GPU上自动启用FA3，大幅提升注意力计算效率
- **torch.compile**：模型编译优化，提升训练吞吐
- **分布式训练**：DDP（Distributed Data Parallel）多卡数据并行，梯度累积达到目标total batch size

### 调参经验

- 当 batch size 扩大/缩小 n 倍时，学习率需要扩大/缩小 √n 倍（对于Adam类自适应优化器使用根号缩放规则）
- Weight decay 按 depth 缩放：weight_decay_scaled = weight_decay × (12/depth)²

### Web聊天界面

实现了一个简单的Web网页，可以与模型进行对话交互。

## 📁 项目结构

```
nanochat/
├── nanochat/               # 核心库
│   ├── gpt.py              # GPT 模型实现（Decoder-only Transformer）
│   ├── engine.py           # 推理引擎（含手写KV Cache）
│   ├── tokenizer.py        # BPE 分词器
│   ├── optim.py            # MuonAdamW 优化器
│   ├── dataloader.py       # 数据加载与BOS对齐Best-Fit打包
│   ├── flash_attention.py  # Flash Attention 3 / SDPA 调度
│   ├── checkpoint_manager.py  # checkpoint 保存与恢复
│   ├── loss_eval.py        # 验证集 bpb 评估
│   └── ui.html             # Web 对话界面
├── scripts/
│   ├── base_train.py       # 预训练脚本（支持torchrun分布式启动）
│   ├── chat_sft.py         # SFT 微调脚本
│   ├── chat_web.py         # Web 聊天服务器（FastAPI）
│   ├── chat_cli.py         # 命令行聊天
│   ├── tok_train.py        # 分词器训练
│   └── tok_eval.py         # 分词器评估
├── tasks/                  # 数据集处理
│   └── smoltalk.py         # SmolTalk对话数据集
├── runs/                   # 训练shell脚本
│   └── speedrun.sh         # 一键运行完整流程（8卡）
└── nanochat_data/          # 数据、checkpoint、分词器（大文件不入Git）
```

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

### 8卡分布式预训练（参考 speedrun.sh）

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26 --target-param-data-ratio=8.5 --device-batch-size=16
```

### 8卡分布式SFT

```bash
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag=d26 --device-batch-size=2
```

### Web 对话

```bash
python -m scripts.chat_web
# 在控制台输出的 URL 中打开聊天界面
```

### 命令行聊天

```bash
python -m scripts.chat_cli
```

### 一键运行完整流程

```bash
bash runs/speedrun.sh
# 在8卡节点上自动完成：分词器训练→预训练→评估→SFT→报告生成
```

## 📝 许可证

MIT License
