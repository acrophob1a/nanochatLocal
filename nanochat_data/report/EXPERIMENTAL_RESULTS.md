# nanochat Experimental Results

## Overview

This document summarizes all experimental results from the nanochat project, including base model training, fine-tuning, tokenizer development, and evaluation benchmarks.

---

## 1. Base Model Training

**Timestamp**: 2026-02-05 23:40:49  
**Model**: d26 (depth=26, aspect_ratio=64, head_dim=128)

### Configuration

| Parameter | Value |
|-----------|-------|
| Number of Parameters | 1.68B |
| Max Sequence Length | 2048 |
| Window Pattern | SSSL |
| Total Batch Size | 524,288 tokens |
| Training Tokens | 7.8T |
| DDP World Size | 8 GPUs |

### Optimization

| Parameter | Value |
|-----------|-------|
| Matrix LR | 0.0200 |
| Embedding LR | 0.3000 |
| Unembedding LR | 0.0040 |
| Weight Decay | 0.2000 |
| Adam Beta1 | 0.8000 |
| Adam Beta2 | 0.9500 |

### Results

| Metric | Value |
|--------|-------|
| Minimum Validation bpb | 0.7433 |
| Final Validation bpb | 0.7433 |
| CORE Metric Estimate | 0.2680 |
| MFU (Model Flops Utilization) | 45.86% |
| Total Training FLOPs | 4.83e+19 |
| Total Training Time | 701.81 minutes |
| Peak Memory Usage | 74,375 MiB |

---

## 2. Base Model Evaluation

**Timestamp**: 2026-02-05 23:57:45  
**Model**: base_model (step 169150)

### Core Metrics

| Metric | Value |
|--------|-------|
| CORE | 0.3405 |
| Train bpb | 0.6768 |
| Validation bpb | 0.6759 |

### Benchmark Results

| Benchmark | Score |
|-----------|-------|
| hellaswag_zeroshot | 0.5030 |
| hellaswag | 0.5172 |
| arc_easy | 0.6448 |
| arc_challenge | 0.2844 |
| squad | 0.4393 |
| coqa | 0.3202 |
| lambada_openai | 0.5474 |
| piqa | 0.5147 |
| winograd | 0.5824 |
| boolq | -0.0108 |
| copa | 0.4200 |
| commonsense_qa | 0.1380 |
| openbook_qa | 0.2213 |
| winogrande | 0.2423 |

---

## 3. SFT Fine-Tuning (Chinese)

**Timestamp**: 2026-06-23 09:10:23  
**Model**: d26 (step 747)

### Configuration

| Parameter | Value |
|-----------|-------|
| Num Epochs | 3 |
| Chinese Only | True |
| Max Seq Length | 2048 |
| Device Batch Size | 2 |
| Total Batch Size | 32,768 |
| dtype | bfloat16 |
| DDP World Size | 1 |

### Results

| Metric | Value |
|--------|-------|
| Number of Iterations | 341 |
| Minimum Validation bpb | 0.4215 |

---

## 4. Tokenizer Training

**Timestamp**: 2026-02-07 02:36:48

### Configuration

| Parameter | Value |
|-----------|-------|
| Max Chars | 2B |
| Doc Cap | 10,000 |
| Vocab Size | 32,768 |
| Num Special Tokens | 9 |
| Training Time | 73.82 seconds |

### Statistics

| Metric | Value |
|--------|-------|
| Token Bytes Min | 1 |
| Token Bytes Max | 19 |
| Token Bytes Mean | 6.60 |
| Token Bytes Std | 2.82 |

---

## 5. Tokenizer Evaluation

**Timestamp**: 2026-02-05 11:15:20

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | Our Tokens | Relative Diff |
|-----------|-------|--------------|------------|---------------|
| news | 1819 | 404 | 403 | +0.2% |
| korean | 893 | 745 | 797 | -7.0% |
| code | 1259 | 576 | 620 | -7.6% |
| math | 1834 | 936 | 1025 | -9.5% |
| science | 1112 | 260 | 258 | +0.8% |
| fwe-train | 4,208,518 | 900,364 | 892,476 | +0.9% |
| fwe-val | 4,768,657 | 1,027,270 | 1,023,546 | +0.4% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | Our Tokens | Relative Diff |
|-----------|-------|--------------|------------|---------------|
| news | 1819 | 387 | 403 | -4.1% |
| korean | 893 | 364 | 797 | -119.0% |
| code | 1259 | 309 | 620 | -100.6% |
| math | 1834 | 832 | 1025 | -23.2% |
| science | 1112 | 249 | 258 | -3.6% |
| fwe-train | 4,208,518 | 874,799 | 892,476 | -2.0% |
| fwe-val | 4,768,657 | 1,001,442 | 1,023,546 | -2.2% |

---

## 6. Hardware Environment

- **Platform**: Linux
- **CPUs**: 128 cores (255 logical)
- **Memory**: 2015.6 GB
- **GPUs**: 8x NVIDIA A100-SXM4-80GB
- **GPU Memory**: 634.6 GB total
- **CUDA Version**: 12.8

---

## 7. Key Takeaways

1. **Base Model**: Successfully trained a 1.68B parameter GPT model with 45.86% MFU, achieving CORE metric of 0.3405.

2. **Chinese SFT**: Fine-tuned on Chinese conversation data, reducing validation bpb from 0.7433 to 0.4215.

3. **Tokenizer**: Custom BPE tokenizer with 32K vocab size shows comparable performance to GPT-2 on news and science text, with room for improvement on code and math.

4. **Efficiency**: The training pipeline demonstrates efficient multi-GPU utilization with DDP across 8 A100 GPUs.