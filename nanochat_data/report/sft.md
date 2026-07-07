## SFT
timestamp: 2026-06-23 09:10:23

- run: dummy
- device_type: 
- dtype: bfloat16
- resume_from: sft
- model_tag: d26
- model_step: 747
- num_iterations: -1
- optimizer_steps: None
- num_epochs: 3
- chinese_only: True
- chinese_repeats: 1
- max_seq_len: 2048
- device_batch_size: 2
- total_batch_size: 32,768
- embedding_lr: 0.3000
- unembedding_lr: 0.0040
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.1000
- eval_every: 50
- eval_tokens: 131,072
- dry_run: False
- Number of iterations: 341
- DDP world size: 1
- Minimum validation bpb: 0.4215

