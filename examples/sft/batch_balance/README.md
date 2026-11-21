# 四卡变长 SFT 批次均衡实验

在同一个 Ray SFT、CUDA FSDP2、SP=1 训练入口比较两种分配。每个优化器 step 的全局样本、初始模型、有效 loss token 归一化和 DP rank 样本数相同。两组共用逐 microbatch 的 token 硬上限。达到上限后开启新组，跨 DP 同步组数，必要时拆分已有组。样本和标签按同一索引排列。

- Baseline 使用 verl 的 `24576 × L + L²` 作为样本代价。
- Calibrated 保留同一全局分配算法，用真实 packed microbatch 的前反向 CUDA event 耗时拟合代价。拟合采用非负系数并留出后段训练 step，常数项只用于预测 microbatch 耗时，按样本分配使用两项长度特征。

CUDA event 记录当前 stream 的区间，可能包含对其他 stream 与 collective 的等待。独立校准阶段打开逐 microbatch 计时，正式 A/B 使用逐步 wall time 和峰值显存，排除预热步。token 硬上限约束装箱，实际显存由四张卡的峰值测量。

## 运行

在已有 verl CUDA 依赖的四卡环境，从 checkout 根目录启动。需要可用的 Ray、PyTorch FSDP2、TensorDict、Transformer 和 FlashAttention。本地模型目录需要包含可直接加载的 HF 权重和 tokenizer。

```bash
bash examples/sft/batch_balance/run_4gpu.sh /workspace/models/Qwen3-1.7B-Base
```

脚本依次完成混合长度校准、拟合、混合长度 A/B、窄长度 A/B。默认全局 batch 为 16，单 microbatch token 上限和最大样本长度均为 16,384。启动参数可用 `PYTHON`、`STEPS`、`PROFILE_STEPS`、`WARMUP_STEPS`、`GLOBAL_BATCH`、`CONTEXT_LEN`、`TOKEN_CAP` 覆盖。`GLOBAL_BATCH` 需能被 4 整除。

默认 `DATALOADER_WORKERS=0`，在训练控制进程中同步加载，适用于 `/dev/shm` 较小的容器。所有实验组沿用同一设置。逐步计时从取出 batch 后开始。容器有足够共享内存时，可用 `DATALOADER_WORKERS` 选择并行加载数。

默认数据是固定长度分布的合成 token 样本，用于系统性能测量。设置 `MIXED_DATA` 和 `NARROW_DATA` 可使用各自的 SFT parquet，数据须有足够样本且经过模板后的序列长度在 token 上限内。四个训练对照都从同一个 HF 模型开始，checkpoint 路径分开。

目录中保留各组 `steps.jsonl` 和 `run.log`、`cost_model.json` 以及 `report.md`。汇总要求同一负载的两组逐步输入哈希、输入 token 数和监督 token 数完全一致，并检查 loss 和梯度范数。报告包括有效监督 token/s、step P50/P95/DP rank 时差与各卡峰值显存。开始扩展到更长序列时，分别核对显存、拟合误差和端到端吞吐。

## 四卡 FSDP2 通信对照

```bash
bash examples/sft/batch_balance/run_comm_4gpu.sh \
  /workspace/models/Qwen3-0.6B /workspace/verl-sft-comm

bash examples/sft/batch_balance/run_comm_trace.sh \
  /workspace/verl-sft-comm /workspace/verl-sft-comm-trace
```

混合长度先做 2×2，对照梯度同步频率和 backward 后的参数重分片。窄长度对照 baseline 与 combined，检查只有一轮 microbatch 时的行为。全部实验保留原版 workload 估算、相同样本和 token 上限。报告逐步核对样本哈希、每个 rank 的样本索引和 microbatch 数。

| 策略 | 梯度同步 | Backward 后重分片 |
| --- | --- | --- |
| baseline | 每个 microbatch | 每个 microbatch |
| defer-grad | 最后一轮 | 每个 microbatch |
| retain-params | 每个 microbatch | 最后一轮 |
| combined | 最后一轮 | 最后一轮 |

延迟梯度同步减少的是梯度 reduce-scatter 轮次。延后 backward 后重分片可复用已聚合的参数，减少下一轮 forward 的 all-gather。forward 后重分片沿用 baseline。两项都可能增加峰值显存，最后一次 backward 恢复常规设置，再执行 optimizer step。`report.md` 汇总逻辑轮次、吞吐、峰值显存、首步 loss 和 grad norm，并给出单项与组合耗时差。设置 `REPEAT_REVERSED=1` 可反向复跑混合组，检查运行顺序的影响。

`run_comm_trace.sh` 另行回放四个混合组，只采 rank 0 的同一个多轮 microbatch step。它核对该 step 的样本指纹和分配等字段。`collectives.md` 分列 CPU 侧 collective 调用事件和 GPU 侧 NCCL kernel 统计。GPU kernel 的时间之和包含可能的重叠，暴露在计算之外的通信耗时需要在时间线上进一步判读。带 profiler 的运行独立于吞吐表。

### 四卡显存和通信时间线

对已完成的某一档 token 上限回放四个混合长度组。`--diagnostics` 复用原实验的模型、数据量和训练总步数，在预热后的多轮 step 中选最常见的 microbatch 数，取满足条件的第一步，并在四个 rank 同时采集这个 step。

```bash
bash examples/sft/batch_balance/run_comm_trace.sh --diagnostics \
  /workspace/verl-sft-comm-scaling/cap-16384 /workspace/verl-sft-comm-diag
```

- `memory.md` 对齐各策略的 rank、microbatch 和采样边界，记录 CUDA allocator 的当前 allocated、reserved 与步内累计 allocated 峰值。关注第一轮 backward 后是否仍有额外活跃分配，以及最后一轮 backward 后的变化。
- `overlap.md` 按 rank 对 CUDA kernel 时间区间取并集，计算 NCCL 与非 NCCL kernel 同时可见的时间，以及只有 NCCL kernel 可见的时间。它描述 GPU 时间线，不能借此推算训练关键路径或替代无 profiler 的步耗时。
- `collectives.md` 保留各 rank 的 collective 计数和 NCCL kernel 时长之和。比较无 profiler 的吞吐或端到端峰值时，使用原实验目录的 `report.md`。
