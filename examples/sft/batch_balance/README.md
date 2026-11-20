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

默认数据是固定长度分布的合成 token 样本，用于系统性能测量。设置 `MIXED_DATA` 和 `NARROW_DATA` 可使用各自的 SFT parquet，数据须有足够样本且经过模板后的序列长度在 token 上限内。四个训练对照都从同一个 HF 模型开始，checkpoint 路径分开。

目录中保留各组 `steps.jsonl` 和 `run.log`、`cost_model.json` 以及 `report.md`。汇总要求同一负载的两组逐步输入哈希、输入 token 数和监督 token 数完全一致，并检查 loss 和梯度范数。报告包括有效监督 token/s、step P50/P95/DP rank 时差与各卡峰值显存。开始扩展到更长序列时，分别核对显存、拟合误差和端到端吞吐。
