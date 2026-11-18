# Athena 固定工作负载搜索 V1

这个入口复用 Athena 的独立 probe 子进程。在同一个模型、数据、采样策略和 GPU 配置下，比较随机搜索、TPE 和先短测后晋级的分级搜索。现有 `offline_auto_tuner` CLI 仍可运行。

## 在 GPU 容器里准备

```bash
git clone --branch athena-search-v1 https://github.com/yestring/verl.git
cd verl
python -m verl.athena.fixed_workload_search --help
```

使用已能跑通 VeRL 和 Athena 的 GPU 环境，搜索前端需要 Optuna。选择一个已验证的共卡 FSDP 或 Megatron YAML，并启用 `trainer.use_v1`、`trainer.v1.trainer_mode: athena_trainer` 和 `actor_rollout_ref.actor.use_dynamic_bsz`。把模型、有效 batch、prompt/response 长度、`rollout.n`、并行拓扑、GPU 数和数据固定在 base YAML。若从 checkpoint 恢复，所有搜索使用同一个 `--resume-from /path/to/global_step_N`，并保留 `trainer.del_local_ckpt_after_load: false`。

```bash
BASE=/path/to/working-athena.yaml
MEMORY_LIMIT_GIB=30
TOKEN_BUDGETS=(2048 4096 6144 8192 10240 12288)
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/athena-v1.XXXXXX")
echo "$RUN_ROOT"

# 先确认模型、数据、现有 Athena 遥测和显存限制都能跑通。
python -m verl.athena.fixed_workload_search \
  --config "$BASE" --work-dir "$RUN_ROOT/smoke" --sampler random \
  --token-budgets "${TOKEN_BUDGETS[0]}" --memory-limit-gib "$MEMORY_LIMIT_GIB" \
  --trials 1 --warmup-steps 1 --probe-steps 3
```

候选 token budget 和显存值是示例。根据已通过的基线与有效样本长度选取候选。显存约束来自 Athena 的 actor allocated 峰值，包括预热步。新目录中的 `trial_*/stdout.log`、`stderr.log` 和 `telemetry/` 可用于排查失败。

## 比较三种搜索

```bash
common=(--config "$BASE" --token-budgets "${TOKEN_BUDGETS[@]}"
        --memory-limit-gib "$MEMORY_LIMIT_GIB" --seed 42
        --trials 9 --warmup-steps 3 --probe-steps 15)

for sampler in random tpe; do
  python -m verl.athena.fixed_workload_search "${common[@]}" \
    --work-dir "$RUN_ROOT/$sampler" --sampler "$sampler"
done

python -m verl.athena.fixed_workload_search "${common[@]}" \
  --work-dir "$RUN_ROOT/staged" --sampler staged \
  --screen-steps 4 --promote-top-k 3

python -m verl.athena.compare_search_reports \
  "$RUN_ROOT/random/fixed_workload_search.json" \
  "$RUN_ROOT/tpe/fixed_workload_search.json" \
  "$RUN_ROOT/staged/fixed_workload_search.json"
```

- **Random 和 TPE** 每次都跑完整 probe，按相同的吞吐目标选可行结果。
- **Staged** 从有效的条件空间做不重复抽样，用短 probe 排序，再完整复测领先的可行配置。短测分数用于晋级，最终配置和主结果曲线由完整 probe 产生。`--trials` 是包括晋级复测在内的 probe 次数上限。候选空间比预算小时可以少跑。
- 短测和完整测量的学习率计划需要一致。Staged 要求使用固定预热的 constant 学习率计划，或显式指定 Megatron 的衰减总步数。
- 初版只搜 token budget。可给三种策略都加 `--sweep-checkpointing` 扩大 FSDP 的搜索空间。已有融合路径且已在目标 GPU 做过正确性检查时，还可给三种策略加相同的 `--kernel-backward` 选项。Megatron 重算沿用 base YAML。
- 先尝试不同长度的短测，核对它能覆盖真实长尾并选出接近完整评测最优的候选。GPU 秒相同是最终对照口径，完整 probe 次数相同会掩盖分级搜索节省的测量成本。

## 固定轨迹与结果口径

默认固定工作负载配置，在线 rollout 仍可能生成不同的长度和内容。如果主要比较训练侧系统参数，可按 [Skip Manager](advance/skip_manager.rst) 先录制一份 V1 rollout，并在三组都启用 `skip.rollout_tq`、`action: cache` 及 `--require-cached-rollouts`。项目名、实验名、dump 路径和所覆盖步号需要一致。从 checkpoint 的第 N 步恢复时，缓存须从 N+1 开始覆盖所有预热和正式测量步。只使用同一配置和 checkpoint 的可信录制。搜索和报告对比都核对重叠步骤的处理 token 数。

报告的目标是 `processed_tokens_per_gpu_s`，计入 prompt 和 response，分母是正式测量的 step 秒数乘 GPU 数。用于比较搜索成本的 `cumulative_gpu_s` 则由每次 probe 的墙钟时间乘配置 GPU 数累计，包含启动、预热和失败。它是按 GPU 配额估算的成本。对比脚本默认取所有报告都到达的最大共同预算，也支持 `--budget-gpu-s` 和 `--target`。

保留每个 sampler 的 `fixed_workload_search.json` 和 `best_config.yaml`。在留出轨迹上复测最终候选，先核对处理 token 数、峰值显存和 P95 step，再评估端到端收益。
