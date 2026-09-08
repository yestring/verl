# Athena × 熵算子优化：项目总结与面试备忘

> 本文档是个人面试经验记录：改造前的状态与问题 → 逐项改进与补救 → 最终效果 → 现有功能 → 启动命令 → 面试要点。

---

## 1. 项目背景

在 verl（LLM 强化学习训练框架）的基础上实现了一套**运行时参数自适应调优系统 Athena**，核心思路是**离线 + 在线双环探测**：

- **离线环（offline loop）**：`verl/athena/offline_auto_tuner.py` 启动隔离的 verl 子进程做 probe，采集结构化遥测（吞吐、step 时间、显存峰值、GPU 利用率、MFU 等），用网格搜索或 Optuna TPE 找最优配置，输出 `tuning_report.json` / `best_config.yaml`。
- **在线环（online loop）**：`verl/athena/controller.py` 在训练过程中观测 `RuntimeState` 窗口（内存、利用率、KL、梯度范数等），只热更新注册为 `HOT` 的参数（如动态 bsz 的 token budget、entropy_coeff、KL 系数、lr、保存频率）；结构性参数（`RESTART_REQUIRED`）写成 `restart_requests.jsonl`，交给离线 tuner 重启探测，训练进程从不原地改冷参数。
- 参数生命周期集中在 `verl/athena/parameters.py` 注册表，`validate_overrides` 按 backend（FSDP / Megatron）做门控。

同时，我在 `verl/utils/kernel/kernels.py` 中重新设计/优化了**交叉熵的 Triton 算子**（前向 mainloop+epilogue、融合前向、以及 7 种 backward 策略：`total_fuse_mn`、`total_separate`、`split_dlogits_n`、`split_dlogits_m`、`split_fused_n`、`split_fused_n_grouped`、`owner_compute`），并把**算子选择与可调参数纳入 Athena 的决策空间**。

生产调用链：`verl/utils/kernel/linear_cross_entropy.py` 的 `torch.autograd.Function`（LinearCrossEntropy）→ `kernels.efficient_entropy_forward/backward` → 模块级 `_config`。FSDP / Megatron 两个 engine 都在 `__init__` 里应用 `_config`。

---

## 2. 改造前的状态与问题清单

### 2.1 算子层面（kernels.py）

| # | 问题 | 现象 / 根因 |
| --- | --- | --- |
| 1 | **fusedforward 性能异常差**（核心问题） | `efficient_entropy_kernel_fused_forward` 只有 **M 方向一维 grid**（M=4096 时仅 32 个 CTA），每个 CTA 串行遍历全部 num_splits × N-block × K-block，hidden tile 每换一个 N-block 就重载一遍 → GPU 严重欠利用 |
| 2 | 标量 logprobs 未初始化 | Sum/Mean reduction 下融合路径对 `torch.empty(())` 做 `atomic_add`，产生垃圾结果 |
| 3 | `_Owner_Compute` 梯度未初始化且 dtype 错 | d_hidden 用 `atomic_add`，但零初始化条件漏了该策略，落入 `torch.empty_like(hidden, dtype=hidden.dtype)` 分支（bf16 下既没清零、类型还不匹配 fp32 累加器） |
| 4 | Grouped kernel label 掩码缺 TP 偏移 | 掩码漏了 `rank * vocab_size`，TP>1 时按错 label 取 logprob |
| 5 | 死代码 | grouped kernel 里 `if USE_TMA: pass` 死分支 |
| 6 | 参数全部硬编码、不可调 | 10 个 kernel 挂着单配置 `@triton.autotune`（形同虚设），block 尺寸 / vocab_per_split / N_GROUP 全部写死在 launch 点，改一个参数要改源码重跑 |
| 7 | 无预热机制 | Triton JIT 编译发生在第一个训练 step 内，污染计时、拉长首 step |
| 8 | benchmark 正确性检测是假的 | 前向对照被 TF32 污染（全策略误报 fwd_correct=false）；后向只查 shape+NaN，`bwd_max_diff` 硬编码 0.0；没有参数 sweep CLI |

### 2.2 Athena 集成层面

- 决策空间只有 batch size、offload、TP/PP、checkpoint 等传统维度，**算子选择和 kernel 调优参数完全不在决策空间里**。
- 配置流只有 `entropy_from_logits_*` 两个老参数，engine config 没有 kernel 字段。
- 离线搜索空间（FSDP 6 个候选 / Megatron 6 个候选）和 Optuna 采样目标里没有 kernel 候选；controller 冷推荐只有 micro-batch 翻倍一种选择。
- 没有覆盖 kernel 配置面的单元测试。

---

## 3. 针对每个问题的改进与补救

### 3.1 A0 + A6：Config 化改造（基础设施，先行）

- 模块级 `Config` dataclass 按 kernel 族分字段，**默认值逐项等于原硬编码值**（不配置 = 行为不变）：
  - forward: `vocab_per_split=1024`、block 128/256/32、stages 3、warps 8
  - split-N 物化族: `vocab_per_split=9504`、dlogits block 128/256/32
  - fused split-N 族: `vocab_per_split=16384`、`grouped_n_group=16`、`owner_compute_n_group=2`、backward block 128/128/32、stages 2
  - MN 族: 128/128/32、stages 3；epilogue 16/64
- 去掉全部 `@triton.autotune`，block 尺寸作为 **constexpr 参数**、stages/warps 作为 **launch 参数**从 `_config` 传入（每套签名只 JIT 一次，走磁盘缓存）。
- 新增 API：`BACKWARD_METHOD_NAMES` 名字↔枚举映射、`get_backward_method_by_name/name`、`SUPPORTED_BACKWARD_METHODS`、`configure(vocab_per_split/n_group/block_size_*)`（校验：`% 128 == 0`、2 的幂、n_group ≥ 1）、`set_backward_method(str|int)`、`set_forward_fused(bool)`（类型校验）。

### 3.2 A1：fused forward 重写为 2D grid（性能异常根因修复）

- grid 从 `(num_pid_m,)` 改为 **`(num_pid_m, num_splits)`**：每个 CTA 只负责一个 (M-tile, split) 对，内部循环 N-block × K-block；数学与 mainloop 完全一致（在线 logsumexp、温度缩放、label 掩码）。
- 部分结果（max/accu/entropy_b）写 `[num_tokens, num_splits]` 缓冲，logprobs 按 label 所在 split 掩码写 `[num_tokens]`（每个 label 恰好属于一个 split，无竞争）；host 端复用现有 epilogue kernel 做最终归约（三种 reduction 都支持）。
- `USE_TMA` 分支按 mainloop 的 tensor descriptor 模式补全。
- **效果**：并行度 8–32 CTA → `num_pid_m × num_splits`（约数千 CTA），hidden 每个 (M-tile, split) 只加载一次。

### 3.3 A2–A5：四个正确性/健壮性修复

- A2: Sum/Mean 下 `torch.empty(())` → `torch.zeros(())`。
- A3: `_Owner_Compute` 加入 fp32 `torch.zeros_like` 分支。
- A4: grouped 掩码补 `rank * vocab_size`。
- A5: 删死 TMA 分支。

### 3.4 A7：预热助手 `warmup_entropy_kernels()`

- `HAVE_TRITON and torch.cuda.is_available()` 门控下，用 dummy 小张量（128 tokens × 1024 hidden × 2048 vocab）把**每个候选 backward 策略 + 两种前向**各跑一遍 forward+backward，JIT 编译发生在 engine init 而非首个 step；`try/finally` 恢复原 `_config`。

### 3.5 A8：benchmark 修复 + sweep CLI

- 前向参考关闭 TF32（`torch.backends.cuda.matmul.fp32_precision="ieee"`，带旧 API 回退）；后向改为真实 `torch.autograd.grad` 对照（reduction=0 语义下 `loss = (dlogprobs·log_probs_ref).sum() + (dentropy·entropy_ref).sum()`），容差做成模块常量。
- 新增 `--backward-methods`（名字或 `all`）/ `--forward-fused` / `--vocab-per-split` / `--n-group` / `--block-size-m|n|k` / `--skip-correctness`，为 Athena 候选列表提供数据。

### 3.6 B1–B2：配置字段 + engine 接线（FSDP + Megatron 双后端）

- `EngineConfig` 基类新增 7 个字段（`entropy_kernel_backward_method`、`forward_fused`、`vocab_per_split`、`n_group`、`block_size_m/n/k`），加入 `_mutable_fields`；`ActorConfig` 与 `dp_actor.yaml` / `megatron_actor.yaml` 同步镜像（`${oc.select:actor_rollout_ref.actor.entropy_kernel_*,<default>}` 模式，与既有 `entropy_from_logits_*` 一致）。
- 两个 engine 的 `__init__` 末尾调用 `_apply_entropy_kernel_config()`：`set_forward_fused` / `set_backward_method`（坏名字 init 即 fail fast）/ `configure`（None 跳过）/ `use_fused_kernels` 时预热。Megatron 的预热放在 `_maybe_enable_fused_kernels` 末尾（只有 patch 完融合路径预热才有意义）。

### 3.7 B3–B5：纳入 Athena 决策空间

- **注册表**：15 个 `RESTART_REQUIRED` 条目（`actor_rollout_ref.model.use_fused_kernels` 总开关 + 7 个 `fsdp_config.entropy_kernel_*` + 7 个 `megatron.entropy_kernel_*`），`validate_overrides` 自动做后端门控。
- **离线搜索空间**：`build_default_fsdp/megatron_search_space` 各追加 6 个 kernel 候选（4 个 backward 策略基线 + fused-forward 组合 + vocab_per_split=16384/n_group=8 组合），JSON 序列化去重；`search_optuna` 先采样 `use_fused_kernels`，为 True 时再采 backward_method（惰性 import `SUPPORTED_BACKWARD_METHODS`）、forward_fused、vocab_per_split [4096–32768]、n_group [2–16]，按 backend 写 `fsdp_config.*` / `megatron.*`。
- **controller 冷推荐**：低内存 + 低利用率 + `use_fused_kernels=False` 时，emit `use_fused_kernels: True + split_fused_n_grouped + forward_fused + vocab_per_split=16384 + n_group=8` 的 RESTART 候选（按 `actor.strategy` 选前缀，once-per-process 防重复）；fused 已开启则回退原 micro-batch 翻倍候选。RESTART 动作只走 `restart_requests.jsonl` → 离线 tuner `--restart-requests`。

### 3.8 验证阶段发现的额外 bug（A6 遗留，已修复）

- **5 个 backward launch 漏传 `GROUP_SIZE_M` constexpr**：去 autotune 后（autotune 原来通过 configs dict 提供该值）任何 backward 调用直接 `TypeError: missing 1 required positional argument: 'GROUP_SIZE_M'`。MN / d_logits / split_N / fused / grouped / owner 六处 launch 全部补齐（各用本族 `*_group_size_m` 默认值）。`GROUP_SIZE_M` 是 flash-attention 式的 **L2 局部性 swizzle 分组大小**，控制 M 方向 CTA 的分组调度。
- **`total_fuse_mn` 在 A100 上 shared memory 超限**：reduction=0（训练路径）变体需要 224KB > A100 上限 163KB（stages 降到 2 仍需 192KB，因为 128×128 的 fp32 d_logits/d_weight tile 本身就要 ~128KB）。处理：从 `SUPPORTED_BACKWARD_METHODS` 排除（避免 Optuna 采到会中途崩的配置，也避免预热只编译 reduction=2 变体造成的假绿灯），保留按名字显式选择的能力（大 shared memory 的 GPU 可用）。
- 顺带修复：`base_config=None` 时 `build_default_fsdp_search_space` 早退会漏掉 kernel 候选（重构为先加 kernel 候选再早退）；一个过期的 `train_batch_size` 测试断言。

### 3.9 TF32 精度问题（按用户决策：保留速度，正确性排到最后）

- 根因定位：0.155 的 logprobs 差异**不是 torch 参考的问题**（torch 侧关闭 TF32 后与 fp64 参考差 1.26e-4），而是 **Triton `tl.dot` 对 f32 输入默认走 TF32 tensor core**（triton 3.6）。这与上游 verl 的实践一致，速度优先，故保留。
- benchmark.py 已具备 TF32-aware 对照能力（`test_correctness` 内部关 TF32），`--skip-correctness` 可跳过；容差校准留作最后一步。

---

## 4. 最终效果

**验证结论（A100-40GB，torch 2.11 + triton 3.6）**

- fused forward 重写后与 mainloop+epilogue 路径 **逐位一致**（max diff = 0.0），且并行度从 ~32 CTA 提升到 `num_pid_m × num_splits`（数千 CTA）——性能异常根因消除。
- 5 个支持的 backward 策略（`total_separate` / `split_dlogits_n` / `split_fused_n` / `split_fused_n_grouped` / `owner_compute`）全部能跑出**有限梯度**（无 NaN/Inf）。
- `warmup_entropy_kernels()` 在 engine init 完成全部 kernel JIT 编译，首 step 不再背编译开销。
- 你此前（去 autotune 之前）的 benchmark 结论：fusedSplitN 已超源码算子、fusedSplitNGroup/OwnerGroup 略逊——**建议修复后重跑一次 `benchmark.py` 全策略回归确认新基线**。
- CPU 单元测试 **28/28 通过**（controller / offline tuner / kernel 配置面）；ruff 全绿；GPU smoke 全绿。

**Athena 决策空间现在的样子**

- 算子层：backward 策略 5 选 1、fused forward 开关、vocab_per_split / n_group / block_size_m/n/k 三个旋钮，全部 `RESTART_REQUIRED`（engine init 时决定 + JIT，天然只能离线探测）。
- 触发路径：低利用率冷推荐（online）→ restart_requests → 离线 probe；离线网格/TPE 直接采 kernel 候选。
- 双后端：FSDP 与 Megatron 都从同一注册表门控，路径前缀不同（`fsdp_config.*` / `megatron.*`）。

---

## 5. 项目现有功能总览

1. **双环自适应调参**：离线 probe 子进程（网格 / Optuna TPE）+ 在线 controller（窗口化、冷却期、once-per-process 冷推荐）。
2. **参数生命周期注册表**：`HOT` / `RESTART_REQUIRED` / `UNSUPPORTED` 三类，按 backend 门控，坏路径直接 `KeyError`。
3. **结构化遥测与可靠性标注**：每个 probe 结果带 `metric_reliability`（reliable / partial / all_zero / unavailable / derived / legacy_console_recovered），缺失指标序列化为 `null` 而不是悄悄置 0；无信息量的指标进 `inactive_metrics` 并把权重重新分配。
4. **效用打分 + 约束**：`utility_weights`（吞吐、step 时间、显存、吞吐 CV、MFU…）+ `constraints`（如 `peak_memory_gb<=8`），失败/OOM/超时 trial 记录而非静默选中；已完成的 trial 可以**无 GPU 重新打分**（`--rescore-report`）。
5. **kernel 决策空间**：算子选择 + 调参纳入离线环（见上）；engine init 应用 + 预热。
6. **测试**：`tests/athena/`（controller / offline tuner / utility）+ `tests/utils/kernel/test_kernels_config.py`（CPU 配置面）+ `benchmark.py` / `benchmark_fwd.py`（GPU 数值与性能）。

---

## 6. 启动 / 使用命令

环境说明：`uv run` 在当前机器上不可用（pyext 在 py3.12 编译失败），统一用 conda 环境：

```bash
export PYTHONPATH=/home/hx/verl
PY=/home/hx/.conda/envs/verl_py311/bin/python   # torch 2.11 + triton 3.6
```

### 6.1 在线 controller（训练中启用）

```yaml
# 训练配置里：
actor_rollout_ref:
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 4096
trainer:
  v1:
    trainer_mode: athena_trainer
  athena_log_dir: ./athena_logs
  athena:
    enabled: true
    window_size: 5
    cooldown_steps: 5
    low_memory_threshold: 0.70
    high_memory_threshold: 0.90
    low_gpu_utilization_threshold: 0.55
```

冷推荐写入 `athena_logs/restart_requests.jsonl`。

### 6.2 离线 tuner（网格搜索 / 重启探测）

```bash
$PY -m verl.athena.offline_auto_tuner \
  --config verl/athena/config_base.yaml \
  --resume-from checkpoints/global_step_100 \
  --restart-requests athena_logs/restart_requests.jsonl \
  --warmup-steps 5 --probe-steps 20 \
  --memory-limit-gb 7 \
  --work-dir athena_tuning
```

Optuna TPE 搜索：

```bash
$PY -m verl.athena.offline_auto_tuner \
  --config verl/athena/config_base.yaml \
  --sampler optuna --trials 12
```

无 GPU 重新打分：

```bash
$PY -m verl.athena.offline_auto_tuner \
  --rescore-report athena_opt_tuning/tuning_report.json \
  --weight mfu=0.9 --weight throughput=0.1 \
  --constraint 'peak_memory_gb<=8' --constraint 'gpu_count==2'
```

Megatron 后端：`config_base.yaml` 里 `model_engine: megatron`（单卡 TP=PP=1）。输出：`tuning_report.json`（全量 trial）+ `best_config.yaml`（最优配置）。

### 6.3 kernel benchmark / 验证

```bash
cd verl/utils/kernel

# 全策略性能回归（小/中 workload）
$PY benchmark.py --workloads small,medium --iters 20 --warmup 5

# 只看速度、跳过正确性（TF32 保留）
$PY benchmark.py --backward-methods all --skip-correctness

# 参数 sweep（为 Athena 候选提供数据）
$PY benchmark.py --backward-methods split_fused_n,split_fused_n_grouped \
  --vocab-per-split 16384 --n-group 8 --block-size-m 128

# 融合前向 vs mainloop 基线
$PY benchmark_fwd.py --num-tokens 4096 --hidden-size 4096 --vocab-size 128000
```

### 6.4 测试 / 静态检查

```bash
# CPU 封闭测试
$PY -m pytest tests/athena tests/utils/kernel/test_kernels_config.py -q

# 静态检查
$PY -m ruff check verl/athena/ verl/utils/kernel/kernels.py tests/athena/
```

---

## 7. 面试要点速记

**Triton / CUDA 层面**
- **grid 分解决定并行度**：一维 M-only grid + 串行遍历 split 是 fusedforward 性能崩盘的根因；改成 `(num_pid_m, num_splits)` 后 CTAs 数量 ×100+，且每 CTA 只加载一次 hidden tile。答性能优化题先问"现在 grid 多大、每个 CTA 干什么"。
- **在线 logsumexp**：数值稳定的 softmax 规约（避免 exp 上溢）；max/accu 部分结果 + 最后 epilogue 归约是 split 设计的核心。
- **constexpr vs 运行时参数 vs launch 参数**：block 尺寸走 constexpr（编译期展开，改变即重编译），stages/warps 走 launch kwargs，`GROUP_SIZE_M` 是 L2 局部性 swizzle 分组。
- **TF32 语义**：Triton `tl.dot` 对 f32 输入默认 TF32（tensor core 加速），对照精确 fp32 参考会差 ~1e-1 量级——这不是 bug，是精度/速度权衡；torch 侧可用 `torch.backends.cuda.matmul.fp32_precision="ieee"` 关闭。
- **shared memory 上限**：A100 163KB；`total_fuse_mn` 的 128×128 fp32 tile 流水后 224KB 直接超限——全融合策略的代价就是 smem/寄存器压力，所以被移出支持列表（只在更大 smem 的 GPU 上可显式启用）。
- **TMA**：`tl.make_tensor_descriptor` 需要 stride==1 的连续性，不满足时回退普通指针加载。

**系统设计层面**
- **参数生命周期**：什么参数能热更新取决于"worker 每次 batch 读不读它"——token budget 在 dynamic bsz 下每 batch 都读所以 HOT；改 data loader / process group / 模型图 / kernel JIT 的都是 RESTART。kernel 参数必然 RESTART，因为 Triton 在 engine init 决定签名并编译。
- **探测与选择分离**：probe 子进程隔离（OOM/超时/失败都记录、不静默选中）；在线环只发出冷推荐，重启验证交给离线环——避免训练进程自毁。
- **度量可靠性**：缺失指标记 `null` 而非 0（0 会污染加权打分）；权重归一化 + inactive 指标剔除。

**遗留待办（最后一步）**：TF32-aware 容差校准 + backward 对 `autograd.grad` 参考的数值对照（benchmark.py 已就位）；修复后重跑全策略 benchmark 确认新基线。
