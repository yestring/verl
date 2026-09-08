# Athena

Athena adds two control loops to VeRL without changing model forward code:

- The online controller observes `RuntimeState` windows and applies only
  parameters registered as `hot`.
- The offline profiler launches isolated VeRL processes for parameters marked
  `restart_required`, aggregates structured telemetry, and writes a tuned YAML.

Parameter lifecycles are defined in `parameters.py`. Batch size, DataLoader,
FSDP/offload, rollout memory allocation, and model-parallel settings are cold.
The actor token budget is hot only when `actor.use_dynamic_bsz=true`, because
VeRL reads `max_token_len_per_gpu` from every actor batch.

## Online controller

Select the Athena trainer and enable dynamic batching:

```yaml
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
    min_token_budget: 1024
    max_token_budget: 32768
```

Telemetry is written to `trace_step_<pid>.jsonl`. Cold recommendations are
written to `restart_requests.jsonl`; the training process never mutates a cold
parameter in place.

## Restart probes

Run deterministic FSDP probes from the same checkpoint:

```bash
python -m verl.athena.offline_auto_tuner \
  --config verl/athena/config_base.yaml \
  --resume-from checkpoints/global_step_100 \
  --restart-requests athena_logs/restart_requests.jsonl \
  --warmup-steps 5 \
  --probe-steps 20 \
  --memory-limit-gb 7 \
  --work-dir athena_tuning
```

Each trial has isolated configuration, stdout/stderr, checkpoint, and telemetry
directories. Failed, timed-out, OOM, invalid-metric, and over-memory trials are
recorded rather than silently selected. `tuning_report.json` contains all trial
results and `best_config.yaml` contains the selected configuration.

The example base config uses `/home/hx/opt-125m`, eager vLLM execution, and a
0.05 vLLM memory fraction so it can be smoke-tested when roughly 9 GiB remains
free on a shared 40 GiB A100. Use a new `--work-dir` for each run so failed old
trials cannot be confused with current results. Megatron is supported by setting
`model_engine: megatron`; on one GPU TP and PP must both remain 1.

Optuna TPE search is optional:

```bash
python -m verl.athena.offline_auto_tuner \
  --config verl/athena/config_base.yaml \
  --sampler optuna --trials 12
```

Without `--resume-from`, every trial starts from the same base configuration and
seed. This is useful before training, but checkpoint-based probes are preferred
for tuning an already-running workload.

## Kernel decision space

Athena also decides how the linear-cross-entropy entropy kernels in
`verl/utils/kernel/kernels.py` are selected and tiled. Every kernel parameter is
`restart_required` (kernels are JIT-compiled and chosen at engine init), so the
offline loop probes them and they take effect on the next restart. When GPU
memory and utilization are both low and fused kernels are disabled, the online
controller's cold recommendation proposes the fused grouped strategy; the
micro-batch increase remains the fallback when fused kernels are already on.

| Parameter (actor-level) | Lifecycle | Backends |
| --- | --- | --- |
| `model.use_fused_kernels` | restart_required | fsdp, megatron |
| `entropy_kernel_backward_method` | restart_required | fsdp, megatron |
| `entropy_kernel_forward_fused` | restart_required | fsdp, megatron |
| `entropy_kernel_vocab_per_split` | restart_required | fsdp, megatron |
| `entropy_kernel_n_group` | restart_required | fsdp, megatron |
| `entropy_kernel_block_size_m` | restart_required | fsdp, megatron |
| `entropy_kernel_block_size_n` | restart_required | fsdp, megatron |
| `entropy_kernel_block_size_k` | restart_required | fsdp, megatron |

Engine-scoped paths are `actor_rollout_ref.actor.fsdp_config.*` under FSDP and
`actor_rollout_ref.actor.megatron.*` under Megatron; the registry rejects
cross-backend overrides. `model.use_fused_kernels` is the backend-agnostic
master switch for the fused linear-cross-entropy path.

`entropy_kernel_backward_method` accepts the canonical strategy names from
`kernels.SUPPORTED_BACKWARD_METHODS`: `total_separate`, `split_dlogits_n`
(default), `split_fused_n`, `split_fused_n_grouped`, and `owner_compute`.
`split_dlogits_m` is declared but never dispatched, and `total_fuse_mn` is
excluded from the tuned space because its fp32 tiles exceed A100 shared
memory at the training reduction (it remains selectable by name on GPUs with
larger shared memory). An invalid name fails at engine init.

The tiling knobs are optional; `null` keeps each strategy family's tuned
default (1024/9504/16384 vocab-per-split for forward/split-N/fused-split-N,
N_GROUP 16 for grouped and 2 for owner-compute, 128/256/32 block sizes per
family). One `entropy_kernel_vocab_per_split` override applies to all three
families; a large split under `split_dlogits_n` materializes a larger d_logits
buffer, so such OOMs are recorded as failed probes rather than silently
selected.

The default offline search spaces (`build_default_fsdp_search_space`,
`build_default_megatron_search_space`) include six kernel candidates: four
backward strategies on the fused baseline, one fused-forward combination, and
one grouped/tiling combination. Optuna search samples `use_fused_kernels`
first, and only then backward method, `forward_fused`, `vocab_per_split`, and
`n_group`. With `use_fused_kernels=true` the engine JIT-compiles every kernel
variant at init so compilation cost stays out of profiled steps.

## Utility and constraints

Weights and constraints can live in the base YAML:

```yaml
trainer:
  athena:
    offline:
      utility_weights:
        throughput: 0.55          # maximize
        step_time: 0.20           # minimize
        peak_memory_gb: 0.10      # minimize
        throughput_cv: 0.15       # minimize instability
      constraints:
        peak_memory_gb: {max: 8}
        gpu_count: {eq: 2}
```

Weights are normalized after unavailable metrics are removed. For example, if
MFU is zero in every trial, `mfu: 0.9` is reported as inactive rather than
silently dominating the score. Supported metrics include throughput, P10
throughput, throughput CV, median/P95 step time, generation/actor time, peak
process memory, total device memory, memory utilization, GPU utilization, MFU,
power, energy per 1K tokens, and throughput per GB.

CLI values override YAML values:

```bash
python -m verl.athena.offline_auto_tuner --config config.yaml \
  --weight throughput=0.5 --weight mfu=0.3 --weight throughput_cv=0.2 \
  --constraint 'peak_memory_gb<=8' --constraint 'gpu_count==2'
```

Completed trials can be re-scored without using a GPU. Athena reloads their raw
telemetry so newly added metrics are also available:

```bash
python -m verl.athena.offline_auto_tuner \
  --rescore-report athena_opt_tuning/tuning_report.json \
  --weight mfu=0.9 --weight throughput=0.1 \
  --constraint 'peak_memory_gb<=8' --constraint 'gpu_count==2'
```

Each probe result also contains `metric_reliability`. Values include `reliable`,
`partial` (for example a single NVML snapshot at step end), `all_zero`,
`unavailable`, `derived`, and `legacy_console_recovered`. Missing metrics are
serialized as JSON `null`, never silently converted to zero. Utility metrics
that are null, all equal, or otherwise carry no cross-trial information are
listed under `inactive_metrics` and their weights are redistributed among the
active metrics.

VeRL currently exposes actor peak allocated/reserved memory and grad norm to the
trainer, so Athena records those directly. The async vLLM path used here does
not expose KV-cache occupancy, TTFT, or TPOT to trainer metrics; those fields are
therefore marked unavailable rather than estimated. End-to-end generation time
remains available as `gen_time`.
