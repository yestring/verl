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
  --memory-limit-gb 38 \
  --work-dir athena_tuning
```

Each trial has isolated configuration, stdout/stderr, checkpoint, and telemetry
directories. Failed, timed-out, OOM, invalid-metric, and over-memory trials are
recorded rather than silently selected. `tuning_report.json` contains all trial
results and `best_config.yaml` contains the selected configuration.

Optuna TPE search is optional:

```bash
python -m verl.athena.offline_auto_tuner \
  --config verl/athena/config_base.yaml \
  --sampler optuna --trials 12
```

Without `--resume-from`, every trial starts from the same base configuration and
seed. This is useful before training, but checkpoint-based probes are preferred
for tuning an already-running workload.
