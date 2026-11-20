#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: %s COMPLETED_BENCHMARK_DIRECTORY [NEW_TRACE_DIRECTORY]\n' "$0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench=$(cd -- "$1" && pwd)
python_bin=${PYTHON:-python}
test -f "$bench/report.md"

config_value() {
    sed -n "s/^$1 //p" "$bench/config.txt"
}

model=$(config_value Model)
dataset_steps=$(config_value 'Dataset steps')
dataset_steps=${dataset_steps:-$(config_value Steps)}
warmup=$(config_value Warmup)
export CUDA_VISIBLE_DEVICES=$(config_value CUDA_VISIBLE_DEVICES)
export GLOBAL_BATCH=$(config_value 'Global batch')
export CONTEXT_LEN=$(config_value 'Context length')
export TOKEN_CAP=$(config_value 'Token cap')
export DATALOADER_WORKERS=$(config_value 'DataLoader workers')
export MIXED_DATA=$(config_value 'Mixed data')
if [[ $MIXED_DATA == synthetic ]]; then
    unset MIXED_DATA
fi
unset NARROW_DATA

trace_step=$("$python_bin" - "$bench/mixed-baseline/steps.jsonl" "$warmup" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open() as stream:
    for row in map(json.loads, stream):
        if row["step"] > int(sys.argv[2]) and min(row["microbatch_count"]) > 1:
            print(row["step"])
            break
    else:
        raise SystemExit("No measured mixed step with multiple microbatches")
PY
)

if [[ $# -eq 2 ]]; then
    mkdir -- "$2"
    out=$(cd -- "$2" && pwd)
else
    out=$(mktemp -d "${TMPDIR:-/tmp}/verl-sft-comm-trace.XXXXXX")
fi
printf 'Trace directory %s\nProfiled step %s on rank 0\n' "$out" "$trace_step"

traces=()
for arm in mixed-baseline mixed-defer-grad mixed-retain-params mixed-combined; do
    run="$out/replay-$arm"
    ARM_ONLY="$arm" STEPS="$trace_step" DATASET_STEPS="$dataset_steps" WARMUP_STEPS=0 \
        bash "$script_dir/run_comm_4gpu.sh" "$model" "$run" \
        "trainer.profile_interval=[$trace_step,$trace_step]" \
        profiler.enable=true profiler.tool=torch 'profiler.ranks=[0]' \
        'profiler.tool_config.torch.contents=[cpu,cuda]' \
        "profiler.save_path=\"$run/trace\""

    "$python_bin" - "$bench/$arm/steps.jsonl" "$run/$arm/steps.jsonl" "$trace_step" <<'PY'
import json
import sys
from pathlib import Path

reference, replay = map(Path, sys.argv[1:3])
step = int(sys.argv[3])

def selected(path):
    with path.open() as stream:
        for row in map(json.loads, stream):
            if row["step"] == step:
                return row
    raise SystemExit(f"{path}: missing step {step}")

expected, actual = selected(reference), selected(replay)
for key in ("batch_sha256", "rank_sample_indices", "microbatch_count", "grad_sync_rounds", "reshard_rounds"):
    if actual[key] != expected[key]:
        raise SystemExit(f"{replay}: profiled step differs from measured step in {key}")
PY
    traces+=("$run/trace")
done

"$python_bin" "$script_dir/summarize_comm_trace.py" "${traces[@]}" | tee "$out/collectives.md"
printf '\nCollective report %s/collectives.md\n' "$out"
