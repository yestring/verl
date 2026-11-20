#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: %s MODEL_DIRECTORY [NEW_OUTPUT_DIRECTORY [HYDRA_OVERRIDES...]]\n' "$0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../../.." && pwd)
python_bin=${PYTHON:-python}
model=$(cd -- "$1" && pwd)
test -f "$model/config.json"
shift
requested_out=${1:-}
if [[ $# -gt 0 ]]; then
    shift
fi

steps=${STEPS:-20}
dataset_steps=${DATASET_STEPS:-$steps}
warmup=${WARMUP_STEPS:-3}
global_batch=${GLOBAL_BATCH:-16}
context=${CONTEXT_LEN:-16384}
cap=${TOKEN_CAP:-16384}
dataloader_workers=${DATALOADER_WORKERS:-0}
repeat_reversed=${REPEAT_REVERSED:-0}
arm_only=${ARM_ONLY:-}

for value in "$steps" "$dataset_steps" "$global_batch" "$context" "$cap"; do
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
        printf 'STEPS, DATASET_STEPS, GLOBAL_BATCH, CONTEXT_LEN and TOKEN_CAP must be positive integers\n' >&2
        exit 2
    fi
done
if (( dataset_steps < steps )); then
    printf 'DATASET_STEPS must be at least STEPS\n' >&2
    exit 2
fi
if [[ ! $warmup =~ ^(0|[1-9][0-9]*)$ ]] || (( steps <= warmup )); then
    printf 'Need a nonnegative warmup and at least one measured step\n' >&2
    exit 2
fi
if [[ ! $dataloader_workers =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'DATALOADER_WORKERS must be a nonnegative integer\n' >&2
    exit 2
fi
if (( global_batch % 4 != 0 || context < 2 || cap < context )); then
    printf 'GLOBAL_BATCH must divide among four ranks and 2 <= CONTEXT_LEN <= TOKEN_CAP\n' >&2
    exit 2
fi
if [[ $repeat_reversed != 0 && $repeat_reversed != 1 ]]; then
    printf 'REPEAT_REVERSED must be 0 or 1\n' >&2
    exit 2
fi
if [[ -n $arm_only ]]; then
    if [[ ! $arm_only =~ ^(mixed|narrow)-(baseline|defer-grad|retain-params|combined)$ ]]; then
        printf 'ARM_ONLY must name one workload and policy, for example mixed-combined\n' >&2
        exit 2
    fi
elif (( $# > 0 )); then
    printf 'Extra Hydra overrides require ARM_ONLY, for a separate single-arm replay\n' >&2
    exit 2
fi

mixed_data=${MIXED_DATA:-}
narrow_data=${NARROW_DATA:-}
if [[ -n $mixed_data ]]; then
    test -f "$mixed_data"
    mixed_data=$(cd -- "$(dirname -- "$mixed_data")" && pwd)/$(basename -- "$mixed_data")
fi
if [[ -n $narrow_data ]]; then
    test -f "$narrow_data"
    narrow_data=$(cd -- "$(dirname -- "$narrow_data")" && pwd)/$(basename -- "$narrow_data")
fi
if [[ -n $requested_out ]]; then
    mkdir -- "$requested_out"
    out=$(cd -- "$requested_out" && pwd)
else
    out=$(mktemp -d "${TMPDIR:-/tmp}/verl-sft-comm.XXXXXX")
fi

sample_count=$(( dataset_steps * global_batch ))
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="$script_dir:$repo${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$repo"
printf 'Results directory %s\n' "$out"
printf 'Model %s\nCUDA_VISIBLE_DEVICES %s\nSteps %s\nDataset steps %s\nWarmup %s\nGlobal batch %s\nContext length %s\nToken cap %s\nDataLoader workers %s\nRepeat reversed %s\nMixed data %s\nNarrow data %s\nSelected arm %s\n' \
    "$model" "$CUDA_VISIBLE_DEVICES" "$steps" "$dataset_steps" "$warmup" "$global_batch" "$context" "$cap" \
    "$dataloader_workers" "$repeat_reversed" "${mixed_data:-synthetic}" "${narrow_data:-synthetic}" \
    "${arm_only:-all}" > "$out/config.txt"

run_arm() {
    local workload=$1 policy=$2 suffix=$3
    shift 3
    local name="$workload-$policy$suffix"
    local arm_dir="$out/$name"
    local parquet=$mixed_data defer=false retain=false
    local dataset
    if [[ $workload == narrow ]]; then
        parquet=$narrow_data
    fi
    if [[ -n $parquet ]]; then
        dataset=("data.train_files=\"$parquet\"")
    else
        dataset=(
            "data.train_files=$workload"
            "data.train_max_samples=$sample_count"
            'data.custom_cls.path=pkg://synthetic_dataset'
            'data.custom_cls.name=SyntheticDataset'
        )
    fi
    case "$policy" in
        defer-grad) defer=true ;;
        retain-params) retain=true ;;
        combined) defer=true; retain=true ;;
    esac

    mkdir -- "$arm_dir"
    printf 'Running %s\n' "$name"
    "$python_bin" -m verl.trainer.sft_trainer_ray \
        engine=fsdp engine.strategy=fsdp2 engine.ulysses_sequence_parallel_size=1 \
        engine.use_torch_compile=false engine.seed=42 engine.reshard_after_forward=true \
        "model.path=\"$model\"" model.use_remove_padding=true \
        data.use_dynamic_bsz=true data.pad_mode=no_padding \
        "data.train_batch_size=$global_batch" "data.max_length=$context" \
        "data.max_token_len_per_gpu=$cap" "data.num_workers=$dataloader_workers" data.val_files=null \
        trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.seed=42 \
        trainer.total_epochs=1 "trainer.total_training_steps=$steps" \
        trainer.balance_batch=true trainer.resume_mode=disable trainer.save_freq=-1 \
        'trainer.logger=[console]' 'checkpoint.save_contents=[]' \
        "trainer.default_local_dir=\"$arm_dir/checkpoints\"" \
        "hydra.run.dir=\"$arm_dir/hydra\"" \
        "trainer.sft_batch_balance_metrics_path=\"$arm_dir/steps.jsonl\"" \
        trainer.sft_batch_balance_profile=false trainer.sft_batch_balance_cost_model=null \
        "trainer.sft_defer_grad_sync=$defer" "trainer.sft_reuse_unsharded_params=$retain" \
        "${dataset[@]}" "$@" 2>&1 | tee "$arm_dir/run.log"
}

if [[ -n $arm_only ]]; then
    run_arm "${arm_only%%-*}" "${arm_only#*-}" "" "$@"
    exit 0
fi

for policy in baseline defer-grad retain-params combined; do
    run_arm mixed "$policy" ""
done
run_arm narrow baseline ""
run_arm narrow combined ""
if [[ $repeat_reversed == 1 ]]; then
    for policy in combined retain-params defer-grad baseline; do
        run_arm mixed "$policy" -r2
    done
fi

"$python_bin" "$script_dir/compare_comm.py" "$out" --warmup-steps "$warmup" \
    --expected-steps "$steps" --repeat-reversed "$repeat_reversed" | tee "$out/report.md"
printf '\nReport %s/report.md\n' "$out"
