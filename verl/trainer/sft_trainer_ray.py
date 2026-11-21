# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import ray
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.logger import log_with_rank
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions
from verl.utils.sft_batch_balance import (
    MICROBATCH_POLICIES,
    load_cost_model,
    microbatch_plan_proxy,
    plan_rank_microbatches,
)
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self._build_config()
        self._build_dataset()
        self._build_dataloader()

        self._build_engine()
        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        print(self.config)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.training_client,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            mode=OrchestrationMode.RAY,
        )

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()
        self._build_batch_balance_experiment()

    def _build_batch_balance_experiment(self):
        trainer = self.config.trainer
        metrics_path = trainer.get("sft_batch_balance_metrics_path")
        cost_path = trainer.get("sft_batch_balance_cost_model")
        self.sft_balance_profile = trainer.get("sft_batch_balance_profile", False)
        self.sft_microbatch_policy = trainer.get("sft_microbatch_policy")
        if self.sft_microbatch_policy is not None and self.sft_microbatch_policy not in MICROBATCH_POLICIES:
            raise ValueError("trainer.sft_microbatch_policy must be null, ffd, sorted, or repack")
        self.sft_defer_grad_sync = trainer.get("sft_defer_grad_sync", False)
        self.sft_reuse_unsharded_params = trainer.get("sft_reuse_unsharded_params", False)
        self.sft_comm_probe = trainer.get("sft_comm_probe", False)
        self.sft_balance_metrics = None
        self.sft_balance_cost = None
        if not metrics_path:
            if (
                self.sft_balance_profile
                or cost_path
                or self.sft_microbatch_policy is not None
                or self.sft_defer_grad_sync
                or self.sft_reuse_unsharded_params
                or self.sft_comm_probe
            ):
                raise ValueError("Set trainer.sft_batch_balance_metrics_path for the batch-balance experiment")
            return

        data = self.config.data
        if not (
            self.config.engine.strategy == "fsdp2"
            and self.config.engine.ulysses_sequence_parallel_size == 1
            and self.config.model.use_remove_padding
            and data.use_dynamic_bsz
            and data.pad_mode == "no_padding"
            and trainer.device == "cuda"
            and trainer.balance_batch
        ):
            raise ValueError("Use CUDA FSDP2 with SP=1, no padding, dynamic batching, and trainer.balance_batch=true")
        if data.max_token_len_per_gpu < 1:
            raise ValueError("data.max_token_len_per_gpu must be positive")
        if self.sft_balance_profile and cost_path:
            raise ValueError("Collect the calibration profile with the baseline cost model")
        if self.sft_comm_probe:
            torch_profile = self.config.profiler.tool_config.torch
            if (
                not self.profiler_config.enable
                or self.profiler_config.tool != "torch"
                or self.start_profile_step < 1
                or self.start_profile_step != self.end_profile_step
                or torch_profile.discrete
                or (torch_profile.contents and "cpu" not in torch_profile.contents)
                or not (self.profiler_config.all_ranks or self.profiler_config.ranks)
            ):
                raise ValueError(
                    "Use a single-step continuous Torch Profiler with CPU events and selected ranks "
                    "for the SFT communication probe"
                )
            if self.sft_balance_profile:
                raise ValueError("Run SFT communication and batch-balance calibration profiling separately")
        if cost_path:
            self.sft_balance_cost = load_cost_model(cost_path)

        self.sft_balance_metrics = Path(metrics_path).resolve()
        self.sft_balance_metrics.parent.mkdir(parents=True, exist_ok=True)
        self.sft_balance_metrics.touch(exist_ok=False)

    def _batch_balance_record(self, data, lengths, step):
        limit = self.config.data.max_token_len_per_gpu
        if not lengths or min(lengths) < 1 or max(lengths) > limit:
            raise ValueError(f"Step {step} contains an empty sample or one exceeding {limit} input tokens")

        # Hash the step's inputs before changing their order. No training tokens are logged.
        digest = hashlib.sha256()
        for key in ("input_ids", "position_ids", "loss_mask"):
            tensor = data[key]
            if tensor.is_nested:
                digest.update(tensor.offsets().cpu().numpy().tobytes())
                tensor = tensor.values()
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        loss_mask = data["loss_mask"]
        loss_tokens = int((loss_mask.values() if loss_mask.is_nested else loss_mask).sum().item())
        if loss_tokens < 1:
            raise ValueError(f"Step {step} contains no supervised tokens")
        return {"step": step, "input_tokens": sum(lengths), "loss_tokens": loss_tokens, "batch_sha256": digest.hexdigest()}

    def _set_sft_microbatch_plan(self, data, batch_seqlens, rank_partitions, balance_record):
        started = time.perf_counter()
        rank_lengths = [[batch_seqlens[index] for index in rank] for rank in rank_partitions]
        limit = self.config.data.max_token_len_per_gpu
        plan = plan_rank_microbatches(rank_lengths, limit, self.sft_microbatch_policy)
        proxy_max, proxy_spread = microbatch_plan_proxy(rank_lengths, plan)

        # One (bin, position) pair per reordered row. TensorDict DP chunking carries
        # the pair with the sample, including the order of samples inside each bin.
        placements = []
        for lengths, bins in zip(rank_lengths, plan, strict=True):
            rank_placements = [(-1, -1)] * len(lengths)
            for slot, indices in enumerate(bins):
                for position, index in enumerate(indices):
                    rank_placements[index] = (slot, position)
            placements.extend(rank_placements)
        data["sft_microbatch_placement"] = torch.tensor(
            placements, dtype=torch.long, device=data["input_ids"].device
        )
        tu.assign_non_tensor(data, sft_microbatch_plan_count=len(plan[0]))
        balance_record.update(
            microbatch_policy=self.sft_microbatch_policy,
            microbatch_proxy_max=proxy_max,
            microbatch_proxy_spread=proxy_spread,
            microbatch_plan=plan,
            rank_sample_lengths=rank_lengths,
            microbatch_plan_ms=1e3 * (time.perf_counter() - started),
        )

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        wg_kwargs = {}
        if self.start_profile_step != -1 and OmegaConf.select(self.config.profiler, "tool") == "nsys":
            wg_kwargs["profile_steps"] = list(range(self.start_profile_step, self.end_profile_step + 1))
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )

        # create resource pool and worker group
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        nnodes = self.config.trainer.nnodes
        self.resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes)
        ray_cls_with_init = RayClassWithInitArgs(ray.remote(TrainingWorker), config=config)
        self.training_client = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=self.config.trainer.device,
            **wg_kwargs,
        )
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.training_client.reset()

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor=processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor=processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = 0
        dp_size = 1

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)
        num_workers = config.data.get("num_workers", 8)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

        # update
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        return batch_seqlens

    def fit(self):
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                )
            ):
                global_step += 1
                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data).tolist()
                # this is necessary. Otherwise, it is interpreted as NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                balance_record = None
                if self.sft_balance_metrics is not None:
                    balance_record = self._batch_balance_record(data, batch_seqlens, global_step)
                    tu.assign_non_tensor(
                        data,
                        sft_balance_strict_cap=True,
                        sft_balance_log_metrics=True,
                        sft_balance_profile=self.sft_balance_profile,
                        sft_balance_defer_grad_sync=self.sft_defer_grad_sync,
                        sft_balance_reuse_unsharded_params=self.sft_reuse_unsharded_params,
                    )
                    if self.sft_comm_probe and global_step == self.start_profile_step:
                        tu.assign_non_tensor(data, sft_comm_probe=True)
                    step_started = time.perf_counter()

                if self.config.trainer.balance_batch:
                    if self.sft_balance_cost is not None:
                        global_seqlen_lst = self.sft_balance_cost.sample_weights(batch_seqlens)
                    elif balance_record is not None:
                        global_seqlen_lst = calculate_workload(torch.tensor(batch_seqlens, dtype=torch.int64)).tolist()
                    else:
                        global_seqlen_lst = torch.Tensor([item.size()[0] for item in data["input_ids"]])
                        global_seqlen_lst = calculate_workload(global_seqlen_lst)
                    dp_size = max(self.training_client._query_dispatch_info("train")) + 1
                    if balance_record is not None and len(batch_seqlens) % dp_size:
                        raise ValueError("The experiment requires an equal sample count on each DP rank")

                    global_partition_lst = get_seqlen_balanced_partitions(
                        global_seqlen_lst, k_partitions=dp_size, equal_size=True
                    )
                    if balance_record is not None:
                        balance_record["rank_input_tokens"] = [
                            sum(batch_seqlens[i] for i in partition) for partition in global_partition_lst
                        ]
                    # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
                    for idx, partition in enumerate(global_partition_lst):
                        partition.sort(key=lambda x: (global_seqlen_lst[x], x))
                        ordered_partition = partition[::2] + partition[1::2][::-1]
                        global_partition_lst[idx] = ordered_partition
                    if balance_record is not None:
                        balance_record["rank_sample_indices"] = global_partition_lst

                    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])

                    data = tu.index_select_tensor_dict(data, global_idx)
                    if self.sft_microbatch_policy is not None:
                        self._set_sft_microbatch_plan(data, batch_seqlens, global_partition_lst, balance_record)

                if balance_record is not None:
                    balance_record["balance_ms"] = 1000 * (time.perf_counter() - step_started)

                # start profile in SPMD mode
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()

                # train for on batch
                output = self.training_client.train_batch(data)
                output = output.get()
                if balance_record is not None:
                    balance_record["step_s"] = time.perf_counter() - step_started

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                metrics = tu.get(output, "metrics")
                if balance_record is not None:
                    balance_record.update(
                        loss=float(metrics["loss"]),
                        grad_norm=float(metrics["grad_norm"]),
                        rank_wall_ms=metrics.pop("sft_balance_rank_wall_ms"),
                        peak_bytes=metrics.pop("sft_balance_step_peak_bytes"),
                        microbatch_count=metrics.pop("sft_comm_microbatch_count"),
                        grad_sync_rounds=metrics.pop("sft_comm_grad_sync_rounds"),
                        reshard_rounds=metrics.pop("sft_comm_reshard_rounds"),
                    )
                    if self.sft_balance_profile:
                        balance_record["microbatches"] = metrics.pop("sft_balance_microbatches")
                    with self.sft_balance_metrics.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(balance_record, allow_nan=False) + "\n")

                # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                metrics["train/loss"] = metrics.pop("loss")
                metrics["train/grad_norm"] = metrics.pop("grad_norm")
                metrics["train/lr"] = metrics.pop("lr")
                metrics["train/mfu"] = metrics.pop("mfu")
                metrics["train/global_tokens"] = torch.sum(torch.tensor(batch_seqlens, device=self.device_name)).item()
                total_tokens += metrics["train/global_tokens"]
                metrics["train/total_tokens(B)"] = total_tokens / 1e9
                tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)
                        output = output.get()
                        metrics = tu.get(output, "metrics")
                        val_losses.append(metrics["loss"])

                    val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))

                    metric = {"val/loss": val_loss.detach().item()}
                    tracking.log(data=metric, step=global_step)
                    last_valid_metric = metric

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    print(f"Total time for train steps: {train_time:.2f}s")
                    print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    ray.init()
    trainer = SFTTrainer(config=config)
    trainer.fit()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()
