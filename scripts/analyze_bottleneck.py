#!/usr/bin/env python3
# scripts/analyze_bottleneck.py
"""
Athena Runtime Analyzer

读取 Athena 采集的 JSONL 日志，生成训练瓶颈分析报告和可视化图表。

Usage:
    python scripts/analyze_bottleneck.py --logdir ./athena_logs --output ./reports
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 设置中文字体（如果系统有）
try:
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'sans-serif']
except:
    pass
rcParams['axes.unicode_minus'] = False


# ============================================================
# 数据加载
# ============================================================

def load_athena_logs(log_path: str) -> List[Dict[str, Any]]:
    """
    加载 Athena JSONL 日志文件。

    如果 log_path 是目录，则加载该目录下所有 trace_step_*.jsonl 文件；
    如果 log_path 是文件，则只加载该文件。

    Args:
        log_path: 日志文件或目录的路径

    Returns:
        日志记录列表（每条记录是一个字典）
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {log_path}")

    # 确定要读取的文件列表
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("trace_step_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No trace_step_*.jsonl files found in {log_path}")

    records = []
    for f in files:
        with open(f, 'r') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {f}: {e}")
                        continue
    return records


def flatten_state(record: Dict[str, Any]) -> Dict[str, Any]:
    """将嵌套的 RuntimeState 展平为单层字典，便于分析"""
    flat = {}
    
    # 基础字段
    for key in ['global_step', 'epoch', 'timestamp', 
                'gen_time', 'ref_time', 'values_time', 'adv_time',
                'update_actor_time', 'update_critic_time', 'update_weights_time',
                'total_step_time', 'throughput', 'total_num_tokens',
                'avg_prompt_len', 'avg_response_len']:
        flat[key] = record.get(key, 0.0)
    
    # GPU 状态
    gpu = record.get('gpu', {})
    if isinstance(gpu, dict):
        for key in ['utilization', 'memory_allocated_gb', 'memory_reserved_gb',
                    'memory_peak_gb', 'memory_used_gb', 'memory_utilization',
                    'power_watts', 'temperature_celsius']:
            flat[f'gpu_{key}'] = gpu.get(key, 0.0)
    
    # vLLM 状态
    vllm = record.get('vllm', {})
    if isinstance(vllm, dict):
        for key in ['num_running', 'num_waiting', 'num_preempted', 'num_evictions',
                    'kv_usage', 'prefix_cache_hit', 'batch_size',
                    'prefill_tokens', 'decode_tokens', 'scheduler_queue']:
            flat[f'vllm_{key}'] = vllm.get(key, 0.0)
    
    # TransferQueue 状态
    tq = record.get('transfer_queue', {})
    if isinstance(tq, dict):
        for key in ['running', 'finished', 'failure', 'pending', 'queue_length', 'latency_ms']:
            flat[f'tq_{key}'] = tq.get(key, 0.0)
    
    # Rollout 状态
    rollout = record.get('rollout', {})
    if isinstance(rollout, dict):
        for key in ['generate_latency_ms', 'request_latency_ms', 'ttft_ms', 'tpot_ms',
                    'num_preempted', 'num_aborted']:
            flat[f'rollout_{key}'] = rollout.get(key, 0.0)
    
    # Training 状态
    training = record.get('training', {})
    if isinstance(training, dict):
        for key in ['entropy', 'grad_norm', 'lr', 'ppo_kl', 'clip_fraction',
                    'pg_loss', 'kl_loss', 'actor_loss', 'critic_loss',
                    'reward', 'advantage', 'returns', 'mfu']:
            flat[f'training_{key}'] = training.get(key, 0.0)
    
    # Extra
    extra = record.get('extra', {})
    if isinstance(extra, dict):
        for key, val in extra.items():
            flat[f'extra_{key}'] = val
    
    return flat


# ============================================================
# 分析核心
# ============================================================

def compute_stage_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算各阶段耗时占比"""
    stages = ['gen_time', 'ref_time', 'values_time', 'adv_time',
              'update_actor_time', 'update_critic_time', 'update_weights_time']
    
    # 提取每个 stage 的数据
    stage_data = {stage: [] for stage in stages}
    total_times = []
    
    for r in records:
        total = r.get('total_step_time', 0.0)
        if total <= 0:
            continue
        total_times.append(total)
        for stage in stages:
            val = r.get(stage, 0.0)
            stage_data[stage].append(val)
    
    if not total_times:
        return {'error': 'No valid step data found'}
    
    # 计算平均值和占比
    avg_total = np.mean(total_times)
    result = {
        'total_steps': len(total_times),
        'avg_total_step_time': avg_total,
        'stages': {}
    }
    
    for stage, vals in stage_data.items():
        if vals:
            avg_val = np.mean(vals)
            result['stages'][stage] = {
                'mean': avg_val,
                'std': np.std(vals) if len(vals) > 1 else 0.0,
                'pct': avg_val / avg_total * 100 if avg_total > 0 else 0.0,
                'max': np.max(vals),
                'min': np.min(vals),
            }
        else:
            result['stages'][stage] = {'mean': 0.0, 'std': 0.0, 'pct': 0.0, 'max': 0.0, 'min': 0.0}
    
    return result


def compute_gpu_analysis(flat_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析 GPU 状态"""
    mem_peak = [r.get('gpu_memory_peak_gb', 0.0) for r in flat_records if r.get('gpu_memory_peak_gb', 0.0) > 0]
    util = [r.get('gpu_utilization', 0.0) for r in flat_records if r.get('gpu_utilization', 0.0) > 0]
    power = [r.get('gpu_power_watts', 0.0) for r in flat_records if r.get('gpu_power_watts', 0.0) > 0]
    temp = [r.get('gpu_temperature_celsius', 0.0) for r in flat_records if r.get('gpu_temperature_celsius', 0.0) > 0]
    
    return {
        'memory_peak_gb': {'mean': np.mean(mem_peak) if mem_peak else 0.0,
                           'max': np.max(mem_peak) if mem_peak else 0.0,
                           'min': np.min(mem_peak) if mem_peak else 0.0},
        'utilization': {'mean': np.mean(util) if util else 0.0,
                        'max': np.max(util) if util else 0.0,
                        'min': np.min(util) if util else 0.0},
        'power_watts': {'mean': np.mean(power) if power else 0.0,
                        'max': np.max(power) if power else 0.0},
        'temperature_celsius': {'mean': np.mean(temp) if temp else 0.0,
                                'max': np.max(temp) if temp else 0.0},
    }


# ============================================================
# 可视化
# ============================================================

def plot_step_time_breakdown(records: List[Dict[str, Any]], output_dir: Path):
    """
    绘制 Step 时间分解堆叠图
    """
    stages = ['gen_time', 'ref_time', 'values_time', 'adv_time',
              'update_actor_time', 'update_critic_time', 'update_weights_time']
    
    stage_labels = {
        'gen_time': 'Rollout (Gen)',
        'ref_time': 'Reference Model',
        'values_time': 'Values (Critic)',
        'adv_time': 'Advantage',
        'update_actor_time': 'Actor Update',
        'update_critic_time': 'Critic Update',
        'update_weights_time': 'Weight Sync',
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#F39C12']
    
    steps = [r.get('global_step', i) for i, r in enumerate(records)]
    stage_values = {stage: [r.get(stage, 0.0) for r in records] for stage in stages}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 堆叠
    bottoms = np.zeros(len(records))
    for stage, color in zip(stages, colors):
        vals = np.array(stage_values[stage])
        ax.bar(steps, vals, bottom=bottoms, label=stage_labels.get(stage, stage), 
               color=color, edgecolor='white', linewidth=0.5)
        bottoms += vals
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Step Time Breakdown by Phase', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加总时间线
    total_times = [r.get('total_step_time', 0.0) for r in records]
    ax2 = ax.twinx()
    ax2.plot(steps, total_times, 'o-', color='#2C3E50', linewidth=1.5, 
             markersize=4, label='Total Step Time', alpha=0.7)
    ax2.set_ylabel('Total Step Time (s)', fontsize=12, color='#2C3E50')
    ax2.tick_params(axis='y', labelcolor='#2C3E50')
    
    plt.tight_layout()
    output_path = output_dir / 'step_time_breakdown.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_stage_pie(breakdown: Dict[str, Any], output_dir: Path):
    """
    绘制各阶段耗时饼图
    """
    stages = breakdown.get('stages', {})
    if not stages:
        return
    
    labels = []
    sizes = []
    colors = []
    
    stage_labels = {
        'gen_time': 'Rollout (Gen)',
        'ref_time': 'Reference Model',
        'values_time': 'Values (Critic)',
        'adv_time': 'Advantage',
        'update_actor_time': 'Actor Update',
        'update_critic_time': 'Critic Update',
        'update_weights_time': 'Weight Sync',
    }
    
    color_map = {
        'gen_time': '#FF6B6B',
        'ref_time': '#4ECDC4',
        'values_time': '#45B7D1',
        'adv_time': '#96CEB4',
        'update_actor_time': '#FFEAA7',
        'update_critic_time': '#DDA0DD',
        'update_weights_time': '#F39C12',
    }
    
    for stage, data in stages.items():
        pct = data.get('pct', 0.0)
        if pct > 0.5:
            labels.append(stage_labels.get(stage, stage))
            sizes.append(pct)
            colors.append(color_map.get(stage, '#888888'))
    
    if not sizes:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11},
        explode=[0.02] * len(sizes)
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Average Step Time Breakdown', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'stage_pie.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_throughput(records: List[Dict[str, Any]], output_dir: Path):
    """
    绘制 Throughput 趋势图
    """
    steps = [r.get('global_step', i) for i, r in enumerate(records)]
    throughputs = [r.get('throughput', 0.0) for r in records]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(steps, throughputs, 'o-', color='#27AE60', linewidth=2, markersize=5)
    ax.axhline(np.mean(throughputs), color='#E74C3C', linestyle='--', alpha=0.7,
               label=f'Mean: {np.mean(throughputs):.1f} tok/s')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=12)
    ax.set_title('Training Throughput Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'throughput.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_gpu_memory(records: List[Dict[str, Any]], flat_records: List[Dict[str, Any]], output_dir: Path):
    """
    绘制 GPU 显存趋势图
    """
    steps = [r.get('global_step', i) for i, r in enumerate(records)]
    
    # 从展平数据中提取
    peak = [r.get('gpu_memory_peak_gb', 0.0) for r in flat_records]
    allocated = [r.get('gpu_memory_allocated_gb', 0.0) for r in flat_records]
    reserved = [r.get('gpu_memory_reserved_gb', 0.0) for r in flat_records]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(steps, peak, 'o-', color='#E74C3C', linewidth=2, markersize=5, label='Peak Memory')
    ax.plot(steps, allocated, 's-', color='#3498DB', linewidth=1.5, markersize=4, label='Allocated')
    ax.plot(steps, reserved, '^-', color='#2ECC71', linewidth=1.5, markersize=4, label='Reserved')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_title('GPU Memory Usage Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'gpu_memory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_gpu_utilization(flat_records: List[Dict[str, Any]], output_dir: Path):
    """
    绘制 GPU 利用率、功耗、温度
    """
    steps = [r.get('global_step', i) for i, r in enumerate(flat_records)]
    
    util = [r.get('gpu_utilization', 0.0) for r in flat_records]
    power = [r.get('gpu_power_watts', 0.0) for r in flat_records]
    temp = [r.get('gpu_temperature_celsius', 0.0) for r in flat_records]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Utilization
    axes[0].plot(steps, util, 'o-', color='#8E44AD', linewidth=2, markersize=5)
    axes[0].set_ylabel('GPU Utilization', fontsize=12)
    axes[0].set_title('GPU Utilization', fontsize=12)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].set_ylim(0, 1.05)
    
    # Power
    axes[1].plot(steps, power, 's-', color='#E67E22', linewidth=2, markersize=5)
    axes[1].set_ylabel('Power (Watts)', fontsize=12)
    axes[1].set_title('GPU Power Consumption', fontsize=12)
    axes[1].grid(alpha=0.3, linestyle='--')
    
    # Temperature
    axes[2].plot(steps, temp, '^-', color='#E74C3C', linewidth=2, markersize=5)
    axes[2].set_xlabel('Step', fontsize=12)
    axes[2].set_ylabel('Temperature (°C)', fontsize=12)
    axes[2].set_title('GPU Temperature', fontsize=12)
    axes[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'gpu_utilization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


# ============================================================
# 文本报告
# ============================================================

def generate_report(breakdown: Dict[str, Any], gpu_analysis: Dict[str, Any],
                    records: List[Dict[str, Any]], flat_records: List[Dict[str, Any]]) -> str:
    """生成文本分析报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("Athena Runtime Analysis Report")
    lines.append("=" * 60)
    lines.append("")
    
    # 基本统计
    n_steps = len(records)
    lines.append(f"Total Steps Analyzed: {n_steps}")
    lines.append("")
    
    # Step 时间分解
    avg_total = breakdown.get('avg_total_step_time', 0.0)
    lines.append(f"Average Step Time: {avg_total:.3f} seconds")
    lines.append("")
    lines.append("Stage Breakdown:")
    lines.append("-" * 40)
    
    stages = breakdown.get('stages', {})
    sorted_stages = sorted(stages.items(), key=lambda x: x[1].get('mean', 0.0), reverse=True)
    
    stage_labels = {
        'gen_time': 'Rollout (Gen)',
        'ref_time': 'Reference Model',
        'values_time': 'Values (Critic)',
        'adv_time': 'Advantage',
        'update_actor_time': 'Actor Update',
        'update_critic_time': 'Critic Update',
        'update_weights_time': 'Weight Sync',
    }
    
    for stage, data in sorted_stages:
        mean = data.get('mean', 0.0)
        pct = data.get('pct', 0.0)
        label = stage_labels.get(stage, stage)
        lines.append(f"  {label:20s}: {mean:7.3f}s  ({pct:5.1f}%)")
    
    lines.append("")
    
    # 瓶颈识别
    lines.append("Performance Bottlenecks:")
    lines.append("-" * 40)
    
    # 找出最大的 stage
    if sorted_stages:
        top_stage, top_data = sorted_stages[0]
        top_label = stage_labels.get(top_stage, top_stage)
        top_pct = top_data.get('pct', 0.0)
        lines.append(f"  ⚠️  Top bottleneck: {top_label} ({top_pct:.1f}% of step time)")
        
        if top_pct > 30:
            lines.append(f"     → {top_label} consumes >30% of step time. Consider optimization.")
    
    # GPU 分析
    lines.append("")
    lines.append("GPU Status:")
    lines.append("-" * 40)
    mem = gpu_analysis.get('memory_peak_gb', {})
    util = gpu_analysis.get('utilization', {})
    power = gpu_analysis.get('power_watts', {})
    temp = gpu_analysis.get('temperature_celsius', {})
    
    lines.append(f"  Memory Peak:      {mem.get('mean', 0.0):.2f} GB  (max: {mem.get('max', 0.0):.2f} GB)")
    lines.append(f"  GPU Utilization:  {util.get('mean', 0.0)*100:.1f}%  (max: {util.get('max', 0.0)*100:.1f}%)")
    lines.append(f"  Power:            {power.get('mean', 0.0):.1f} W  (max: {power.get('max', 0.0):.1f} W)")
    lines.append(f"  Temperature:      {temp.get('mean', 0.0):.1f} °C  (max: {temp.get('max', 0.0):.1f} °C)")
    
    # Throughput
    lines.append("")
    lines.append("Throughput:")
    lines.append("-" * 40)
    throughputs = [r.get('throughput', 0.0) for r in records if r.get('throughput', 0.0) > 0]
    if throughputs:
        lines.append(f"  Mean:  {np.mean(throughputs):.1f} tok/s")
        lines.append(f"  Max:   {np.max(throughputs):.1f} tok/s")
        lines.append(f"  Min:   {np.min(throughputs):.1f} tok/s")
        lines.append(f"  Std:   {np.std(throughputs):.1f} tok/s")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Athena Runtime Analyzer')
    parser.add_argument('--logdir', type=str, default='./athena_logs/trace_step_5667.jsonl',
                        help='Directory containing Athena JSONL logs')
    parser.add_argument('--output', type=str, default='./reports',
                        help='Output directory for reports and figures')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots, only print text report')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Athena] Loading logs from: {args.logdir}")
    records = load_athena_logs(args.logdir)
    
    if not records:
        print("[Athena] No records found!")
        return
    
    print(f"[Athena] Loaded {len(records)} step records")
    flat_records = [flatten_state(r) for r in records]
    
    # 分析
    print("[Athena] Computing analysis...")
    breakdown = compute_stage_breakdown(records)
    gpu_analysis = compute_gpu_analysis(flat_records)
    
    # 生成文本报告
    report = generate_report(breakdown, gpu_analysis, records, flat_records)
    print(report)
    
    # 保存报告
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n[Athena] Report saved to: {report_path}")
    
    # 生成图表
    if not args.no_plots:
        print("\n[Athena] Generating plots...")
        plot_step_time_breakdown(records, output_dir)
        plot_stage_pie(breakdown, output_dir)
        plot_throughput(records, output_dir)
        plot_gpu_memory(records, flat_records, output_dir)
        plot_gpu_utilization(flat_records, output_dir)
        print(f"\n[Athena] All plots saved to: {output_dir}")
    
    print("\n[Athena] Analysis complete!")


if __name__ == "__main__":
    main()