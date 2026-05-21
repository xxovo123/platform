"""
数据管理模块

功能:
  1. MDM 数据缓存 (保存/加载 .npz 文件)
  2. 仿真元数据管理 (文件名生成、参数记录)
  3. CIR / 频谱数据导出
"""

import os
import json
import numpy as np
from datetime import datetime


# =============================================================================
# 文件名生成
# =============================================================================
def make_cache_filename(
    n_antennas,
    scenarios,
    antenna_mode,
    no_multipath,
    resolution,
    randomize_rx_y_pos,
    num_movements,
    movement_step_cm,
    prefix='mdm_rw',
):
    """生成 MDM 缓存文件名 (不含路径)。"""
    scenarios_str = "-".join(sorted(scenarios))
    return (
        f"{prefix}_ant{n_antennas}_scen-{scenarios_str}_mode-{antenna_mode}"
        f"_nomp{no_multipath}_res{resolution}"
        f"_randy{randomize_rx_y_pos}_steps{num_movements}_stepcm{movement_step_cm}.npz"
    )


# =============================================================================
# MDM 数据缓存
# =============================================================================
def save_mdm_cache(
    filepath,
    mdm_data_sequence,
    all_scenario_paths,
    dt_meep,
    tx_positions_si,
    rx_positions_si,
    extra_metadata=None,
):
    """
    将 MDM 序列数据保存为 .npz 缓存文件。

    参数:
        filepath (str): 保存路径 (含文件名)
        mdm_data_sequence (list[np.ndarray]): 每步 MDM 快照
        all_scenario_paths (list[list[mp.Vector3]]): 目标移动路径
        dt_meep (float): Meep 时间步长
        tx_positions_si (np.ndarray): TX 天线位置
        rx_positions_si (np.ndarray): RX 天线位置
        extra_metadata (dict): 额外元数据
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    # 转换路径为数组
    mdm_arr = np.array(mdm_data_sequence)
    paths_arr = np.array([
        [[p.x, p.y, p.z] for p in path]
        for path in all_scenario_paths
    ])

    save_dict = {
        'mdm_data_sequence': mdm_arr,
        'all_scenario_paths': paths_arr,
        'dt_meep': dt_meep,
        'tx_positions_si': tx_positions_si,
        'rx_positions_si': rx_positions_si,
    }

    if extra_metadata:
        save_dict['metadata'] = extra_metadata

    np.savez(filepath, **save_dict)
    print(f"MDM 数据已缓存至: {filepath}")


def load_mdm_cache(filepath):
    """
    加载缓存的 MDM 数据。

    返回:
        dict: {
            'mdm_data_sequence': list[np.ndarray],
            'all_scenario_paths': list[list],
            'dt_meep': float,
            'tx_positions_si': np.ndarray,
            'rx_positions_si': np.ndarray,
            'metadata': dict or None,
        }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"缓存文件未找到: {filepath}")

    print(f"加载 MDM 缓存: {filepath}")
    data = np.load(filepath, allow_pickle=True)

    result = {
        'mdm_data_sequence': [data['mdm_data_sequence'][i] for i in range(data['mdm_data_sequence'].shape[0])],
        'all_scenario_paths': data['all_scenario_paths'],
        'dt_meep': float(data['dt_meep']),
        'tx_positions_si': data['tx_positions_si'],
        'rx_positions_si': data['rx_positions_si'],
        'metadata': data.get('metadata', None),
    }
    return result


# =============================================================================
# CIR 数据保存
# =============================================================================
def save_cir_data(
    output_dir,
    run_id,
    cir_snapshots,
    time_axis,
    freq_axis,
    metadata=None,
):
    """
    保存 CIR 快照及相关元数据。

    参数:
        output_dir (str): 输出目录
        run_id (str): 运行标识
        cir_snapshots (list[np.ndarray]): CIR 快照列表
        time_axis (np.ndarray): 时间轴 [秒]
        freq_axis (np.ndarray): 频率轴 [Hz]
        metadata (dict): 附加元数据
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"cir_{run_id}.npz")

    np.savez(
        filepath,
        cir_snapshots=np.array(cir_snapshots),
        time_axis=time_axis,
        freq_axis=freq_axis,
    )
    print(f"CIR 数据已保存至: {filepath}")

    # 保存 JSON 元数据
    if metadata:
        meta_path = os.path.join(output_dir, f"cir_{run_id}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        print(f"元数据已保存至: {meta_path}")


# =============================================================================
# 仿真运行日志
# =============================================================================
def make_run_id(prefix='sim', include_timestamp=True):
    """生成唯一的运行 ID。"""
    if include_timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{ts}"
    return prefix


def export_simulation_summary(output_dir, run_id, params, file_sizes=None):
    """
    将仿真运行的关键参数汇总为可读的文本文件。

    参数:
        output_dir (str): 输出目录
        run_id (str): 运行标识
        params (dict): 仿真参数字典
        file_sizes (dict): 输出文件大小 {filename: bytes}
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"summary_{run_id}.txt")

    lines = []
    lines.append("=" * 60)
    lines.append(f"Simulation Summary: {run_id}")
    lines.append(f"Date: {datetime.now().isoformat()}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("[Simulation Parameters]")
    for key, value in sorted(params.items()):
        lines.append(f"  {key}: {value}")

    if file_sizes:
        lines.append("")
        lines.append("[Output Files]")
        for fname, fsize in sorted(file_sizes.items()):
            lines.append(f"  {fname}: {fsize / 1024:.1f} KB")

    lines.append("")
    lines.append("=" * 60)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"汇总已保存至: {filepath}")
