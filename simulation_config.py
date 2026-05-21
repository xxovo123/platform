"""
仿真平台统一入口

功能:
  1. 参数配置 (频率、带宽、采样率、域大小、CIR 点数等)
  2. 场景定义 (目标位置预设)
  3. 运行完整仿真流水线: FDTD → MDM → CIR
  4. 数据缓存与导出

无算法内容 — 仅负责参数管理、流水线调度与数据保存。
"""

import os
import time
import numpy as np
import meep as mp

from fdtd_simulator import (
    make_car_interior_geometry,
    generate_random_walk_path,
    collect_mdm_sequence,
)
from antenna_config import build_antenna_arrays
from signal_converter import (
    generate_reference_pulse_baseband,
    prepare_baseband_data,
)
from data_manager import (
    make_cache_filename,
    save_mdm_cache,
    load_mdm_cache,
    save_cir_data,
    make_run_id,
    export_simulation_summary,
)


# =============================================================================
# 默认仿真参数
# =============================================================================
DEFAULT_SIM_PARAMS = {
    # --- 物理常数 ---
    'c0': 2.99792458e8,

    # --- 频率参数 ---
    'fc_si': 7.9872e9,           # 中心频率 [Hz]
    'bandwidth_si': 500e6,        # 带宽 [Hz]

    # --- FDTD 参数 ---
    'resolution': 250,            # Meep 分辨率 [pixels/Meep unit]
    'domain_size_x_meep': 3.5,    # 域 X 尺寸 [Meep 单位]
    'domain_size_y_meep': 3.5,    # 域 Y 尺寸 [Meep 单位]
    'run_time_meep': 20,          # 仿真总时长 [Meep 单位]
    'epsilon_r_background': 1.0,  # 背景相对介电常数

    # --- 天线参数 ---
    'n_antennas': 12,
    'antenna_mode': 'separated',  # 'separated' | 'colocated' | 'hybrid_tdm' | 'hybrid_tdm_tx_top'
    'array_center_y': 0.0,
    'randomize_rx_y_pos': False,

    # --- 场景参数 ---
    'active_scenarios': ['rear_right_seat'],
    'no_multipath': True,
    'include_pillars': True,
    'include_seats': True,

    # --- 随机游走参数 ---
    'enable_random_walk': False,
    'num_movements': 20,
    'movement_step_cm': 5.0,
    'movement_radius_cm': 10.0,

    # --- 信号处理参数 ---
    'use_baseband_processing': True,
    'target_sample_rate': 1e9,    # 目标采样率 [Hz]
    'cir_points': 64,             # CIR 截断点数

    # --- 数据管理参数 ---
    'output_dir': './sim_output',
    'cache_dir': './mdm_cache',
    'use_cache': True,
    'run_id': None,               # None 则自动生成
}


# =============================================================================
# 预设场景 (目标位置)
# =============================================================================
PRESET_SCENARIOS = {
    'front_left_seat':  (mp.Vector3(-0.13, 0.5 - 0.075 - 0.10), mp.Vector3(-0.13, 0.5 - 0.075 - 0.15)),
    'front_right_seat': (mp.Vector3(0.13, 0.5 - 0.075 - 0.10),  mp.Vector3(0.13, 0.5 - 0.075 - 0.15)),
    'rear_left_seat':   (mp.Vector3(-0.5, -0.8 + 0.075 + 0.10), mp.Vector3(-0.5, -0.8 + 0.075 + 0.15)),
    'rear_right_seat':  (mp.Vector3(0.5, -0.8 + 0.075 + 0.10),  mp.Vector3(0.5, -0.8 + 0.075 + 0.15)),
}


# =============================================================================
# 自定义场景注册
# =============================================================================
def register_scenario(name, start_pos, end_pos):
    """注册自定义目标场景。"""
    PRESET_SCENARIOS[name] = (start_pos, end_pos)


# =============================================================================
# 主仿真运行器
# =============================================================================
def run_simulation_pipeline(params=None, callback_cir_ready=None):
    """
    运行完整仿真流水线: FDTD → CIR。

    流水线步骤:
      1. 合并参数 (用户参数覆盖默认值)
      2. 构建天线阵列
      3. 定义场景几何体
      4. 生成随机游走路径
      5. 运行 FDTD 仿真 / 加载缓存
      6. FDTD 通带信号 → 基带 CIR (IQ 解调 + 下采样 + 匹配滤波)
      7. 保存 CIR 数据

    参数:
        params (dict): 覆盖默认参数 (可选)
        callback_cir_ready (callable): 回调函数, 每个差分 CIR 快照就绪时调用
            签名: callback(snapshot_index, cir, time_axis, freq_axis, true_positions)

    返回:
        dict: {
            'cir_snapshots': list[np.ndarray],
            'time_axis': np.ndarray,
            'freq_axis': np.ndarray,
            'true_positions': list,
            'antenna_info': dict,
            'dt_meep': float,
            'run_id': str,
            'params': dict,
        }
    """
    # --- 1. 合并参数 ---
    p = DEFAULT_SIM_PARAMS.copy()
    if params:
        p.update(params)

    if p['run_id'] is None:
        p['run_id'] = make_run_id()

    c0 = p['c0']
    fc_si = p['fc_si']
    bandwidth_si = p['bandwidth_si']
    freq_range_si = (fc_si - bandwidth_si / 2, fc_si + bandwidth_si / 2)
    m = 1.0  # Meep 长度尺度因子

    os.makedirs(p['output_dir'], exist_ok=True)
    os.makedirs(p['cache_dir'], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Simulation Pipeline: {p['run_id']}")
    print(f"{'='*60}")
    print(f"  fc = {fc_si / 1e9:.2f} GHz, BW = {bandwidth_si / 1e6:.0f} MHz")
    print(f"  Antennas = {p['n_antennas']}, Mode = {p['antenna_mode']}")
    print(f"  Scenarios = {p['active_scenarios']}")
    print(f"  Random Walk = {p['enable_random_walk']}")
    print(f"{'='*60}")

    # --- 2. 天线阵列 ---
    antenna_info = build_antenna_arrays(
        n_antennas=p['n_antennas'],
        fc_si=fc_si,
        epsilon_r_background=p['epsilon_r_background'],
        antenna_mode=p['antenna_mode'],
        array_center_y=p['array_center_y'],
        randomize_rx_y_pos=p['randomize_rx_y_pos'],
        c0=c0,
    )
    # 标记模式以便后续使用
    antenna_info['_antenna_mode'] = p['antenna_mode']

    lambda_c_si = c0 / (fc_si * np.sqrt(p['epsilon_r_background']))
    print(f"  波长 = {lambda_c_si * 100:.2f} cm, 天线间距 = {antenna_info['antenna_spacing_si'] * 100:.2f} cm")

    # --- 3. 场景几何体 ---
    static_geometry = make_car_interior_geometry(
        resolution=p['resolution'],
        include_pillars=p['include_pillars'],
        include_seats=p['include_seats'],
    )

    if p['no_multipath']:
        static_geometry = []  # 清空多径散射体

    # --- 4. 目标路径 ---
    all_scenario_paths = []
    for name in p['active_scenarios']:
        if name not in PRESET_SCENARIOS:
            print(f"[警告] 场景 '{name}' 未注册, 跳过。")
            continue
        start_pos, _ = PRESET_SCENARIOS[name]

        if p['enable_random_walk']:
            movement_step_m = p['movement_step_cm'] / 100.0
            movement_radius_m = p['movement_radius_cm'] / 100.0
            path = generate_random_walk_path(
                start_pos,
                p['num_movements'],
                movement_step_m,
                movement_radius_m,
            )
        else:
            path = [start_pos]  # 静止目标

        all_scenario_paths.append(path)

    if not all_scenario_paths:
        raise ValueError("无有效场景。")

    num_moving_targets = len(all_scenario_paths)

    # --- 5. FDTD 仿真 / 缓存 ---
    fc_meep = fc_si * m / c0
    duration_si = 2.0 / bandwidth_si
    duration_meep = duration_si * c0 / m

    pml_thickness_meep = 2 * lambda_c_si / m
    pml_layers = [mp.PML(pml_thickness_meep)]
    cell_size = mp.Vector3(p['domain_size_x_meep'], p['domain_size_y_meep'], 0)
    background_medium = mp.Medium(epsilon=p['epsilon_r_background'])

    mdm_filename = make_cache_filename(
        n_antennas=p['n_antennas'],
        scenarios=p['active_scenarios'],
        antenna_mode=p['antenna_mode'],
        no_multipath=p['no_multipath'],
        resolution=p['resolution'],
        randomize_rx_y_pos=p['randomize_rx_y_pos'],
        num_movements=p['num_movements'],
        movement_step_cm=p['movement_step_cm'],
    )
    mdm_filepath = os.path.join(p['cache_dir'], mdm_filename)

    if p['use_cache'] and os.path.exists(mdm_filepath):
        cache = load_mdm_cache(mdm_filepath)
        mdm_sequence = cache['mdm_data_sequence']
        dt_meep = cache['dt_meep']
        # 从缓存恢复天线位置 (如果不存在)
        if 'tx_positions_si' in cache:
            antenna_info['tx_positions_si'] = cache['tx_positions_si']
            antenna_info['rx_positions_si'] = cache['rx_positions_si']
    else:
        print("运行 FDTD 仿真...")
        t_start = time.time()

        mdm_sequence, dt_meep = collect_mdm_sequence(
            geometry_static=static_geometry,
            cell_size=cell_size,
            resolution=p['resolution'],
            pml_layers=pml_layers,
            background_medium=background_medium,
            antenna_info=antenna_info,
            all_scenario_paths=all_scenario_paths,
            fc_meep=fc_meep,
            duration_meep=duration_meep,
            run_time_meep=p['run_time_meep'],
            lambda_c_si=lambda_c_si,
            m=m,
        )

        elapsed = time.time() - t_start
        print(f"FDTD 仿真完成, 耗时 {elapsed:.1f} 秒。")

        save_mdm_cache(
            filepath=mdm_filepath,
            mdm_data_sequence=mdm_sequence,
            all_scenario_paths=all_scenario_paths,
            dt_meep=dt_meep,
            tx_positions_si=antenna_info['tx_positions_si'],
            rx_positions_si=antenna_info['rx_positions_si'],
        )

    # --- 6. FDTD → CIR 转换 ---
    cir_snapshots = []
    true_positions_list = []
    time_axis_final = None
    freq_axis_final = None

    time_axis_full = np.arange(mdm_sequence[0].shape[2]) * (dt_meep * m / c0)
    ref_pulse = None

    if p['use_baseband_processing']:
        ref_pulse = generate_reference_pulse_baseband(
            time_axis_full, fc_si, duration_si, p['target_sample_rate']
        )
        print(f"基带参考脉冲已生成: {len(ref_pulse)} 点")

    num_snapshots = len(mdm_sequence) - 1 if p['enable_random_walk'] else 1

    for i in range(num_snapshots):
        if p['enable_random_walk']:
            mdm_diff = mdm_sequence[i + 1] - mdm_sequence[i]
            # 差分 CIR 的参考位置: 路径中点
            true_pos = [
                mp.Vector3(
                    (path[i].x + path[i + 1].x) / 2,
                    (path[i].y + path[i + 1].y) / 2,
                    0,
                )
                for path in all_scenario_paths
            ]
        else:
            mdm_diff = mdm_sequence[0]
            true_pos = [path[0] for path in all_scenario_paths]

        if p['use_baseband_processing']:
            cir, t_axis, f_axis = prepare_baseband_data(
                mdm_passband=mdm_diff,
                time_axis_si=time_axis_full,
                fc_si=fc_si,
                target_fs_si=p['target_sample_rate'],
                reference_pulse_baseband=ref_pulse,
                target_num_points=p['cir_points'],
            )
        else:
            # 直通模式: 保持通带
            cir = mdm_diff
            t_axis = time_axis_full
            f_axis = np.fft.fftfreq(len(t_axis), d=(t_axis[1] - t_axis[0]))

        cir_snapshots.append(cir)
        true_positions_list.append(true_pos)

        if time_axis_final is None:
            time_axis_final = t_axis
            freq_axis_final = f_axis

        if callback_cir_ready:
            callback_cir_ready(i, cir, t_axis, f_axis, true_pos)

        print(f"CIR 快照 {i + 1}/{num_snapshots} 就绪: shape={cir.shape}")

    # --- 7. 保存 ---
    save_cir_data(
        output_dir=p['output_dir'],
        run_id=p['run_id'],
        cir_snapshots=cir_snapshots,
        time_axis=time_axis_final,
        freq_axis=freq_axis_final,
        metadata={
            'fc_si': fc_si,
            'bandwidth_si': bandwidth_si,
            'freq_range_si': freq_range_si,
            'target_sample_rate': p['target_sample_rate'],
            'n_antennas': p['n_antennas'],
            'antenna_mode': p['antenna_mode'],
            'epsilon_r_background': p['epsilon_r_background'],
            'active_scenarios': p['active_scenarios'],
            'num_snapshots': num_snapshots,
            'num_moving_targets': num_moving_targets,
            'antenna_info_tx': antenna_info['tx_positions_si'].tolist(),
            'antenna_info_rx': antenna_info['rx_positions_si'].tolist(),
        },
    )

    export_simulation_summary(p['output_dir'], p['run_id'], p)

    result = {
        'cir_snapshots': cir_snapshots,
        'time_axis': time_axis_final,
        'freq_axis': freq_axis_final,
        'true_positions': true_positions_list,
        'antenna_info': antenna_info,
        'dt_meep': dt_meep,
        'run_id': p['run_id'],
        'params': p,
    }

    print(f"\n仿真流水线 '{p['run_id']}' 完成。")
    return result


# =============================================================================
# 便捷函数: 快速启动
# =============================================================================
def quick_sim(
    n_antennas=4,
    scenarios=None,
    antenna_mode='separated',
    enable_random_walk=False,
    num_movements=10,
    movement_step_cm=1.0,
    movement_radius_cm=3.0,
    cir_points=32,
    output_dir='./sim_output',
    **kwargs,
):
    """
    快速启动仿真的便捷函数。

    参数:
        n_antennas (int): 天线数
        scenarios (list[str]): 场景名列表, 默认 ['rear_right_seat']
        antenna_mode (str): 天线模式
        enable_random_walk (bool): 是否启用随机游走
        num_movements (int): 随机游走步数
        movement_step_cm (float): 步长 [cm]
        movement_radius_cm (float): 最大半径 [cm]
        cir_points (int): CIR 点数
        output_dir (str): 输出目录
        **kwargs: 其他参数, 见 DEFAULT_SIM_PARAMS

    返回:
        dict: run_simulation_pipeline 的返回值
    """
    if scenarios is None:
        scenarios = ['rear_right_seat']

    params = {
        'n_antennas': n_antennas,
        'active_scenarios': scenarios,
        'antenna_mode': antenna_mode,
        'enable_random_walk': enable_random_walk,
        'num_movements': num_movements,
        'movement_step_cm': movement_step_cm,
        'movement_radius_cm': movement_radius_cm,
        'cir_points': cir_points,
        'output_dir': output_dir,
        'cache_dir': os.path.join(output_dir, 'mdm_cache'),
        **kwargs,
    }
    return run_simulation_pipeline(params)


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FDTD → CIR 仿真平台')
    parser.add_argument('--antennas', type=int, default=4, help='天线数量')
    parser.add_argument('--mode', default='separated',
                        choices=['separated', 'colocated', 'hybrid_tdm', 'hybrid_tdm_tx_top'],
                        help='天线模式')
    parser.add_argument('--scenarios', nargs='+', default=['rear_right_seat'],
                        help='目标场景')
    parser.add_argument('--random-walk', action='store_true', help='启用随机游走')
    parser.add_argument('--movements', type=int, default=10, help='随机游走步数')
    parser.add_argument('--step-cm', type=float, default=1.0, help='步长 [cm]')
    parser.add_argument('--radius-cm', type=float, default=3.0, help='游走半径 [cm]')
    parser.add_argument('--cir-points', type=int, default=32, help='CIR 点数')
    parser.add_argument('--output', default='./sim_output', help='输出目录')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存')
    parser.add_argument('--multipath', action='store_true', help='启用多径散射体')

    args = parser.parse_args()

    result = quick_sim(
        n_antennas=args.antennas,
        scenarios=args.scenarios,
        antenna_mode=args.mode,
        enable_random_walk=args.random_walk,
        num_movements=args.movements,
        movement_step_cm=args.step_cm,
        movement_radius_cm=args.radius_cm,
        cir_points=args.cir_points,
        output_dir=args.output,
        use_cache=not args.no_cache,
        no_multipath=not args.multipath,
    )

    print(f"\nCIR 快照数: {len(result['cir_snapshots'])}")
    for i, cir in enumerate(result['cir_snapshots']):
        print(f"  快照 {i}: shape = {cir.shape}, dtype = {cir.dtype}")
