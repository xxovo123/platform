"""
FDTD 仿真核心模块 (基于 Meep)

功能:
  1. 定义场景几何体 (汽车内部: 金属柱、座椅)
  2. 定义移动目标 (随机游走路径)
  3. 运行 Meep FDTD 仿真, 采集 MDM (Multi-Dimensional Matrix) 通带信号

无算法内容, 仅负责电磁仿真与信号采集。
"""

import meep as mp
import numpy as np
import time


# =============================================================================
# 发射脉冲: Blackman-Harris 导数脉冲
# =============================================================================
def blackmann_harris_derivative_pulse(t, fc, duration):
    """
    Blackman-Harris 窗包络的导数脉冲。

    参数:
        t (float or np.ndarray): 时间
        fc (float): 中心频率 [Meep 单位]
        duration (float): 脉冲持续时间 [Meep 单位]

    返回:
        float or np.ndarray: 脉冲波形值
    """
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    t0 = duration / 2.0
    t_rel = t - t0 + duration / 2.0
    in_pulse = np.where((t >= 0) & (t <= duration), 1, 0)
    arg = 2 * np.pi * t_rel / duration

    w_t = (a0 - a1 * np.cos(arg) + a2 * np.cos(2 * arg) - a3 * np.cos(3 * arg)) * in_pulse
    dw_dt = (2 * np.pi / duration) * (a1 * np.sin(arg) - 2 * a2 * np.sin(2 * arg) + 3 * a3 * np.sin(3 * arg)) * in_pulse

    carrier = np.cos(2 * np.pi * fc * (t - t0))
    carrier_deriv = -2 * np.pi * fc * np.sin(2 * np.pi * fc * (t - t0))

    return dw_dt * carrier + w_t * carrier_deriv


# =============================================================================
# 场景几何体定义
# =============================================================================
def make_car_interior_geometry(
    resolution,
    car_x_min=-1.0,
    car_x_max=1.0,
    pillar_length=0.50,
    include_pillars=True,
    include_seats=True,
    pillar_material=None,
    seat_material=None,
):
    """
    创建汽车内部场景几何体 (金属柱 + 座椅)。

    参数:
        resolution (float): Meep 分辨率 [pixels/Meep unit]
        car_x_min, car_x_max (float): 车身 X 范围 [Meep 单位]
        pillar_length (float): 柱子长度 [Meep 单位]
        include_pillars (bool): 是否包含 A/B/C 柱
        include_seats (bool): 是否包含座椅
        pillar_material: Meep 材料, 默认 mp.metal
        seat_material: Meep 材料, 默认 mp.metal

    返回:
        list[mp.Block]: 几何体列表
    """
    if pillar_material is None:
        pillar_material = mp.metal
    if seat_material is None:
        seat_material = mp.metal

    geometry = []

    if include_pillars:
        pillar_thickness = 12 * (1 / resolution)
        b_pillar_y, a_pillar_y, c_pillar_y = 0.0, 1.0, -1.0
        left_x = car_x_min - pillar_thickness / 2
        right_x = car_x_max + pillar_thickness / 2

        for y_center in [a_pillar_y, b_pillar_y, c_pillar_y]:
            geometry.append(mp.Block(
                center=mp.Vector3(left_x, y_center),
                size=mp.Vector3(pillar_thickness, pillar_length),
                material=pillar_material
            ))
            geometry.append(mp.Block(
                center=mp.Vector3(right_x, y_center),
                size=mp.Vector3(pillar_thickness, pillar_length),
                material=pillar_material
            ))

    if include_seats:
        seat_size = mp.Vector3(0.55, 0.2)
        seat_centers = {
            'front_left':  mp.Vector3(-0.5, 0.5),
            'front_right': mp.Vector3(0.5, 0.5),
            'rear_left':   mp.Vector3(-0.5, -0.8),
            'rear_right':  mp.Vector3(0.5, -0.8),
        }
        for pos in seat_centers.values():
            geometry.append(mp.Block(center=pos, size=seat_size, material=seat_material))

    return geometry


# =============================================================================
# 移动目标定义
# =============================================================================
def generate_random_walk_path(
    start_pos,
    num_steps,
    step_size_m,
    max_radius_m,
):
    """
    为单个移动目标生成随机游走路径。

    参数:
        start_pos (mp.Vector3): 起始位置
        num_steps (int): 移动步数
        step_size_m (float): 每步步长 [米]
        max_radius_m (float): 距起始点的最大半径 [米]

    返回:
        list[mp.Vector3]: 路径 (长度 = num_steps + 1, 含起始点)
    """
    path = [start_pos]
    current_pos = start_pos

    for _ in range(num_steps):
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            candidate = mp.Vector3(
                current_pos.x + step_size_m * np.cos(angle),
                current_pos.y + step_size_m * np.sin(angle),
                current_pos.z,
            )
            if (candidate - start_pos).norm() <= max_radius_m:
                path.append(candidate)
                current_pos = candidate
                break

    return path


def make_moving_targets(paths, step_idx, radius_si, lambda_c_si, material=None):
    """
    根据所有路径的指定步创建移动目标圆柱体列表。

    参数:
        paths (list[list[mp.Vector3]]): 每个目标的完整路径
        step_idx (int): 当前步索引
        radius_si (float): 圆柱体半径 [米] (通常为 1.5λ)
        lambda_c_si (float): 中心频率波长 [米]
        material: Meep 材料

    返回:
        list[mp.Cylinder]: 当前步的移动目标几何体
    """
    if material is None:
        material = mp.metal

    targets = []
    for path in paths:
        pos = path[step_idx]
        targets.append(mp.Cylinder(
            center=pos,
            radius=radius_si,
            material=material,
        ))
    return targets


# =============================================================================
# 核心: FDTD 数据采集
# =============================================================================
def collect_mdm_data(
    geometry,
    cell_size,
    resolution,
    pml_layers,
    background_medium,
    tx_positions_meep,
    rx_positions_meep,
    fc_meep,
    duration_meep,
    run_time_meep,
):
    """
    运行 Meep FDTD 仿真, 采集 MDM (Multi-Dimensional Matrix) 通带信号。

    对每个发射天线依次运行一次仿真, 在所有接收天线位置记录 Ez 场。

    参数:
        geometry (list): Meep 几何体列表
        cell_size (mp.Vector3): 仿真域尺寸
        resolution (float): 分辨率 [pixels/Meep unit]
        pml_layers (list): PML 吸收边界层
        background_medium: 背景介质
        tx_positions_meep (list[mp.Vector3]): 发射天线位置
        rx_positions_meep (list[mp.Vector3]): 接收天线位置
        fc_meep (float): 中心频率 [Meep 单位]
        duration_meep (float): 脉冲持续时间 [Meep 单位]
        run_time_meep (float): 仿真总时长 [Meep 单位]

    返回:
        tuple: (mdm_data, dt_meep)
            - mdm_data (np.ndarray): [n_rx, n_tx, n_timesteps] 通带信号
            - dt_meep (float): Meep 时间步长
    """
    n_tx = len(tx_positions_meep)
    n_rx = len(rx_positions_meep)

    # 获取时间步长
    sim_temp = mp.Simulation(cell_size=cell_size, resolution=resolution)
    sim_temp.init_sim()
    dt_meep = sim_temp.fields.dt
    num_steps = int(run_time_meep / dt_meep)
    del sim_temp

    mdm_data = np.zeros((n_rx, n_tx, num_steps))

    # 创建脉冲源
    bh_pulse_func = lambda t: blackmann_harris_derivative_pulse(t, fc_meep, duration_meep)
    custom_src = mp.CustomSource(src_func=bh_pulse_func, end_time=duration_meep)

    for tx_idx in range(n_tx):
        if n_rx == 0:
            continue

        print(f"    TX {tx_idx + 1}/{n_tx} ...")

        sim = mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            boundary_layers=pml_layers,
            geometry=geometry,
            default_material=background_medium,
            dimensions=2,
        )
        sim.sources = [mp.Source(custom_src, component=mp.Ez, center=tx_positions_meep[tx_idx])]

        recorded_fields = []

        def record_ez(sim_instance):
            fields = [sim_instance.get_field_point(mp.Ez, pos) for pos in rx_positions_meep]
            recorded_fields.append(fields)

        sim.run(mp.at_every(dt_meep, record_ez), until=run_time_meep)

        if recorded_fields:
            recorded_arr = np.array(recorded_fields).T  # [n_rx, n_recorded]
            slice_len = min(recorded_arr.shape[1], mdm_data.shape[2])
            mdm_data[:, tx_idx, :slice_len] = recorded_arr[:, :slice_len]

        del sim, recorded_fields

    return mdm_data, dt_meep


# =============================================================================
# 混合 TDM 模式下的 MDM 采集
# =============================================================================
def collect_mdm_data_hybrid_tdm(
    geometry,
    cell_size,
    resolution,
    pml_layers,
    background_medium,
    all_antennas_si,
    num_tx_antennas,
    fc_meep,
    duration_meep,
    run_time_meep,
):
    """
    混合 TDM 模式下的 MDM 采集。

    每个天线依次发射, 其余天线全部接收 (排除自身)。

    参数:
        all_antennas_si (np.ndarray): [n_total, 3] 全部天线位置 (SI)
        num_tx_antennas (int): 发射天线数量
        其余同 collect_mdm_data

    返回:
        tuple: (mdm_data, dt_meep)
    """
    num_total = all_antennas_si.shape[0]

    sim_temp = mp.Simulation(cell_size=cell_size, resolution=resolution)
    sim_temp.init_sim()
    dt_meep = sim_temp.fields.dt
    num_samples = int(run_time_meep / dt_meep)
    del sim_temp

    mdm_snapshot = np.zeros((num_total, num_total, num_samples))

    for i in range(num_tx_antennas):
        current_tx = all_antennas_si[i]
        rx_indices = list(range(num_total))
        rx_indices.pop(i)  # 排除自身
        rx_positions = all_antennas_si[rx_indices]

        tx_meep = [mp.Vector3(current_tx[0], current_tx[1])]
        rx_meep = [mp.Vector3(p[0], p[1]) for p in rx_positions]

        partial_mdm, _ = collect_mdm_data(
            geometry, cell_size, resolution, pml_layers, background_medium,
            tx_meep, rx_meep, fc_meep, duration_meep, run_time_meep,
        )
        mdm_snapshot[rx_indices, i, :] = partial_mdm[:, 0, :]

    return mdm_snapshot, dt_meep


# =============================================================================
# 完整 MDM 序列采集 (多步 / 随机游走)
# =============================================================================
def collect_mdm_sequence(
    geometry_static,
    cell_size,
    resolution,
    pml_layers,
    background_medium,
    antenna_info,
    all_scenario_paths,
    fc_meep,
    duration_meep,
    run_time_meep,
    lambda_c_si,
    m=1.0,
):
    """
    采集多步 MDM 序列 (用于随机游走场景)。

    参数:
        geometry_static (list): 静态几何体 (不含移动目标)
        antenna_info (dict): build_antenna_arrays 的返回值
        all_scenario_paths (list[list[mp.Vector3]]): 每个目标的随机游走路径
        m (float): Meep 长度尺度因子

    返回:
        tuple: (mdm_data_sequence, dt_meep)
            - mdm_data_sequence (list[np.ndarray]): 每步的 MDM 快照
            - dt_meep (float): Meep 时间步长
    """
    antenna_mode = antenna_info.get('_antenna_mode', 'separated')
    tx_si = antenna_info['tx_positions_si']
    rx_si = antenna_info['rx_positions_si']
    all_si = antenna_info.get('all_antennas_si')

    num_movements = len(all_scenario_paths[0]) - 1
    target_radius_si = lambda_c_si * 1.5

    mdm_sequence = []
    dt_meep = None

    for step_idx in range(num_movements + 1):
        print(f"\n  [Step {step_idx + 1}/{num_movements + 1}]")

        # 构建当前步的完整几何体
        moving_targets = make_moving_targets(
            all_scenario_paths, step_idx, target_radius_si, lambda_c_si
        )
        geometry_current = geometry_static + moving_targets

        if antenna_mode in ['separated', 'colocated']:
            tx_meep = [mp.Vector3(p[0], p[1]) for p in tx_si]
            rx_meep = [mp.Vector3(p[0], p[1]) for p in rx_si]
            snapshot, dt = collect_mdm_data(
                geometry_current, cell_size, resolution, pml_layers,
                background_medium, tx_meep, rx_meep,
                fc_meep, duration_meep, run_time_meep,
            )
            if dt_meep is None:
                dt_meep = dt

        elif antenna_mode in ['hybrid_tdm', 'hybrid_tdm_tx_top']:
            if antenna_mode == 'hybrid_tdm':
                num_tx = all_si.shape[0]
            else:
                num_tx = all_si.shape[0] // 2

            snapshot, dt = collect_mdm_data_hybrid_tdm(
                geometry_current, cell_size, resolution, pml_layers,
                background_medium, all_si, num_tx,
                fc_meep, duration_meep, run_time_meep,
            )
            if dt_meep is None:
                dt_meep = dt

        else:
            raise ValueError(f"未知天线模式: {antenna_mode}")

        mdm_sequence.append(snapshot)

    return mdm_sequence, dt_meep
