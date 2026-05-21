"""
天线阵列参数配置模块

支持 4 种天线模式:
  - separated:      收发分离 (TX 在上, RX 在下)
  - colocated:      收发同址
  - hybrid_tdm:     混合 TDM (全部天线交替收发)
  - hybrid_tdm_tx_top: 混合 TDM (仅顶行发射, 全部接收)

以及:
  - 阵列几何计算 (均匀线阵 ULA)
  - RX 位置随机化
  - 坐标转换 (SI ↔ Meep)
"""

import numpy as np
import meep as mp


# =============================================================================
# 天线模式枚举
# =============================================================================
ANTENNA_MODES = [
    'separated',
    'colocated',
    'hybrid_tdm',
    'hybrid_tdm_tx_top',
]

MODE_DISPLAY_NAMES = {
    'separated':          'Separated TX/RX',
    'colocated':          'Co-located',
    'hybrid_tdm':         'Hybrid TDM',
    'hybrid_tdm_tx_top':  'Hybrid TDM (Top TX Only)',
}


# =============================================================================
# 天线阵列配置
# =============================================================================
def build_antenna_arrays(
    n_antennas,
    fc_si,
    epsilon_r_background=1.0,
    antenna_mode='separated',
    array_center_y=0.0,
    randomize_rx_y_pos=False,
    c0=2.99792458e8,
):
    """
    根据模式构建 TX 和 RX 天线阵列的位置 (SI 单位制)。

    参数:
        n_antennas (int): 天线数量 (每种模式含义不同)
        fc_si (float): 中心频率 [Hz]
        epsilon_r_background (float): 背景介质相对介电常数
        antenna_mode (str): 'separated' | 'colocated' | 'hybrid_tdm' | 'hybrid_tdm_tx_top'
        array_center_y (float): 阵列中心 Y 坐标 [m]
        randomize_rx_y_pos (bool): 是否随机化 RX 的 Y 位置
        c0 (float): 光速 [m/s]

    返回:
        dict: {
            'tx_positions_si':   np.ndarray [n_tx, 3],
            'rx_positions_si':   np.ndarray [n_rx, 3],
            'all_antennas_si':   np.ndarray [n_total, 3] or None,
            'antenna_spacing_si': float,
            'array_center':      np.ndarray [2],
            'mode_display_name': str,
        }
    """
    # 波长 & 半波长间距
    lambda_c_medium_si = c0 / (fc_si * np.sqrt(epsilon_r_background))
    antenna_spacing_si = lambda_c_medium_si / 2

    # X 坐标: 均匀线阵, 中心在 x=0
    array_x_start = -(n_antennas - 1) / 2 * antenna_spacing_si
    x_coords = [array_x_start + i * antenna_spacing_si for i in range(n_antennas)]

    array_center_si = np.array([0.0, array_center_y])

    tx_positions_si = None
    rx_positions_si = None
    all_antennas_si = None

    if antenna_mode == 'separated':
        separation = lambda_c_medium_si
        tx_y = array_center_y + separation / 2.0
        rx_y = array_center_y - separation / 2.0
        tx_positions_si = np.array([[x, tx_y, 0] for x in x_coords])
        rx_positions_si = np.array([[x, rx_y, 0] for x in x_coords])

    elif antenna_mode == 'colocated':
        tx_positions_si = np.array([[x, array_center_y, 0] for x in x_coords])
        rx_positions_si = tx_positions_si.copy()

    elif antenna_mode == 'hybrid_tdm':
        separation = lambda_c_medium_si
        top_y = array_center_y + separation / 2.0
        bottom_y = array_center_y - separation / 2.0
        top_row = np.array([[x, top_y, 0] for x in x_coords])
        bottom_row = np.array([[x, bottom_y, 0] for x in x_coords])
        all_antennas_si = np.vstack([top_row, bottom_row])
        tx_positions_si = all_antennas_si
        rx_positions_si = all_antennas_si

    elif antenna_mode == 'hybrid_tdm_tx_top':
        separation = lambda_c_medium_si
        top_y = array_center_y + separation / 2.0
        bottom_y = array_center_y - separation / 2.0
        top_row = np.array([[x, top_y, 0] for x in x_coords])
        bottom_row = np.array([[x, bottom_y, 0] for x in x_coords])
        all_antennas_si = np.vstack([top_row, bottom_row])
        tx_positions_si = top_row
        rx_positions_si = all_antennas_si

    else:
        raise ValueError(f"未知的 antenna_mode: '{antenna_mode}'。可选: {ANTENNA_MODES}")

    # 随机化 RX 的 Y 位置 (在 [0, λ/2] 范围内)
    if randomize_rx_y_pos:
        half_wavelength = lambda_c_medium_si / 2
        random_offsets = np.random.uniform(0, half_wavelength, rx_positions_si.shape[0])
        rx_positions_si = rx_positions_si.copy()
        rx_positions_si[:, 1] -= random_offsets

    return {
        'tx_positions_si': tx_positions_si,
        'rx_positions_si': rx_positions_si,
        'all_antennas_si': all_antennas_si,
        'antenna_spacing_si': antenna_spacing_si,
        'array_center': array_center_si,
        'mode_display_name': MODE_DISPLAY_NAMES.get(antenna_mode, 'Unknown'),
    }


def si_to_meep(positions_si, m=1.0):
    """
    将 SI 单位的天线位置转换为 Meep 单位。

    参数:
        positions_si (np.ndarray): [n, 3] SI 坐标
        m (float): Meep 长度尺度因子 (默认 1.0 表示 1 Meep unit = 1 m)

    返回:
        list[mp.Vector3]: Meep 坐标列表
    """
    return [mp.Vector3(p[0] / m, p[1] / m) for p in positions_si]


def meep_to_si(positions_meep, m=1.0):
    """将 Meep 单位坐标转换回 SI 单位。"""
    return np.array([[p.x * m, p.y * m, p.z * m] for p in positions_meep])
