# -*- coding: utf-8 -*-
"""B0 同模型数值正确性校验包（标准全二维联合 MUSIC，独立列向量实现）。

模块：
  cfr_replay      本地真实 CIR 重放（独立帧解析 + 背景 EMA + 128 帧窗）
  geometry        虚拟通道映射（独立实现，运行时与源工程口径比对）
  standard_music  独立标准全二维 MUSIC（R = X^T X^*/L、a^H E_n E_n^H a）
  comparison      逐格谱差 / 归一化谱相关 / 峰集一致性 / 共轭关系核对

边界：B0 只做同模型数值正确性（与硬件 radar_algorithm 在同一输入上逐格比对），
不是任何单篇论文方法的复现、不是性能基线、不是 FDTD/旧仿真结论的验证。
"""

from .cfr_replay import (  # noqa: F401
    FrameParser, ReplayError, cropped_snapshots, load_meta, replay_window, replay_windows,
)
from .geometry import virtual_pairs  # noqa: F401
from .standard_music import (  # noqa: F401
    MusicConfig, MusicError, build_observation_matrix, config_from_meta, frequency_axis,
    sample_covariance, standard_music_spectrum, steering_matrix,
)
from .comparison import (  # noqa: F401
    conjugation_check, music_form_equivalence_check, peak_set_comparison, spectrum_comparison,
)
