"""
FDTD → CIR 通用仿真平台

模块结构:
  fdtd_simulator.py   — FDTD 仿真核心 (Meep), MDM 数据采集
  signal_converter.py — 通带 MDM → 基带 CIR 转换
  antenna_config.py   — 天线阵列参数配置 (4 种模式)
  data_manager.py     — 数据缓存与导出
  simulation_config.py — 统一仿真入口与参数管理
"""

from .antenna_config import (
    ANTENNA_MODES,
    MODE_DISPLAY_NAMES,
    build_antenna_arrays,
    si_to_meep,
    meep_to_si,
)

from .signal_converter import (
    generate_reference_pulse_baseband,
    prepare_baseband_data,
)

from .fdtd_simulator import (
    blackmann_harris_derivative_pulse,
    make_car_interior_geometry,
    generate_random_walk_path,
    make_moving_targets,
    collect_mdm_data,
    collect_mdm_data_hybrid_tdm,
    collect_mdm_sequence,
)

from .data_manager import (
    make_cache_filename,
    save_mdm_cache,
    load_mdm_cache,
    save_cir_data,
    make_run_id,
    export_simulation_summary,
)

from .simulation_config import (
    DEFAULT_SIM_PARAMS,
    PRESET_SCENARIOS,
    register_scenario,
    run_simulation_pipeline,
    quick_sim,
)
