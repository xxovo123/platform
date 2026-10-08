# -*- coding: utf-8 -*-
"""Q4 子包：硬件清单 / 可移植 CIR 包导出 / 跨域比较。

公共入口::

    from q4 import build_manifest, export_from_cache, export_from_bin, run_compare

统一 CLI：``python -B q4/q4_cli.py {inventory,export,compare,demo} ...``
"""

from .q4_common import (cross_domain_peak_delta, complex_rho, envelope_rho, geometry_of,
                        mid_index, rms_envelope, select_virtual_channels, wrap_deg)
from .q4_batch import (hw_recordings, hw_truth_windows, plan_batch, reconstruct_hw_predictions,
                       run_batch)
from .q4_compare import compare_window, demo_synthetic, load_predictions, run_compare, summarize
from .q4_export import export_from_bin, export_from_cache
from .q4_eval import build_q4_case, deps_ready, eval_clean, eval_plan, run_methods, select_channels
from .q4_inventory import build_manifest, scan_recordings, write_manifest
from .q4_scenes import Q4_GEOMETRY_OVERRIDE, Q4_PROVENANCE, export_scene_specs, generate

__all__ = [
    "build_manifest", "scan_recordings", "write_manifest",
    "export_from_cache", "export_from_bin",
    "run_compare", "compare_window", "summarize", "load_predictions", "demo_synthetic",
    "plan_batch", "run_batch", "hw_recordings", "hw_truth_windows", "reconstruct_hw_predictions",
    "export_scene_specs", "generate", "Q4_GEOMETRY_OVERRIDE", "Q4_PROVENANCE",
    "eval_plan", "eval_clean", "deps_ready", "build_q4_case", "select_channels", "run_methods",
    "cross_domain_peak_delta", "complex_rho", "envelope_rho", "rms_envelope",
    "select_virtual_channels", "mid_index", "geometry_of", "wrap_deg",
]
