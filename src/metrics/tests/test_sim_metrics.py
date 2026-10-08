# -*- coding: utf-8 -*-
"""手算边界自测：gate 边界、tie、missed/false/nohit、OSPA、RMS、分辨率、聚合。

运行::

    & .venv/Scripts/python.exe -B -m unittest discover -s metrics/tests -t metrics
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 06_仿真实验_20260926

import numpy as np  # noqa: E402

from metrics import (  # noqa: E402
    MetricError,
    aggregate,
    evaluate_predictions,
    match_by_gate,
    ospa,
    ospa_squared,
    polar_of,
    wrap_deg,
)

C = 0.30


class GateBoundaryTest(unittest.TestCase):
    def test_exact_gate_is_matched(self):
        r = evaluate_predictions([[0.0, 0.0]], [[0.30, 0.0]])
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["fp"], 0)

    def test_just_over_gate_is_not_matched(self):
        r = evaluate_predictions([[0.0, 0.0]], [[0.30 + 1e-6, 0.0]])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["fp"], 1)

    def test_tolerance_is_1e_minus_12_not_more(self):
        # 0.30 + 5e-13 <= 0.30 + 1e-12 -> 命中
        self.assertEqual(evaluate_predictions([[0.0, 0.0]], [[0.30 + 5e-13, 0.0]])["tp"], 1)
        # 0.30 + 1e-9 > 0.30 + 1e-12 -> 不命中
        self.assertEqual(evaluate_predictions([[0.0, 0.0]], [[0.30 + 1e-9, 0.0]])["tp"], 0)


class MatchingTest(unittest.TestCase):
    def test_tie_break_lexicographic_pairs(self):
        # 两个真值、一个预测均命中；最大匹配数=1；总距离相等 -> 取 (0,0)
        m = match_by_gate([[0.0, 0.0], [0.1, 0.0]], [[0.05, 0.0]])
        self.assertEqual(m["pairs"], [(0, 0, 0.05)])
        self.assertEqual(m["fn"], [1])

    def test_max_cardinality_beats_greedy(self):
        # 贪心最近会先占 P1；最大匹配数应取 2（T0-P1, T1-P0）
        m = match_by_gate([[0.0, 0.0], [0.4, 0.0]], [[0.35, 0.0], [0.02, 0.0]])
        self.assertEqual(m["tp"], 2)
        self.assertEqual([(t, p) for t, p, _d in m["pairs"]], [(0, 1), (1, 0)])
        self.assertAlmostEqual(m["pairs"][0][2], 0.02, places=12)
        self.assertAlmostEqual(m["pairs"][1][2], 0.05, places=12)

    def test_min_total_distance_after_cardinality(self):
        m = match_by_gate([[0.0, 0.0], [0.05, 0.0]], [[0.03, 0.0]])
        self.assertEqual(m["tp"], 1)
        self.assertEqual([(t, p) for t, p, _d in m["pairs"]], [(1, 0)])
        self.assertAlmostEqual(m["pairs"][0][2], 0.02, places=12)
        self.assertEqual(m["fn"], [0])


class CountsTest(unittest.TestCase):
    def test_missed(self):
        r = evaluate_predictions([[0.0, -2.0], [0.0, -3.0]], [[0.1, -2.0]])
        self.assertEqual((r["tp"], r["fn"], r["fp"]), (1, 1, 0))
        self.assertAlmostEqual(r["dr"], 0.5)
        self.assertAlmostEqual(r["ospa2"], 0.05, places=12)

    def test_false(self):
        r = evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0], [0.9, -2.0]])
        self.assertEqual((r["tp"], r["fn"], r["fp"]), (1, 0, 1))
        self.assertAlmostEqual(r["ospa2"], (0.01 + 0.09) / 2, places=12)

    def test_nohit(self):
        r = evaluate_predictions([[0.0, -2.0]], [[0.5, -2.0]])
        self.assertEqual((r["tp"], r["fn"], r["fp"]), (0, 1, 1))
        self.assertAlmostEqual(r["ospa2"], C ** 2, places=12)
        self.assertAlmostEqual(r["ospa"], C, places=12)

    def test_failed_status_is_empty_and_counted(self):
        r = evaluate_predictions([[0.0, -2.0], [0.0, -3.0]], [[0.0, -2.0], [0.0, -3.0]], status="failed")
        self.assertEqual(r["status_used"], "empty")
        self.assertEqual(r["n_predictions"], 0)
        self.assertEqual((r["tp"], r["fn"], r["fp"]), (0, 2, 0))
        self.assertAlmostEqual(r["ospa2"], C ** 2, places=12)
        self.assertFalse(r["success"])


class OspaTest(unittest.TestCase):
    def test_doc_example_1(self):
        t = [[0.0, -2.0], [0.0, -3.0]]
        p = [[0.1, -2.0], [0.1, -3.0]]
        self.assertAlmostEqual(ospa_squared(t, p), 0.01, places=12)
        self.assertAlmostEqual(ospa(t, p), 0.1, places=12)

    def test_doc_example_2_and_3(self):
        t = [[0.0, -2.0], [0.0, -3.0]]
        self.assertAlmostEqual(ospa_squared(t, [[0.1, -2.0]]), 0.05, places=12)
        # 第二条极远（被 c 截断）-> 与只命中一条相同
        self.assertAlmostEqual(ospa_squared(t, [[0.1, -2.0], [1.0, -3.0]]), 0.05, places=12)

    def test_empty_semantics(self):
        self.assertEqual(ospa_squared([], []), 0.0)
        self.assertAlmostEqual(ospa_squared([], [[0.0, -1.0], [0.2, -2.0]]), C ** 2, places=12)
        self.assertAlmostEqual(ospa([], [[0.0, -1.0]]), C, places=12)

    def test_ospa_assignment_is_independent_of_tp_gate(self):
        # TP gate 只命中 1 个；OSPA 仍对两真值做截断指派（两条都配对，远的截断）
        t = [[0.0, -2.0], [0.0, -3.0]]
        p = [[0.1, -2.0], [1.0, -3.0]]
        self.assertEqual(evaluate_predictions(t, p)["tp"], 1)
        self.assertAlmostEqual(ospa_squared(t, p), 0.05, places=12)


class ResolutionTest(unittest.TestCase):
    def test_within_min_gate(self):
        r = evaluate_predictions([[0.0, -2.0], [0.2, -2.0]], [[0.08, -2.0], [0.28, -2.0]],
                                 resolution_distance_m=0.30)
        self.assertIsNotNone(r["resolution"])
        self.assertAlmostEqual(r["resolution"]["gate_m"], 0.10, places=12)
        self.assertTrue(r["resolution"]["exactly_two"])
        self.assertTrue(r["success"])

    def test_tight_d_overrides(self):
        r = evaluate_predictions([[0.0, -2.0], [0.15, -2.0]], [[0.08, -2.0], [0.23, -2.0]],
                                 resolution_distance_m=0.15)
        self.assertAlmostEqual(r["resolution"]["gate_m"], 0.05, places=12)
        self.assertFalse(r["success"])

    def test_requires_exactly_two(self):
        r = evaluate_predictions([[0.0, -2.0], [0.2, -2.0]],
                                 [[0.05, -2.0], [0.25, -2.0], [0.9, -2.0]],
                                 resolution_distance_m=0.30)
        self.assertFalse(r["resolution"]["exactly_two"])
        self.assertFalse(r["success"])


class AggregateTest(unittest.TestCase):
    def test_rms_not_mean(self):
        rows = [
            evaluate_predictions([[0.0, -2.0], [0.0, -3.0]], [[0.1, -2.0], [0.1, -3.0]]),  # ospa 0.1
            evaluate_predictions([[0.0, -2.0], [0.0, -3.0]], [[0.3, -2.0], [0.3, -3.0]]),  # ospa 0.3
        ]
        s = aggregate(rows)
        self.assertAlmostEqual(s["rms_ospa_m"], math.sqrt(0.05), places=12)
        self.assertAlmostEqual(s["mean_ospa_m"], 0.2, places=12)
        self.assertNotAlmostEqual(s["rms_ospa_m"], s["mean_ospa_m"], places=6)
        self.assertEqual(s["n_rows"], 2)

    def test_wrapper_metrics_field_and_failure_row(self):
        base = evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0]])
        rows = [base, {"metrics": base}, {"status": "failed", "n_truth": 1}]
        s = aggregate(rows)
        self.assertEqual(s["n_rows"], 3)
        self.assertEqual(s["n_failed"], 1)
        self.assertEqual(s["n_missing_rows"], 1)
        expect = math.sqrt((0.01 + 0.01 + C ** 2) / 3)
        self.assertAlmostEqual(s["rms_ospa_m"], expect, places=12)

    def test_synthesised_failed_row_keeps_dr_denominator(self):
        # 失败行带 K=2 -> tp=0, fn=2（不得缩成 tp=fn=0）
        s = aggregate([{"status": "failed", "n_truth": 2}])
        self.assertEqual(s["n_rows"], 1)
        self.assertEqual(s["n_truth_total"], 2)
        self.assertEqual((s["tp"], s["fn"], s["fp"]), (0, 2, 0))
        self.assertAlmostEqual(s["dr"], 0.0)
        self.assertAlmostEqual(s["rms_ospa_m"], C, places=12)

    def test_not_applicable_included_in_main_denominator(self):
        ok = evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0]])
        # NA 行按 empty 表示：n_predictions=0, fn=K, OSPA=c（不是照抄 ok 行的数值）
        na = evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0]], status="not_applicable")
        self.assertEqual(na["status_used"], "empty")
        self.assertAlmostEqual(na["ospa2"], C ** 2, places=12)
        s = aggregate([ok, na])
        self.assertEqual(s["n_not_applicable"], 1)
        # 主 summary：两行都计入分母（NA 按 empty 计 FN+OSPA c）
        self.assertEqual(s["n_rows"], 2)
        self.assertAlmostEqual(s["rms_ospa_m"], math.sqrt((0.01 + C ** 2) / 2), places=12)
        # 条件子表可选，单列 NA 排除后的口径
        cond = s["conditional_excluding_not_applicable"]
        self.assertIsNotNone(cond)
        self.assertEqual(cond["n_rows"], 1)
        self.assertAlmostEqual(cond["rms_ospa_m"], 0.1, places=12)

    def test_aggregate_missing_truth_count_raises(self):
        with self.assertRaises(MetricError):
            aggregate([{"status": "failed"}])

    def test_success_rate_general(self):
        good = evaluate_predictions([[0.0, -2.0], [0.2, -2.0]], [[0.05, -2.0], [0.25, -2.0]])
        extra = evaluate_predictions([[0.0, -2.0]], [[0.05, -2.0], [0.9, -2.0]])
        s = aggregate([good, extra])
        self.assertAlmostEqual(s["success_rate"], 0.5, places=12)

    def test_k0_failure_is_not_success(self):
        r = evaluate_predictions([], [[0.1, -2.0]], status="ok")
        self.assertFalse(r["success"])
        r2 = evaluate_predictions([], [], status="failed")
        self.assertFalse(r2["success"])


class ConventionTest(unittest.TestCase):
    def test_wrap_deg(self):
        self.assertAlmostEqual(wrap_deg(180.0), 180.0)
        self.assertAlmostEqual(wrap_deg(-180.0), 180.0)
        self.assertAlmostEqual(wrap_deg(190.0), -170.0)
        self.assertAlmostEqual(wrap_deg(-190.0), 170.0)

    def test_polar(self):
        self.assertAlmostEqual(polar_of(0.0, -1.0)[0], 1.0)
        self.assertAlmostEqual(polar_of(0.0, -1.0)[1], 0.0)
        self.assertAlmostEqual(polar_of(1.0, 0.0)[1], 90.0)

    def test_theta_error_uses_abs_wrap(self):
        r = evaluate_predictions([[0.1, -2.0]], [[-0.1, -2.0]])
        e = r["errors"][0]
        # 两个近轴点落在 ±theta 两侧；e_theta 应为两角度差
        self.assertGreater(e["e_theta"], 0.0)
        self.assertLess(e["e_theta"], 180.0)

    def test_json_serializable(self):
        r = evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0]], resolution_distance_m=0.3)
        json.dumps(r)
        json.dumps(aggregate([r]))

    def test_numpy_arrays_do_not_trigger_ambiguous_truth(self):
        t = np.array([[0.0, -2.0], [0.0, -3.0]])
        p = np.array([[0.1, -2.0]])
        r = evaluate_predictions(t, p)
        self.assertEqual(r["tp"], 1)
        m = match_by_gate(t, p)
        self.assertEqual(m["tp"], 1)


class NamingValidationTest(unittest.TestCase):
    def test_fp_naming(self):
        r = evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0], [0.9, -2.0]])
        # fp_per_estimate 统一为 FP per window（单窗 = fp）；逐预测占比另名
        self.assertEqual(r["fp_per_estimate"], 1)
        self.assertAlmostEqual(r["false_discovery_fraction"], 0.5, places=12)
        s = aggregate([r, r])
        self.assertEqual(s["fp_per_window"], 1.0)
        self.assertEqual(s["fp_per_estimate"], 1.0)
        self.assertAlmostEqual(s["false_discovery_fraction"], 2 / 4, places=12)

    def test_resolution_invalid_d_raises(self):
        for bad in (0.0, -0.3, float("nan"), float("inf")):
            with self.assertRaises(MetricError):
                evaluate_predictions([[0.0, -2.0]], [[0.1, -2.0]], resolution_distance_m=bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
