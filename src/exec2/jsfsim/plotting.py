#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""出图：**只消费已保存结果**，不重跑估计（规范 §9）。

图规范：热力图 `cmap='jet'` + 网格 + 等比例坐标 + 统一坐标范围；主图线性归一化 [0,1]；
真值/预测标记图例完整；dB 图仅诊断用固定 10log10(P/Pmax) 截至 −30 dB，明确只是显示。
样例固定 trial=0、SNR=10 dB；S1–S5 与 E2 的 0.30/0.50/1.00 m 样例全保留。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import protocol as P  # noqa: E402

FIGSIZE = (6.4, 4.8)


def _rows(output_root, runs) -> list:
    base = Path(output_root) / "runs"
    out = []
    if not base.is_dir():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if runs and not any(d.name.startswith(r) for r in runs):
            continue
        f = d / "rows.jsonl"
        if f.is_file():
            out.extend(json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()
                       if line.strip())
    return out


def _summ(output_root) -> dict:
    p = Path(output_root) / "summary_all.json"
    if not p.is_file():
        raise SystemExit("缺 summary_all.json：先运行 summarize（出图只消费已保存结果）")
    return json.loads(p.read_text(encoding="utf-8"))


def _cond(rows, **eq):
    out = rows
    for k, v in eq.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _rmse_dr_curves(summ: dict, out_dir: Path):
    rows = summ["by_condition"]
    for layer in ("A", "B"):
        for task in ("E1_A" if layer == "A" else "E1_B",):
            sub = [r for r in rows if r["task"] == task]
            if not sub:
                continue
            fig, ax = plt.subplots(figsize=FIGSIZE)
            for layout in sorted({r["layout"] for r in sub}):
                s = sorted([r for r in sub if r["layout"] == layout], key=lambda r: r["snr_db"])
                x = [r["snr_db"] for r in s]
                y = [r["rmse_pooled_m"] for r in s]
                lo = [r["rmse_ci95_lo"] for r in s]
                hi = [r["rmse_ci95_hi"] for r in s]
                ax.plot(x, y, marker="o", label=layout)
                ax.fill_between(x, lo, hi, alpha=0.15)
            ax.set_xlabel("SNR (dB)"); ax.set_ylabel("pooled position RMSE (m)")
            ax.set_title("%s position RMSE vs SNR (pooled, trial-whole bootstrap 95%%)" % task)
            ax.grid(True); ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / ("V_B_rmse_vs_snr_%s.png" % layer), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=FIGSIZE)
            for layout in sorted({r["layout"] for r in sub}):
                s = sorted([r for r in sub if r["layout"] == layout], key=lambda r: r["snr_db"])
                ax.plot([r["snr_db"] for r in s], [r["dr"] for r in s], marker="s", label=layout)
            ax.set_xlabel("SNR (dB)"); ax.set_ylabel("DR (0.30 m gate)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("%s detection rate vs SNR (0.30 m gate, independent matching)" % task)
            ax.grid(True); ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / ("V_B_dr_vs_snr_%s.png" % layer), dpi=150)
            plt.close(fig)


def _e2_curves(summ: dict, out_dir: Path):
    rows = summ["e2_spacing_table"]
    for layer, task in (("A", "E2_A"), ("B", "E2_B")):
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        snrs = sorted({r["snr_db"] for r in sub})
        fig, axes = plt.subplots(1, len(snrs), figsize=(5.0 * len(snrs), 4.2), squeeze=False)
        for j, snr in enumerate(snrs):
            ax = axes[0][j]
            for direction in sorted({r["direction"] for r in sub}):
                s = sorted([r for r in sub if r["direction"] == direction and r["snr_db"] == snr],
                           key=lambda r: r["d_nominal_m"])
                if not s:
                    continue
                ax.plot([r["d_nominal_m"] for r in s], [r["success_rate"] for r in s],
                        marker="o", label=direction)
            ax.axvline(min(0.10, 0.15 / 3.0), color="k", ls=":", lw=1,
                       label="gate min(0.10,d/3) at d=0.15")
            ax.set_xlabel("target separation d (m)")
            ax.set_ylabel("resolution success rate")
            ax.set_title("%s @ %.0f dB" % (task, snr))
            ax.set_ylim(-0.02, 1.02); ax.grid(True)
            if j == 0:
                ax.legend(fontsize=8)
        fig.suptitle("E2 multi-target separability: success rate vs separation")
        fig.tight_layout()
        fig.savefig(out_dir / ("V_C_resolution_vs_d_%s.png" % layer), dpi=150)
        plt.close(fig)


def _e4_curves(summ: dict, out_dir: Path):
    rows = summ["e4_L_table"]
    for layer, task in (("A", "E4_A"), ("B", "E4_B")):
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for snr in sorted({r["snr_db"] for r in sub}):
            s = sorted([r for r in sub if r["snr_db"] == snr], key=lambda r: r["L"])
            ax.plot([r["L"] for r in s], [r["rmse_pooled_m"] for r in s], marker="o",
                    label="%.0f dB" % snr)
        ax.set_xscale("log", base=2)
        ax.set_xticks(P.E4_L_LIST); ax.set_xticklabels([str(v) for v in P.E4_L_LIST])
        ax.set_xlabel("snapshots L (same clean prefix)")
        ax.set_ylabel("pooled position RMSE (m)")
        ax.set_title("%s snapshot sensitivity (fixed S2)" % task)
        ax.grid(True); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
        fig.savefig(out_dir / ("V_D_rmse_vs_L_%s.png" % layer), dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for snr in sorted({r["snr_db"] for r in sub}):
            s = sorted([r for r in sub if r["snr_db"] == snr], key=lambda r: r["L"])
            ax.plot([r["L"] for r in s], [r["dr"] for r in s], marker="s", label="%.0f dB" % snr)
        ax.set_xscale("log", base=2)
        ax.set_xticks(P.E4_L_LIST); ax.set_xticklabels([str(v) for v in P.E4_L_LIST])
        ax.set_xlabel("snapshots L"); ax.set_ylabel("DR (0.30 m gate)")
        ax.set_ylim(-0.02, 1.02); ax.grid(True); ax.legend(fontsize=8, ncol=2)
        ax.set_title("%s detection vs L" % task); fig.tight_layout()
        fig.savefig(out_dir / ("V_D_dr_vs_L_%s.png" % layer), dpi=150)
        plt.close(fig)


def _layouts_figure(out_dir: Path):
    """S1–S5 布局图（规范 §13 V-A）。"""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for name, pts in P.S1_S5_LAYOUTS.items():
        xy = [P.polar_to_xy(r, t) for r, t in pts]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "o", ms=8, label=name)
        for p, (r, t) in zip(xy, pts):
            ax.annotate("(%.1f m, %.0f°)" % (r, t), p, textcoords="offset points",
                        xytext=(6, 4), fontsize=7)
    ax.plot([0.0], [0.0], "k^", ms=10, label="array (1Tx4Rx)")
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-3.3, 0.2)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Controlled layouts S1-S5 (exact polar, no legacy shift)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "V_A_layouts_S1_S5.png", dpi=150); plt.close(fig)


def _heatmaps(output_root, out_dir: Path):
    """样例热图（trial=0、SNR=10 dB；含 S1–S5 与 E2 三档间距）。"""
    base = Path(output_root) / "runs"
    if not base.is_dir():
        return []
    made = []
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        sdir = run_dir / "samples"
        if not sdir.is_dir():
            continue
        for npz in sorted(sdir.glob("*.npz")):
            meta_p = npz.with_suffix(".json")
            if not meta_p.is_file():
                continue
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            with np.load(npz, allow_pickle=False) as z:
                keys = z.files
                if "spectrum" not in keys:
                    continue
                spec = np.asarray(z["spectrum"], dtype=float)
                gx = np.asarray(z["grid_x"], dtype=float)
                gy = np.asarray(z["grid_y"], dtype=float)
                preds = np.asarray(z["predictions"], dtype=float).reshape(-1, 3)
                truth = np.asarray(z["truth_xy"], dtype=float).reshape(-1, 2)
            denom = float(np.max(np.abs(spec))) if spec.size else 0.0
            norm = (spec / denom) if denom > 0 else spec
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            im = ax.pcolormesh(gx, gy, norm, cmap="jet", shading="auto", vmin=0.0, vmax=1.0)
            ax.plot(truth[:, 0], truth[:, 1], "w*", ms=14, mec="k", label="truth")
            ax.plot(preds[:, 0], preds[:, 1], "ko", ms=7, mfc="none", label="prediction")
            ax.set_xlim(P.GRID_X_MIN, P.GRID_X_MAX); ax.set_ylim(P.GRID_Y_MIN, P.GRID_Y_MAX)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_title("%s  L=%d  SNR=%.0f dB  trial=0\nlinear normalized pseudo-spectrum"
                         % (meta["task_id"], meta["L"], meta["snr_db"]), fontsize=9)
            ax.grid(True, color="k", alpha=0.25, lw=0.4)
            ax.legend(fontsize=8, loc="upper right")
            fig.colorbar(im, ax=ax, shrink=0.85, label="normalized P/Pmax")
            fig.tight_layout()
            name = "V_B_spectrum_%s.png" % meta["task_id"].replace("/", "_")
            fig.savefig(out_dir / name, dpi=150)
            plt.close(fig)
            made.append(name)
    return made


def _q4_figures(output_root, out_dir: Path):
    q4 = Path(output_root) / "q4"
    doc_p = q4 / "q4_cross_domain.json"
    if not doc_p.is_file():
        return []
    doc = json.loads(doc_p.read_text(encoding="utf-8"))
    made = []
    # 配置等权汇总条形图
    cfgs = [(c, v) for c, v in doc["per_config"].items()
            if isinstance(v, dict) and v.get("spectral_rho_mean") is not None]
    if cfgs:
        cfgs.sort(key=lambda kv: kv[0])
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.bar([c for c, _ in cfgs], [v["spectral_rho_mean"] for _, v in cfgs],
               yerr=[v.get("spectral_rho_sd") or 0.0 for _, v in cfgs], capsize=3)
        ax.set_ylabel("mean Pearson rho (per config, by env)")
        ax.set_ylim(-0.1, 1.02); ax.grid(True, axis="y")
        ax.set_title("Q4 simulation-to-hardware spectral correlation (205 windows, fixed list)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout(); fig.savefig(out_dir / "VI_Q4_spectral_rho_by_config.png", dpi=150)
        plt.close(fig); made.append("VI_Q4_spectral_rho_by_config.png")
    return made


def plot_all(output_root) -> int:
    root = Path(output_root)
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = _summ(root)
    _layouts_figure(out_dir)
    _rmse_dr_curves(summ, out_dir)
    _e2_curves(summ, out_dir)
    _e4_curves(summ, out_dir)
    heats = _heatmaps(root, out_dir)
    q4 = _q4_figures(root, out_dir)
    names = sorted(p.name for p in out_dir.glob("*.png"))
    print(json.dumps({"figures": names, "n_heatmaps": len(heats), "q4_figures": q4},
                     ensure_ascii=False, indent=1))
    return 0


__all__ = ["plot_all"]
