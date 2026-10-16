#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic redraw of the JSF-MUSIC algorithm flowchart.

The four-stage diagram of the earlier pipeline version is rebuilt from
primitives only: isometric tensor stacks, matrices, block arrows and vector
text. No raster image and no page of the previous PDF is embedded, and every
glyph stays live text in the SVG/PDF output.

Stage content follows the revised manuscript (paper/jsen_update.tex):
  1  Data acquisition & FFT:  CIR  N_rx x N_taps x L  ->  H  N_rx x N_freq x L
  2  Channel-frequency stacking: H_l (N_rx x N_freq) -> x_l in C^M -> X in C^{M x L}
  3  Covariance & forward-backward averaging:
        R^ = X X^H / L,  R_fb = (R^ + J R^* J) / 2,
        R~ = R_fb + 1e-5 tr(R_fb) I   (loaded covariance, as implemented)
  4  EVD at the given K_alg = N_true -> U_s / U_n -> P_JSF-MUSIC(r, theta)
        -> unified peak picking -> paired (r^, theta^)

Outputs
    paper/algo_flowchart_redrawn.pdf    vector, editable text
    paper/algo_flowchart_redrawn.svg    vector, editable text
    paper/algo_flowchart_redrawn.png    600 dpi preview
    preview/flowchart_redrawn.png       copy of the preview
"""

import os
import re
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrow, Polygon, Rectangle

# ----------------------------------------------------------------- style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 200,
        "savefig.dpi": 600,
    }
)

FIG_W, FIG_H = 7.10, 2.82          # inches; aspect 2.52 : 1
UPI = 100.0                        # layout units per inch
W, H = FIG_W * UPI, FIG_H * UPI    # 710 x 282

INK = "#111111"
PANEL_FC, PANEL_EC = "#F4F4F4", "#BFBFBF"
BOX_FC, BOX_EC = "#CBDAE9", "#7E9CB8"
CL = ["#0050EF", "#008A00", "#F0A30A", "#E51400"]   # blue, green, orange, red
GRIDC = "#D9D9D9"
CUBE_FC, CUBE_EC, CUBE_G = "#DCE9F8", "#2E6DA4", "#93B9DC"

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()

# ----------------------------------------------------------------- helpers


def txt(x, y, s, size=6.5, ha="center", va="center", color=INK, rot=0.0,
        weight="normal", style="normal", z=8):
    ax.text(
        x, y, s, fontsize=size, ha=ha, va=va, color=color, rotation=rot,
        rotation_mode="anchor", weight=weight, style=style, zorder=z,
    )


def rect(x0, y0, w, h, fc, ec=INK, lw=0.6, z=3):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec=ec, lw=lw, zorder=z))


def block_arrow(x0, x1, y, w=3.2, hw=8.6, hl=8.6, color=INK, z=6):
    """Chunky solid arrow in the style of the original figure."""
    ax.add_patch(
        FancyArrow(
            x0, y, x1 - x0, 0.0, width=w, head_width=hw, head_length=hl,
            length_includes_head=True, fc=color, ec="none", zorder=z,
        )
    )


def thin_arrow(p0, p1, lw=0.7, color=INK, z=6, scale=5.0):
    ax.annotate(
        "", xy=p1, xytext=p0, zorder=z,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, shrinkA=0.0, shrinkB=0.0,
            mutation_scale=scale,
        ),
    )


def bracket(x0, y0, w, h, tick=5.5, lw=1.0, color=INK, z=6):
    """Square brackets on the left and right of a matrix block."""
    x1, y1 = x0 + w, y0 + h
    for xs, sgn in ((x0, 1.0), (x1, -1.0)):
        ax.add_line(Line2D([xs, xs], [y0, y1], color=color, lw=lw, zorder=z))
        ax.add_line(Line2D([xs, xs + sgn * tick], [y0, y0], color=color,
                           lw=lw, zorder=z))
        ax.add_line(Line2D([xs, xs + sgn * tick], [y1, y1], color=color,
                           lw=lw, zorder=z))


def cell_grid(x0, y0, w, h, ncol, nrow, colours, lw=0.4, edge="#1A1A1A", z=4):
    """Coloured cell matrix; rows are grouped so that colour index == group."""
    cw, ch = w / ncol, h / nrow
    for r in range(nrow):
        for c in range(ncol):
            rect(x0 + c * cw, y0 + r * ch, cw, ch, colours[r], ec="none",
                 lw=0.0, z=z)
    for c in range(ncol + 1):
        ax.add_line(Line2D([x0 + c * cw] * 2, [y0, y0 + h], color=edge,
                           lw=lw, zorder=z + 1))
    for r in range(nrow + 1):
        ax.add_line(Line2D([x0, x0 + w], [y0 + r * ch] * 2, color=edge,
                           lw=lw, zorder=z + 1))
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=edge, lw=0.7,
                           zorder=z + 2))


def band_rows(nrow, palette):
    """Row colour list: equal bands over the palette (top band = palette[0])."""
    n = len(palette)
    return [palette[min(n - 1, (nrow - 1 - r) * n // nrow)] for r in range(nrow)]


ISO_X, ISO_Y = 0.8660254, 0.5


def projector(ox, oy, s):
    def p(u, v, w):
        return (ox + (u - v) * ISO_X * s, oy - (u + v) * ISO_Y * s + w * s)

    return p


def cir_cube(p, U, V, Wd, nu, nv, nw, fc=CUBE_FC, ec=CUBE_EC, gc=CUBE_G, z=3):
    """Transparent isometric lattice box (the CIR tensor)."""
    faces = [
        [p(0, 0, Wd), p(U, 0, Wd), p(U, V, Wd), p(0, V, Wd)],      # top
        [p(U, 0, 0), p(U, V, 0), p(U, V, Wd), p(U, 0, Wd)],        # u = U
        [p(0, V, 0), p(U, V, 0), p(U, V, Wd), p(0, V, Wd)],        # v = V
    ]
    for f in faces:
        ax.add_patch(Polygon(f, closed=True, fc=fc, ec=ec, lw=0.6,
                             alpha=0.9, zorder=z))
    # lattice lines on the three visible faces
    for i in range(1, nu):
        u = U * i / nu
        ax.add_line(Line2D(*zip(p(u, 0, Wd), p(u, V, Wd)), color=gc, lw=0.35,
                           zorder=z + 1))
        ax.add_line(Line2D(*zip(p(u, V, 0), p(u, V, Wd)), color=gc, lw=0.35,
                           zorder=z + 1))
    for j in range(1, nv):
        v = V * j / nv
        ax.add_line(Line2D(*zip(p(0, v, Wd), p(U, v, Wd)), color=gc, lw=0.35,
                           zorder=z + 1))
        ax.add_line(Line2D(*zip(p(U, v, 0), p(U, v, Wd)), color=gc, lw=0.35,
                           zorder=z + 1))
    for k in range(1, nw):
        w = Wd * k / nw
        ax.add_line(Line2D(*zip(p(0, 0, w), p(U, 0, w)), color=gc, lw=0.35,
                           zorder=z + 1))
        ax.add_line(Line2D(*zip(p(0, V, w), p(U, V, w)), color=gc, lw=0.35,
                           zorder=z + 1))
        ax.add_line(Line2D(*zip(p(U, 0, w), p(U, V, w)), color=gc, lw=0.35,
                           zorder=z + 1))
    # silhouette edges
    for a, b in [
        ((0, 0, Wd), (U, 0, Wd)), ((0, 0, Wd), (0, V, Wd)),
        ((U, 0, Wd), (U, V, Wd)), ((0, V, Wd), (U, V, Wd)),
        ((U, V, Wd), (U, V, 0)), ((0, V, Wd), (0, V, 0)),
        ((U, 0, Wd), (U, 0, 0)), ((0, V, 0), (U, V, 0)),
        ((U, V, 0), (U, 0, 0)), ((0, 0, 0), (0, V, 0)),
        ((0, 0, 0), (U, 0, 0)), ((0, 0, 0), (0, 0, Wd)),
    ]:
        ax.add_line(Line2D(*zip(p(*a), p(*b)), color=ec, lw=0.6, zorder=z + 2))


def plate_stack(p, U, V, Wd, nslab, dv, colours, nu, nw, ec=INK, z=3):
    """Stacked isometric slabs (the FFT output tensor)."""
    for k in range(nslab):
        v = k * dv
        quad = [p(0, v, 0), p(U, v, 0), p(U, v, Wd), p(0, v, Wd)]
        ax.add_patch(Polygon(quad, closed=True, fc=colours[k], ec=ec, lw=0.6,
                             zorder=z + k))
        for i in range(1, nu):
            u = U * i / nu
            ax.add_line(Line2D(*zip(p(u, v, 0), p(u, v, Wd)), color=ec, lw=0.3,
                               zorder=z + k + 0.5))
        for j in range(1, nw):
            w = Wd * j / nw
            ax.add_line(Line2D(*zip(p(0, v, w), p(U, v, w)), color=ec, lw=0.3,
                               zorder=z + k + 0.5))


def panel(x0, y0, w, h, title, tsize=8.0):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=PANEL_FC, ec=PANEL_EC, lw=0.7,
                           zorder=0))
    txt(x0 + w / 2.0, y0 + h - 7.0, title, size=tsize)


def formula(x, y, s, size=6.5, z=8):
    txt(x, y, s, size=size, z=z)


# ================================================================= Stage 1


def stage1():
    panel(3, 143, 281, 136, "Stage 1: Data Acquisition & FFT")

    # ---- CIR tensor (N_rx x N_taps x L)
    U, V, Wd, s = 4.0, 4.0, 3.0, 12.5
    p = projector(57.0, 222.0, s)
    cir_cube(p, U, V, Wd, 4, 4, 3)
    txt(30.0, 177.0, r"$N_{rx}$", size=6.4, rot=-30.0)
    txt(86.0, 177.0, r"$N_{taps}$", size=6.4, rot=30.0)
    txt(8.5, 216.0, r"$L$", size=6.4, rot=90.0)
    txt(54.0, 162.0, "CIR tensor", size=6.2)
    txt(54.0, 152.0, r"$N_{rx}\times N_{taps}\times L$", size=6.4)

    # ---- FFT arrow
    block_arrow(112.0, 168.0, 215.0, w=3.4, hw=9.4, hl=9.4)
    txt(146.0, 226.0, "FFT", size=8.0)

    # ---- frequency-domain tensor (N_rx x N_freq x L), 4 coloured slabs
    Uf, Wf, sf, dv = 4.0, 3.0, 12.5, 1.0
    q = projector(219.0, 218.0, sf)
    plate_stack(q, Uf, 0.0, Wf, 4, dv, [CL[3], CL[2], CL[1], CL[0]], 4, 3)
    txt(203.0, 177.0, r"$N_{rx}$", size=6.4, rot=-30.0)
    txt(181.0, 186.0, r"$N_{freq}$", size=6.4, rot=30.0)
    txt(177.0, 218.0, r"$L$", size=6.4, rot=90.0)
    txt(216.0, 162.0, r"$H$ (frequency domain)", size=6.2)
    txt(216.0, 152.0, r"$N_{rx}\times N_{freq}\times L$", size=6.4)


# ================================================================= Stage 2


def stage2():
    panel(288, 143, 419, 136, "Stage 2: Channel-Frequency Stacking")

    # ---- l-th snapshot slice, N_freq rows x N_rx columns
    mx0, my0, mxw, myh = 310.0, 176.0, 72.0, 72.0
    nrow, ncol = 8, 4
    cw, ch = mxw / ncol, myh / nrow
    for c in range(ncol):
        for r in range(nrow):
            rect(mx0 + c * cw, my0 + r * ch, cw, ch, CL[c], ec="none", lw=0.0,
                 z=3)
    for c in range(ncol + 1):
        ax.add_line(Line2D([mx0 + c * cw] * 2, [my0, my0 + myh], color=INK,
                           lw=0.45, zorder=4))
    for r in range(nrow + 1):
        ax.add_line(Line2D([mx0, mx0 + mxw], [my0 + r * ch] * 2, color=INK,
                           lw=0.45, zorder=4))
    ax.add_patch(Rectangle((mx0, my0), mxw, myh, fc="none", ec=INK, lw=0.7,
                           zorder=5))
    txt(347.0, 257.0, r"$H_l$", size=6.8)
    thin_arrow((mx0, my0 - 6.0), (mx0 + mxw + 2.0, my0 - 6.0), lw=0.6,
               scale=4.6)
    txt(347.0, my0 - 13.0, r"$N_{rx}$", size=6.3)
    thin_arrow((mx0 - 6.0, my0), (mx0 - 6.0, my0 + myh + 2.0), lw=0.6,
               scale=4.6)
    txt(mx0 - 13.0, 212.0, r"$N_{freq}$", size=6.3, rot=90.0)

    # ---- flatten arrow
    block_arrow(382.0, 418.0, 212.0, w=3.2, hw=8.8, hl=8.8)
    txt(400.0, 224.0, "Flatten", size=6.8)

    # ---- snapshot vector x_l, channel blocks top to bottom
    sx0, sw, sy0, sh = 424.0, 20.0, 172.0, 80.0
    for k in range(4):
        rect(sx0, sy0 + (3 - k) * sh / 4.0, sw, sh / 4.0, CL[k], ec="none",
             lw=0.0, z=3)
    ax.add_patch(Rectangle((sx0, sy0), sw, sh, fc="none", ec=INK, lw=0.7,
                           zorder=4))
    txt(sx0 + sw + 8.0, 212.0, r"$M=N_{rx}N_{freq}$", size=6.2, rot=90.0)
    txt(sx0 + sw + 18.0, 212.0, r"$\mathbf{x}_l\in\mathbb{C}^{M}$", size=6.6,
        rot=90.0)

    block_arrow(468.0, 492.0, 212.0, w=3.2, hw=8.8, hl=8.8)
    txt(503.0, 212.0, r"$\cdots$", size=11.0)
    block_arrow(514.0, 538.0, 212.0, w=3.2, hw=8.8, hl=8.8)

    # ---- data matrix X, M rows x L columns
    xx0, xy0, xw, xh = 548.0, 180.0, 140.0, 68.0
    cell_grid(xx0, xy0, xw, xh, 10, 12, band_rows(12, CL), lw=0.4)
    bracket(xx0 - 5.0, xy0 - 4.0, xw + 10.0, xh + 8.0, tick=6.0)
    txt(618.0, 162.0, r"$\mathbf{X}\in\mathbb{C}^{M\times L}$", size=6.8)
    txt(618.0, 151.0, r"$M=N_{rx}N_{freq}$", size=6.2)


# ================================================================= Stage 3


def stage3():
    panel(3, 3, 377, 136, "Stage 3: Covariance & Forward-Backward Averaging")

    # ---- data matrix X
    bracket(18.0, 50.0, 60.0, 54.0, tick=6.0)
    txt(48.0, 77.0, r"$\mathbf{X}$", size=11.0)
    txt(48.0, 38.0, r"$M\times L$", size=6.3)

    block_arrow(78.0, 126.0, 77.0)
    formula(102.0, 93.0, r"$\hat{\mathbf{R}}=\frac{1}{L}\mathbf{X}\mathbf{X}^{H}$",
            size=7.0)

    # ---- forward-backward averaging module
    rect(132.0, 52.0, 84.0, 50.0, BOX_FC, ec=BOX_EC, lw=0.8, z=4)
    txt(174.0, 84.0, "Forward-Backward", size=6.8)
    txt(174.0, 72.0, "Averaging", size=6.8)
    formula(174.0, 36.0,
            r"$\mathbf{R}_{fb}=\frac{1}{2}(\hat{\mathbf{R}}+\mathbf{J}\hat{\mathbf{R}}^{*}\mathbf{J})$",
            size=6.5)

    block_arrow(216.0, 246.0, 77.0)

    # ---- loaded covariance
    bracket(250.0, 50.0, 60.0, 54.0, tick=6.0)
    txt(280.0, 77.0, r"$\tilde{\mathbf{R}}$", size=11.0)
    formula(281.0, 36.0,
            r"$\tilde{\mathbf{R}}=\mathbf{R}_{fb}+10^{-5}\,\mathrm{tr}(\mathbf{R}_{fb})\,\mathbf{I}$",
            size=6.2)

    block_arrow(320.0, 390.0, 77.0)


# ================================================================= Stage 4


def stage4():
    panel(384, 3, 323, 136, "Stage 4: Subspace Processing & Pseudo-Spectrum")

    # ---- eigendecomposition at the given subspace dimension
    rect(386.0, 47.0, 64.0, 60.0, BOX_FC, ec=BOX_EC, lw=0.8, z=4)
    txt(418.0, 97.0, "EVD", size=8.4, weight="bold")
    txt(418.0, 85.0, r"$\tilde{\mathbf{R}}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^{H}$",
        size=6.2)
    txt(418.0, 71.0, r"$K_{\mathrm{alg}}=N_{\mathrm{true}}$", size=6.2)
    txt(418.0, 61.0, "(given)", size=5.8, style="italic")

    # ---- U_s: terminal output of the partition, it does not enter the spectrum
    block_arrow(454.0, 482.0, 96.0, w=2.6, hw=7.0, hl=7.0)
    txt(475.0, 106.0, "signal subspace", size=5.2, style="italic",
        color="#555555")
    txt(490.0, 96.0, r"$\mathbf{U}_s$", size=7.2)

    # ---- U_n: noise subspace, one of the two inputs of the pseudo-spectrum
    block_arrow(454.0, 500.0, 72.0, w=2.6, hw=7.0, hl=7.0)
    txt(488.0, 82.0, r"$\mathbf{U}_n$", size=7.2)
    txt(475.0, 62.0, "noise subspace", size=5.2, style="italic",
        color="#555555")

    # ---- pseudo-spectrum inset (schematic, not measured data)
    gx0, gy0, gw, gh = 502.0, 48.0, 102.0, 64.0
    ax.add_patch(Rectangle((gx0, gy0), gw, gh, fc="#FCFCFC", ec=INK, lw=0.6,
                           zorder=3))
    for i in range(1, 9):
        ax.add_line(Line2D([gx0 + gw * i / 9.0] * 2, [gy0, gy0 + gh],
                           color=GRIDC, lw=0.3, zorder=3.2))
    for j in range(1, 6):
        ax.add_line(Line2D([gx0, gx0 + gw], [gy0 + gh * j / 6.0] * 2,
                           color=GRIDC, lw=0.3, zorder=3.2))
    # search-grid axes drawn inside the inset, so no label collides outside
    thin_arrow((gx0 + 4.0, gy0 + 5.0), (gx0 + gw - 6.0, gy0 + 5.0), lw=0.6,
               scale=4.4, z=5.5)
    thin_arrow((gx0 + 5.0, gy0 + 4.0), (gx0 + 5.0, gy0 + gh - 8.0), lw=0.6,
               scale=4.4, z=5.5)
    txt(gx0 + gw - 3.0, gy0 + 11.0, r"$r$", size=6.0, z=6)
    txt(gx0 + 11.0, gy0 + gh - 12.0, r"$\theta$", size=6.0, z=6)
    for cx, cy in ((532.0, 92.0), (570.0, 66.0)):
        for rw, rh, col in ((11.0, 8.0, "#F6D9B8"), (7.5, 5.5, "#EFA45E"),
                            (4.2, 3.1, "#CE5A17")):
            ax.add_patch(Ellipse((cx, cy), 2 * rw, 2 * rh, fc=col, ec="#8A3A0F",
                                 lw=0.35, alpha=0.92, zorder=4))
        ax.add_line(Line2D([cx - 2.2, cx + 2.2], [cy - 2.2, cy + 2.2],
                           color=INK, lw=0.8, zorder=5))
        ax.add_line(Line2D([cx - 2.2, cx + 2.2], [cy + 2.2, cy - 2.2],
                           color=INK, lw=0.8, zorder=5))
    txt(556.0, 106.0, "Schematic", size=5.0, style="italic", color="#6E6E6E")

    # ---- steering vector evaluated on the same search grid
    txt(508.0, 24.0, r"$\mathbf{a}(r,\theta)$", size=6.2)
    thin_arrow((514.0, 30.0), (514.0, 45.0), lw=0.8, scale=5.0)
    formula(622.0, 30.0,
            r"$P_{\mathrm{JSF-MUSIC}}(r,\theta)=\frac{1}{\mathbf{a}(r,\theta)^{H}\mathbf{U}_{n}\mathbf{U}_{n}^{H}\mathbf{a}(r,\theta)}$",
            size=6.5)

    # ---- unified peak picking -> paired estimate
    block_arrow(608.0, 640.0, 77.0)
    txt(624.0, 96.0, "unified", size=6.0)
    txt(624.0, 88.0, "peak picking", size=6.0)
    rect(646.0, 52.0, 54.0, 54.0, "#EDEDED", ec="#9A9A9A", lw=0.7, z=4)
    txt(673.0, 87.0, r"$(\hat{r},\hat{\theta})$", size=9.0)
    txt(673.0, 70.0, "paired", size=6.0)
    txt(673.0, 62.0, "joint estimate", size=5.4, color="#444444")


# ================================================================= output


def main():
    stage1()
    stage2()
    stage3()
    stage4()

    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    paper = os.path.join(base, "paper")
    preview = os.path.join(base, "preview")
    os.makedirs(paper, exist_ok=True)
    os.makedirs(preview, exist_ok=True)

    pdf = os.path.join(paper, "algo_flowchart_redrawn.pdf")
    svg = os.path.join(paper, "algo_flowchart_redrawn.svg")
    png = os.path.join(paper, "algo_flowchart_redrawn.png")
    png_prev = os.path.join(preview, "flowchart_redrawn.png")

    # one write per artefact, with the embedded timestamps pinned so that a
    # re-run of this script reproduces the same bytes (hash-stable output)
    fig.savefig(pdf, format="pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(png, format="png", dpi=600)
    shutil.copyfile(png, png_prev)
    plt.close(fig)

    # matplotlib derives SVG clip-path ids from object addresses, so they change
    # from run to run. Renumber them by first appearance to make the file
    # byte-stable as well.
    with open(svg, "r", encoding="utf-8") as fh:
        svg_text = fh.read()
    counter = {}

    def _renumber(match):
        key = match.group(0)
        if key not in counter:
            counter[key] = "clip{0:03d}".format(len(counter))
        return counter[key]

    svg_text = re.sub(r"p[0-9a-f]{10}", _renumber, svg_text)
    with open(svg, "w", encoding="utf-8", newline="") as fh:
        fh.write(svg_text)

    for path in (pdf, svg, png, png_prev):
        print("wrote {0}  ({1} bytes)".format(path, os.path.getsize(path)))
    print("figure size: {0} x {1} in".format(FIG_W, FIG_H))


if __name__ == "__main__":
    main()
