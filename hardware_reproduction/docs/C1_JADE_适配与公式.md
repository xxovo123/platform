# C1（JADE 1998 / SIJADE）硬件侧适配与公式对照

**对象**：van der Veen, Vanderveen & Paulraj, *Joint Angle and Delay Estimation Using
Shift-Invariance Techniques*, IEEE TSP 46(2):405-418, 1998，DOI 10.1109/78.655425（原文自称 SIJADE）。
**实现**：`hardware/methods/c1_jade.py`（只依赖 numpy）；**测试**：`hardware/methods/tests/test_c1_jade.py`；
**决策先于实现的记录**：`hardware/logs/C1_variant_decision.md`；**公式页渲染**：`hardware/logs/C1_pdf_pages/p03..p08,p11.png`。
**性质**：实现与公式正确性；**本文件不含任何性能结论**，214 窗性能未跑。

## 1. 原文页码 ↔ 公式 ↔ 本实现（逐项对应，不只取名）

| 原文位置 | 公式/结论 | 本实现 | 测试 |
|---|---|---|---|
| P407 II-B，式(4)(5) | DFT 把时延映为频域相移；除以已知脉冲形状 `ghat`（式(6)） | `_hbar_from_case`：按 `freq_order` 排序频点、除 `S[freq_order[k]]` | 非平坦 S 用例 |
| P408 式(6)(7) | 频点须**按递增频率排列**（J_g，"the final floor(LW/2) samples … moved up front"）；`Hbar = A B F` | 频率排序 + `freq_order` 写入 diagnostics | 原始 fftfreq 序用例 |
| P409 式(8) | `H := [Hbar(1); …; Hbar(m1)]`（频率左移叠加） | `_hankel_2d` 的 j 维 | 数学 oracle |
| P409 式(9) | `A = A_phi ◇ A_theta`（Khatri–Rao），`phi_i=e^{-j2pi*df*tau_i}`、`theta_i=e^{+j2pi*D*sin(theta_i)}` | `_native_model_matrix` / cube 构造 | 数学 oracle |
| P409 式(10) | 选择矩阵 J_xphi/J_yphi/J_xtheta/J_ytheta | `_selection_indices` | 数学 oracle |
| P409-410 式(11) | 二维平滑的广义 H（m1,m2）：行 =(j,a)、列 =(i,b) | `_hankel_2d`（含 m2 维） | 数学 oracle |
| P410 式(12)(13) | 前后向共轭增广 `H_e=[H, J H^{(c)}]`（J 行交换） | `_forward_backward` | 数学 oracle |
| P410 式(14) | SVD 秩削减 + QR/LS 投影得 r×r 的 E 矩阵 | `_rank_reduce` + `_qr_reduce` | 秩守卫用例 |
| P410-411 式(15) | **III-D 实数变换（不做）** | 明确不实现（原文自陈 numerically inadvisable…） | — |
| P411 式(16) | `E_xphi^{-1}E_yphi = T^{-1}Phi T`、`E_xtheta^{-1}E_ytheta = T^{-1}Theta T` | `_pencil_eig` | 配对用例 |
| P411 末段 | **"An alternative but similar way"**：独立求 T_phi/T_theta，`T_theta^{-1}T_phi` 在特征值互异时应接近置换阵 | `_normalized_abs_m` + `_permutation_assignment` + `_pairing_report` | 3-cycle / 冲突 / K=2 用例 |
| P412 式(19) 及 a)~c) | 可辨识条件与 (m1,m2) 最优取值 | `_identifiability`、`m1=11,m2=2` | diagnostics 断言 |
| P410/P412 | 同延时需 `m2>=d/2, M>=3d/2`；同角度需 `m1>=d+1, LW>=3d/2` | 容量申报（只报告不丢弃） | 同延时/同角度用例 |

**不冒充的部分**：III-E 只做**末段 alternative**。原文 Method "Q"（ACMP）需要"Schur 起手 + 局部 2×2
Jacobi 旋转，每次旋转接近单位阵或接近置换、并同步 Phi 的标号"，且原文 P411 明确它 *would fail in the
theoretical case of exactly repeated eigenvalues of the first matrix*；Method "T" 依赖 III-D 实数变换，
原文 P413 指出临界情形下特征向量矩阵变复、`Phi`/`Theta` 不再实值（角度与延时混叠）；Method "QZ"
为迭代法，原文 P411-412 说其 `convergence properties have been disappointing`、需要 accurate
starting point。本实现不写局部旋转版，以免在噪声下把 Phi 的对角打散后仍冒充成配对。

## 2. 固定参数及理由（冻结，不接受 params 覆盖）

| 项 | 取值 | 理由/来源 |
|---|---|---|
| m1 | 11 | 原文式(19)：`round((N+1)(2-√2)) = round(10.544)`，N=17 |
| m2 | 2 | 原文式(19)：`round(M(√2-1)) = round(1.657)`，M=4 |
| 数据增广 | 式(11) 二维平滑 **+** 式(12) 前后向 | 原文把共轭增广列为"Improved resolution"（Abstract、III-B） |
| 秩削减 | H_e 的 SVD 取前 K 列 + QR/LS（式14） | 原文 III-C 的 LS 过程（比 TLS 省、原文称几乎同样准） |
| 末级 | III-E 末段 alternative | 见上；配对显式、局限可报告 |
| 参考频率 f_ref | `config.center_freq_si = 7.9872 GHz` | 原文窄带假设的"带内两端波长近似"→ 取带中心 |
| 延时参考 | **ref-anchor**：tau 锚定参考阵元 x=v_x_ref=-0.038 m | 见 §3.3（若用 origin-carrier 会有 <=1.9 cm 系统偏差） |
| 共用域 | **严格** bbox + y<0 + 距离环 + FoV，仅 `DOMAIN_EPS=1e-9 m` | 与 B1 现行 `_in_domain` 的严格实现一致；**无 half-cell 外延** |
| 时间容差 | `TIME_EPS=1e-18 s`（独立于空间容差） | 把 1e-9 当秒用会外扩 ±1 ns ≈ ±0.15 m |
| 数值秩阈值 | `rel_tol = 1e-8`（相对 s_max） | **冻结的数值近似阈值**（人为选定、非物理判据）；作用范围**仅限自测覆盖到的情形**（零能量输入；严格窄带模型下单源却声明 K=3），**不构成"真实秩确实不足"的一般断言**，见 §3.1/§4 |
| 共享后处理 | `common/q6_contract.py::postprocess_native_candidates`（0.10 m、cap=K） | 路线 §8.2；**不加**二维全谱 3×3 + 0.3 阈值 |
| 原生 score | `||G[l,:]||_2, G = A_nat^+ H_e` | 逐条路径"原生模型 LS 解释能量"，不用真值 |

## 3. 适配界限（声明近似，不是代码缺陷）

### 3.1 窄带阵列 vs 本机频变空间相位（A5）
原文 A_theta 与频率无关（P405 条件 5：*narrowband phased array … wavelengths at both ends of the band
are roughly the same*）；本机真实空间相位 `+2pi f v_x,ch sin(theta)/v_prop` **随频率变化**。
本实现**不**把流形改成频变版本（那已是另一种算法），按原文窄带结构估计单个 theta_hat，再用
f_ref 换算角度。本批相对带宽 500/7987.2 = 6.26%、两端 lambda 差 6.4%。
**实测后果（本次最小自测，无噪、声明平面波模型）**：角度误差 < 0.05°，位置误差 <= 7.1 mm
（主因是宽带失配耦合进 phi 相位造成几 mm 的 r 偏差），见 §5 的日志。
该失配还会以"额外秩"的形式出现：**单快照 `rank(Hbar)` 实测 = 4（C=4）**。因此 §4 的数值秩守卫
（1e-8 是**人为选定的冻结近似阈值**）只拦"零能量"与"严格窄带模型下 K>rank"这两类**自测已覆盖**的
输入；它**不把宽带失配当秩不足**，也**不对真实秩作一般断言** —— diagnostics 同时给出奇异值头、
秩估计、gap 与所用阈值，供主审自行判断。

### 3.2 相位唯一性与限定 FOV（A6）
D_ref = d·f_ref/v_prop = 0.506206（> 0.5，即 d 略大于 lambda/2）。theta_hat 的唯一解域为
|theta| <= asin(1/(2D_ref)) = **81.03°**。**不按 `fov_degrees=180` 一刀切判 NA**：实际候选域 =
grid_x[-2.2,2.2] × grid_y[-4.2,-0.55] ∩ y<0 ∩ 距离环，域内 theta_max = atan2(2.2,0.55) = **75.96°**，
D_ref·sin(theta_max) = **0.49109 < 0.5**（余量 1.78%）→ 限定域内相位唯一。
诊断同时给出 band-edge 风险：用 f_hi=8.2372 GHz 时 D(f_hi)·sin(theta_max) = 0.5065 > 0.5，
即**最高频点**在最大角度处已进入缠绕区，属模型局限（原文要求"带宽远小于载频"）。

### 3.3 双程 + system_delay + 参考阵元（A7，ref-anchor）
Phi 是**共同**延时相位（式(7)），锚定在参考阵元：`tau_anchor = (2 r_origin - v_x_ref sin(theta))/v_prop
+ system_delay`。反解 `r_origin = v_prop(tau_hat - system_delay)/2 + (v_x_ref/2) sin(theta_hat)`
（本批 -0.019 m），再 `x = r sin(theta_hat)`、`y = -r cos(theta_hat)`。
**域边界同式换算**：`tau_lo/hi(theta) = system_delay + 2 r_min/max/v_prop - v_x_ref sin(theta)/v_prop`。
理由：不修正会有 <=1.9 cm 的系统性距离偏差；直接用 origin 环当 tau 域会整体偏
`v_x_ref sin(theta)/v_prop`，其量级为 `0.038/v_prop = 0.12675 ns`（sin(theta)=1 上界；
本批域内 theta_max=75.96° 处为 0.1229 ns），足以影响 32 ns 周期下的分支判定。

### 3.4 延时相位周期与分支（A9）
`phi = e^{-j2 pi df tau}`，df = 31.25 MHz → 相位周期 1/df = 32 ns，对应**双程距离周期
v_prop/(2 df) = 4.7967 m**（2.398 m 是相位主值半周期对应的 tau = 16 ns）。
本批延时域跨度 = 2(r_max - r_min)/v_prop = 30 ns <= 32 ns → "取模后落域内"的分支**唯一**
（余量 2 ns = 6.25%）。
**多解时**：该 pair 标 `delay_branch_ambiguous`，**不计入成功**（全部分支写入
`branch_candidates`）；若全部 pair 都 ambiguous，整体 `status='failed'`。**不用"环中心"猜分支**。

## 4. 失败 / 不适用 / 退化（显式，不静默）

* **not_applicable（结构）**：k=0；C<2；F<3；H 行数<k；虚通道 x 不等间距或 y/z 跨通道极差≠0；
  带内频点非等间隔；k 超过式(19) 可辨识上限。
* **failed（输入/状态）**：缺键、形状不符、X 含 NaN/Inf、pairs 越界、S 长度不符、
  **S 在被选频点为 0**（式(6) 除法退化）、**数值秩 < k**（含零能量）、**全部候选分支 ambiguous**、
  以及传入非空 params（参数冻结）。`np.linalg.solve` 在近退化 pencil 上抛 `LinAlgError` 时
  同样记 **failed**（只如实申报；**不为近退化情形加特判/正则/优化**）。
* **ok 但零候选**：全部原生解落在共同域外（含取模后落域外）——"允许 <K"。
* **退化只报告不丢弃**：同延时/同角度容量、`min_pairwise_phi_delay_sep_ns`、配对纯度/偏差、
  `cond(T_phi)/cond(T_theta)`、两 pencil 最小特征值间距、以及 K=2 自逆置换的掩盖效应，
  全部落 diagnostics，**不删不利样本**。

## 5. 运行命令与日志 / SHA

```powershell
# 自测（BLAS 线程=1；34 项检查）
& .venv/Scripts/python.exe -B 大修执行_20260923/04_Q6/hardware/methods/tests/test_c1_jade.py
#   -> hardware/logs/C1_tests_20260924.log（SUMMARY: 34/34 PASS）

# 真实窗合同 smoke（3 个录制首窗，含在自测最后 3 项；非性能测试）
# 公式页渲染（供主审核对原文）
& .venv/Scripts/python.exe -c "<pymupdf: JADE1998_SIJADE_delft_repository.pdf -> dpi=200 png>"
#   -> hardware/logs/C1_pdf_pages/p03..p08,p11.png
```

SHA256 见 `hardware/logs/C1_sha_manifest.log`（模块/测试/文档/日志/渲染件）。

**未做的事（明确）**：未跑 214 窗正式性能、未改论文与公共文件、未动 B0/B1-B4 与缓存/真值、
未按真值调参、未用真实效果门槛。
