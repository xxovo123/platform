# B5 硬件侧：Wang 2014 §3.1 传统 full-2D (τ,θ) MUSIC 的四通道 CIR 扩展 —— 原文公式、适配与偏离

> **性质**：方法实现说明 + 首轮自测记录。**不改**论文 tex/正式回复/其它方法；**不读真值**；
> **不写历史产物**。原法结果与"同处理受控对照"**分开陈述**：本文件描述的是**适配后的传统对照**，
> **不称 Wang 原法原样复现**。

## 0. 身份与来源（本地 PDF 原页已核对）

| 项 | 值 |
|---|---|
| 原文 | Wang, Zhang & Wang, "Joint Estimation of TOA and DOA in IR-UWB System Using a Successive MUSIC Algorithm", **Wireless Personal Communications (2014) 77:2445–2464**, DOI 10.1007/s11277-014-1644-z（Springer） |
| 原页 | 封面 Wang_p01；式(6)–(10) 见期刊 p.2448（Wang_p04）；式(11)–(15) 与 §3.1 标题见 p.2449（Wang_p05）；§3.2 successive 见 p.2450（Wang_p06） |
| 多阵元结构参考 | Deng, Li, Tang & Zhang 2023, **Sensors 23(14):6363**, §3.1 式(6)–(10)（原页 Deng_p04/p05） |
| 方法 id | b5_wang_full2d（模块 hardware/methods/b5_wang_full2d.py） |
| 与旧 B3 的关系 | **不同项**。旧 B3 = Wang 2014 **成功逐次（successive）** MUSIC 的两 Rx 版（§3.2 起）；本项 = **§3.1 的双参数联合二维谱**准则 + Deng 多阵元结构 ⇒ 四通道 (τ,θ) **联合**搜索 |

**原法原样描述（Wang §3.1，双天线）**：S ∈ C^{N×N} 为**发射谱对角阵**（元素 = 发射信号频率采样
S(ω_n)）；E_τ = [e_1,…,e_L] ∈ C^{N×L} 为**频率×路径**时延矩阵（**不是**对角阵），
列 e_l = [1, e^{−jΔω τ_l}, …, e^{−j(N−1)Δω τ_l}]^T；Y_1 = S E_τ B + W_1、Y_2 = S E_ζ B + W_2（式
(7)(8)）；Z = [Y_1; Y_2] = [S E_τ; S E_ζ] B + [W_1; W_2]（式(13)，**通道优先堆叠**）；
R̂ = Z Z^H / K，P_2D-MUSIC(τ,ζ) = 1/(a^H Û_N Û_N^H a)（式(15)）；Δτ_l = ζ_l − τ_l = d sinθ_l/c，
θ̂_l = arcsin(Δτ_l c/d)（式(11)(12)）。**本模块不实现 §3.2 的逐次一维版**。

**Deng §3.1（多阵元，本项的直接结构来源）**：e_{τ_i l} = Φ^{i−1}(θ_l) e_{τ_1 l}，
Φ(θ) = diag(e^{−jω_n d sinθ/c})（式(6)(7)）；Z = [Y_1;…;Y_M] = A(τ,θ)B + V ∈ C^{MP×K}（式(8)，
只取非零带内频点 P）；R̂_ZZ = ZZ^H/K（式(9)）；P_2D-MUSIC(τ,θ) = 1/(a^H Û_N Û_N^H a)（式(10)）。

## 1. 四通道 CIR 适配（逐条；偏离必须随结果引用）

- **A1 观测接口**：原文为"发射 → 多径 → 阵列接收"单向信道；本数据为 Tx → 目标 → Rx 双程 CIR。
  适配（与 B1 同一平面波适配）：绝对虚拟坐标 v_ch = tx[pair0] + rx[pair1]，导向
  a[f,ch] = S_f · exp(−j2π f τ) · exp(+j2π f v_x,ch sinθ / v_prop)，
  即 τ_ch ≈ τ − v_x,ch sinθ / v_prop；τ = **坐标原点双程等价到达时延**（含 system_delay）。
  **不使用**本文主方法的 exact Tx→target→Rx 双程球面流形（由自测"tx/rx 同步移轴后谱不变"守卫）。
- **A2 符号**：原文式(7) 印作 e^{−jω d sinθ/c}；本坐标（θ = atan2(x,−y)，视轴 −Y）取
  **相对**相位 +2π f (v_x,ch − v_x,ref) sinθ / v_prop（反号），与 B1 A2 同。
- **A3 阵结构**：虚通道须沿 x **等间距**、v_y/v_z **通道间全同**（原文 Φ^{m−1} 结构），
  否则 not_applicable。本批 4 通道：v_x = [−0.038, −0.019, 0, 0.019] m（间距 0.019 m）、v_y = 0、
  v_z = −0.02 m。
- **A4 v_y / v_z 共同分量**：远场一阶项 u_θ·v_ch（u_θ = (sinθ, −cosθ, 0)，目标在 z = 0 平面）。
  **v_z 共同分量**对 u_θ·v_ch 贡献**恒为 0**（u_z = 0），故它**不进入**平面波一阶模型 ——
  **不声称它被吸收进 τ**；它只经二阶项影响精确几何，属已披露的近似残差。
  **v_y 共同分量**（u_y = −cosθ ≠ 0）**会**进入一阶项；本模块**未实现** u_y 项，故对
  |v_y| > 1e-9 m 的窗**显式 not_applicable**（本批 v_y = 0，不触发）。
- **A5 频率顺序**：case X 为 (L, F*C) **频率优先**（index = f*C + ch）。**不排序**，按
  frequencies_hz 原始序（本批非单调）逐点构流形；无 bare J、无额外置换。
- **A6 堆叠置换与共轭（关键口径）**：原文式(13)(8) 为**通道优先**堆叠，本 case 为**频率优先**，
  两者只差固定行置换 P（Z_channel_major = P·Z_freq_major，Z_freq_major = X^T），故
  R̂ = Z Z^H / K 在本 case 口径下即 R̂ = X^T X^* / L（X.T @ X.conj()，**不得**写成 X^H X）。
  式(15)(10) 的分母是**论文口径** denom = a^H U_N U_N^H a = ||U_N^H a||²；实现上取
  proj = A @ U_N.conj()（A 行 = a^T，故 proj 行 = (U_N^H a)^T）。
  **不得**写成 A @ U_N（那是 ||U_N^T a||²，另一共轭口径）；**main-style** 的
  Re Σ (A P) conj(A) 与论文口径在**同一个 U_N** 下**一般不等价**（只在 U_N 为实数等特殊情形相同），
  两种口径不得混用或互相折算。独立验证用**标量** conj(a) @ (U_N U_N^H) @ a。
- **A7 协方差**：原文式(15)(9) **无前后向平均（FB）、无对角加载**；本模块**只实现原生口径**
  （不提供 FB/加载开关）。case 的 config.is_forward_backward = True 属主方法选择，**不继承**。
- **A8 谱与取峰**：P(τ,θ) = 1/(a^H U_N U_N^H a)、噪声子空间维 M − K（M = F*C = 68）。
  返回**未归一**的谱 1/(denom + 1e-12)，并用**同一冻结取峰规则**在 estimate() 内取自身峰
  （3x3 非严格极大 + 8 连通平台去重 + 0.30 线性相对阈值 + 0.10 m 确定性 NMS + cap = K，
  来自 knownk_protocol.json 的 peaks 段）。评测仍走 **spectrum 路径**（eval_path='spectrum' +
  显式 spectrum_grid），由运行器/评测器重新取峰。
- **A9 二维搜索域与映射**：二维参数域 = (τ,θ)。公共 xy 网格（y < 0）与 (τ,θ) **在域内一一对应**：
  r = hypot(x,y)、θ = atan2(x,−y)、τ = 2r/v_prop + system_delay；反解
  r = v_prop(τ − system_delay)/2、x = r sinθ、y = −r cosθ。故**正式评估网格 = case 的公共
  Cartesian 网格**（6586 个掩膜内格点）—— 这**就是** (τ,θ) 二维搜索（同一二维域换采样），
  并天然复用公共域/取峰口径。FROZEN_PARAMS 的 native_selftest_tau_step_distance_m /
  native_selftest_angle_step_deg **只用于**导出函数 full2d_spectrum_native（原生矩形 (τ,θ)
  网格，自测/公式核验），**不参与** estimate()；**不得**把本模块描述成"在均匀 τ/θ 网格上跑正式结果"。
- **A10 S（发射/共同响应谱）**：原文 S 是**发射谱对角阵**。本数据的 CIR 里真实发射谱**未知**，
  故默认 **S = 1 是"通道间共同响应"假设**（**不是**已知平坦真实发射谱、**不是**已校准响应）。
  **S 允许含零元素**（Deng 式(8) 本就只在非零带内频点构造 Z；零元素只把该频点行置零）。
  注意 B1 的输入解析含一条**降维（§4）专属**约束 S[ref] ≠ 0；**它不是 full-2D 的要求**：
  本模块复用 B1 解析时**自动**挑一个非零频点作参考索引，并在 diagnostics 申报该索引
  **只用于通过 B1 解析、不参与 B5 的 full-2D 运算**（B1 代码未改）。
- **A11 计时**：estimate() 从 case['X'] 起算（协方差 → EVD → 导向 → 公共网格谱 → **自身取峰**），
  与主方法计时范围一致；不含解码/重放/裁剪/FFT 构造。

## 2. 已知局限（随结果一起引用，不得省略）

1. **不是原法复现**：4 虚拟通道、平面波适配、CIR 频点集、S=1 假设均为**适配选择**；
   原法为远场双天线/多阵元、发射谱对角阵、信道多径模型。
2. **平面波一阶近似**：不含精确双程球面项；共同 v_z 只在一阶项外影响精确几何（未量化在计时内）。
3. **无 FB / 无对角加载**：原文如此；本批硬件数据是否因此更差**属基线自身情况**，不为提升效果改原法。
4. **无噪"针状峰"与网格步长的关系**（自测已观察）：无噪极限下二维泛函在精确 (τ,θ) 处为 δ 峰，
   **不在公共 5 cm 网格点上的目标可能被漏掉**（自测：非格点目标 5 cm 网格 err ≈ 0.040 m、
   1 cm 局部网格 err ≈ 0.0045 m）。这是**采样性质**，不是公式偏差；正式结果**必须**按 5 cm 公共网格口径陈述。
5. **S = 1 是共同假设**：真实发射谱未知；若给出非平坦 S，结果会变（自测已核 S 逐频点进入导向）。
6. **本模块不做**：不读真值、不按效果调参、不加 FB、不改 grid5cm 采样预算、不声称与原法等价。

## 3. 首轮自测结果（2026-09-27；45 项，45 PASS / 0 FAIL）

命令：python -B 大修执行_20260923/04_Q6/hardware/methods/tests/test_b5_wang_full2d.py

| # | 组 | 关键结论 |
|---|---|---|
| 1 | 公式 | 模块流形 == **独立重写**的平面波公式（max_abs ≈ 2.3e-13）；模块 denom == conj(a)(U_N U_N^H)a；A@U_N 与 A@U_N.conj() 在复 U_N 上**确不等价**（5.58 vs 8.04）；main-style ≠ 论文口径（同一 U_N）；U_N 为实数时两口径一致；非平坦 S 逐频点进入导向 |
| 2 | 原生无噪 | 单/双/三目标：真值点 denom ≈ 0（≤1.7e-15，相对 ‖a‖²=68 约 2.5e-17）；(τ,θ) 联合峰映射位置误差 ≤ 0.0033 m；全局谱最大均在真值邻域 |
| 3 | 坐标/映射 | 掩膜内 6586 点满覆盖；(x,y)↔(r,θ)↔τ 往返 <1e-9 m；公共网格谱 == 1/(conj(a)Pa+1e-12)（rel ≈ 4e-17）；native 单点值与公共网格一致；spectrum_grid 显式申报；**无噪格点目标**被冻结取峰器精确取到（<1e-9 m） |
| 4 | 堆叠/共轭/四通道 | R(X_perm) == P R(X) P^T（0.0 差）；数据列 + pairs 同步置换后谱不变；**消融任一通道都改变谱**（4 通道全用）；M = 68、噪声子空间维 = M−K；tx/rx 同步移轴后谱不变（**未用 exact 双程几何**） |
| 5 | S / v_y / FB / 边界 | S=1 标注为**共同响应假设**；S[0]=0 时 **B1 解析拒绝、B5 正常**（自动参考索引=1，已申报）；S 全零 → not_applicable；非零共同 v_y → not_applicable；fb_used=False 且协方差等于无 FB 公式；k=0 / k≥M / C<2 → not_applicable；X 含 NaN → failed |
| 6 | 真实窗合同 | 3 个代表录制首窗：status=ok、谱 + spectrum_grid 与 case 网格一致、预测数 ≤ K、M=68、fb_used=False（**合同 smoke，不是性能测试**） |

**计时**（本机，BLAS 线程=1）：estimate() 单窗约 **0.030 s**（含自身取峰；首窗约 0.056 s 含导入），
与主方法（0.028 s）/B1（0.027 s）同量级；口径 = 协方差 + EVD + 导向 + 谱 + 自身取峰。

## 4. 与集成层的接口

- 注册名 b5_wang_full2d；入口 estimate(case[, params]) -> {predictions, status, spectrum, diagnostics}；
  status ∈ {ok, failed, not_applicable}；NA 必带 reason（物理/结构原因）。
- 输出谱形状 = (len(grid_y), len(grid_x))，域外像元为 0，值与主方法同归一化口径（未归一 1/(denom+1e-12)）。
- diagnostics 关键字段：grid_kind / n_candidates_evaluated(=6586) / dictionary_size=None /
  M_features / L_snapshots / n_channels_used / fb_used=False / eval_path='spectrum' /
  spectrum_grid / official_grid / native_selftest_grid / peak_rule(+来源与协议 sha) /
  n_own_peaks / own_peak_status / adaptation(含 v_y·v_z 处理、S 来源与零元素、映射与反解)。
- 全量 214 窗运行需 hardware/approval/methods_approved.json 对该方法模块 sha256 的放行记录
  （模块改动后须重新审核；不绕开放行门槛）。

## 5. 改动文件与实际状态（截至 2026-09-27，均已完成）

- **方法实现**：hardware/methods/b5_wang_full2d.py；**自测**：hardware/methods/tests/
  test_b5_wang_full2d.py（45 项，45 PASS）；**本文档**。
- **运行器注册**：common/q6_run_hardware_eval.py 注册 b5_wang_full2d，并纳入默认主比较组
  → **现行默认主比较 = 六法**（本方法 + B1 + B2 + C1 + C3 + B5）。
- **放行记录**：hardware/approval/methods_approved.json 新增 b5_wang_full2d 条目
  （模块 sha256 647b515c9563…；同时登记被复用的 b1_deng.parse_case 模块 sha dc60203dab2b…）。
- **全量 214 实际运行已完成**：封装脚本 analysis_wang_full2d_20260927/run_b5_214.py；
  新归档 hardware/results/wang_full2d_20260927/（214/214 status=ok、无 failed、无 NA；
  wall ≈ 20.3 s；mean estimate 0.033602 s/窗）。
- **结果报告**：analysis_wang_full2d_20260927/REPORT.md（含 summary.csv / by_k.csv /
  by_environment.csv / per_window.csv / timing.csv / report.json）：既有五法冻结值 + B5，
  **214 实际运行**与**固定同一 205 子集后处理**两套；交叉核验五法 214 复现冻结产物
  （ua_overall 最大绝对差 5.551e-17；ospa_per_window 0.0）。
- **自测运行日志（SHA 绑定）**：analysis_wang_full2d_20260927/selftest_b5_wang_full2d.log
  与 .json；日志 sha256 = 4c0ded5888bc03f21d544de72d865117bf1a9ca6b859873c8be7cc7d54028163（同值写入 report.json 的 selftest_log_sha256）。
- **状态同步**：根 README_审稿修改路线.md（新增 B5 块，并在旧相邻块内**显式标注**被取代句；
  §13 逐字不改，尾段 106189 字节 / SHA256 8886e6bf…4602e9）、hardware/README.md、
  传统二维MUSIC四通道对照与全点误差方案_20260927.md。
- **未做**：不改论文 tex / 正式回复；不 SSH、不仿真；不覆盖任何历史归档。
  **独立核验与结果审计（Euclid）已完成并通过**：实现核验 **44 PASS / 0 FAIL / 3 INFO**（independent_verification/IV_B5_报告.md + iv_b5_independent_results.json）、结果审计 **10/10 PASS**（independent_verification/iv_b5_result_audit.json；独立 scipy 平方 Hungarian 与生产汇总逐项差 0、32 录制与 214 窗指纹一致），被测模块 SHA256 647b515c… 一致。

