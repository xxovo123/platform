# C3（Kim 2015 DFT-ESPRIT 的 CIR 适配版）原文公式与适配记录

实现体：大修执行_20260923/04_Q6/hardware/methods/c3_kim.py
自测：大修执行_20260923/04_Q6/hardware/methods/tests/test_c3_kim.py
日志：hardware/logs/C3_selfcheck_20260924.log（原始输出）、hardware/logs/C3_selfcheck_20260924.json（机器可读）
渲染核查件：hardware/logs/C3_pdf_render/（见 §1）

## 0. 状态与验收口径（诚实版，2026-09-24）

* **本模块只做实现与实现正确性验证**：10 项自测全部 PASS（退出码 0）。
* **不是性能结论**：**未运行** 214 窗正式性能比较；本文件与三份日志**不含**任何精度/命中率/优越性结论。
* 验收口径（主审 2026-09-24）：以**方法与实现正确**为准（原公式对应 + 独立 oracle 数值一致）；
  **效果不佳属基线自身**，不作为验收门槛；**不为提升效果改原算法、不按结果调参**（默认 zeropad=1，见 §5）。
* 只读 case、**不读真值**；候选去重/cap 直接调用 common/q6_contract.py::postprocess_native_candidates。
* 本轮**不改**论文/回复/README/公共文件与 B0-B4；C3 全部写在 hardware/methods/ 下独占文件前缀。
* **不声称逐式原样复现**：这是明确声明的 **CIR 适配版**（原文为 FMCW 去斜拍频观测）。

## 1. 原文逐式对应（按 PDF 渲染件逐式核对）

原文：Sangdong Kim, Daegun Oh, Jonghun Lee, Joint DFT-ESPRIT Estimation for TOA and DOA in
Vehicle FMCW Radars, IEEE AWPL 14:1710-1713, 2015, DOI 10.1109/LAWP.2015.2420579。
PDF 页与期刊页对应：PDF p1 = 1710（式(1)-(4)）；PDF p2 = 1711（式(5)-(14)）；PDF p3 = 1712
（Table I/II、式(15)(16)）；PDF p4 = 1713（实验）。渲染 PNG（400 dpi 全页 + 700 dpi 公式裁片）：

| 文件 | 内容 |
|---|---|
| C3_Kim2015_p2_full.png / C3_Kim2015_p2_left_eq5-8.png | 式(5)(6)(7)(8) 与 III 节结构 |
| C3_Kim2015_p2_right_eq9-14.png | 式(9)(10)(11)(12)(13)(14) 与 Step 1-4 |
| C3_Kim2015_p3_full.png / C3_Kim2015_p3_left_eq15-16.png | Table I/II 与式(15)(16) |
| C3_Kim2015_p1_full.png / C3_Kim2015_p1_left_signalmodel.png | 式(1)-(4) 与 ULA 定义 |
| C3_Kim2015_p4_full.png | 第 VII 节实测描述（1 Tx 2 Rx 演示） |

| 原文 | 本实现 | 处置 |
|---|---|---|
| 式(5) 去斜后 p_k[n] = sum_m a_m e^{j(mu tau_m T_s n + omega_s tau_m - mu/2 tau_m^2 + pi k sin theta_m)} + w_k[n] | 逐阵元时延锚点 CFR（见 §3 A10） | 保身份（本机无去斜，见 A1） |
| 式(6) 第一阵元 1D-DFT：P_1[l] = sum_n p_1[n] e^{-j2pi n l/N} | CFR 到时延谱逆 DFT（delay_spectra） | **替换核**：频率维逆 DFT（A1） |
| 式(7) tau_hat_m = I_m/(N*T_s*mu) | 不实现 | **替换**：FMCW 斜率式属去斜专有，本机无 mu；改为原生 range bin 换算（A1） |
| 式(8) P_m = [P_1[I_m] ... P_L[I_m]]^T，R_m = P_m P_m^H | esprit_from_bin：同索引跨阵元外积（快照平均扩展，A5） | 保结构；rank 1 限制见 §2 |
| 式(9) R_m 的 EVD：U_1 信号 / U_2 噪声 | np.linalg.eigh，**只取首个**特征向量 | 保算子；列数按 §2 决策取 1 |
| 式(10) U_{1,0} = J_0 U_1，U_{1,1} = J_1 U_1 | u0 = u[:C-1]、u1 = u[1:] | 逐字 |
| 式(11)(12) U_{1,0}=J_0 A T、U_{1,1}=J_0 A Phi T；A 第 k 行 e^{j pi k sin theta_m} | 窄带 fc 空间相位（A3）；比值闭式含实际阵元间距 dv_x | 保结构；符号按本坐标（A3） |
| 式(13) (U_{1,0}^dagger U_{1,1} - xi_m I) x_m = 0 | 单向量下退化为标量 xi = (J_0 u)^H (J_1 u)/||J_0 u||^2 | 逐字（广义 EVD 的 1x1 情形） |
| 式(14) theta_hat_m = asin( (1/pi) angle(xi_m) ) | sin(theta) = angle(xi) * v_prop/(2 pi f_c dv_x) | dv_x = lambda_c/2 时与原文完全一致 |
| Step 1-4 | 顺序逐字保留：第一阵元 DFT 取峰，同索引跨阵元自相关，广义 EVD 得 DOA | 无 MUSIC / 无 beamformer 替代 |

## 2. 原文 notation 矛盾与主审决策（如实报告，不擅自增强）

* 第 II 节正文写 ULA consisting of **L** elements，即阵元数 = L；式(8) 的 P_m 长度即阵元数。
* 但式(9) 写 U_1 = [u_0 ... u_{M-1}]（**M** 个信号特征向量）、U_2 为 L-M 维；式(10)(12) 又改用 **K**
  表示阵元数（J_0 = [I_{K-1} 0]、A 第 K 行为 e^{j pi (K-1) sin theta}）。**L / K / M 三者在原文中混用**。
* 更关键：式(8) 是**单向量外积**，rank(R_m) = 1，恒只有 1 个非零特征值，故 M 个信号特征向量
  **只可能对应 M = 1**；原文正文却把 M 当作目标总数，二者**不能同时成立** —— 同一 TOA（距离）索引上
  多目标时，式(8) 的秩结构**无法**分辨。
* **主审 2026-09-24 决策（本实现遵循）**：**每个距离峰只给 1 个角度**（空间维只取首个特征向量）；
  **不做**每 bin 多源 / 子阵平滑 / 前后向 / 对角加载等任何增强；**同距离多目标保留原限制**；
  多快照平均协方差**声明为式(8) 的统计扩展**；**不声称逐式原样复现**。

## 3. 本机适配（A1-A11，理由与界限）

* **A1 CFR 到时延谱（替换式(7)）**：case 给 X(L, F*C)，**频率优先**，且 17 个带内频点**原始数组序
  非单调**（[fc, fc+df, ..., fc+8df, fc-8df, ..., fc-df]）。式(6) 的 DFT 要求等间隔有序样本，故先按
  频率**排序**并对 X 列做**一致置换**（纯索引重标号，已申报 column_permutation_applied），再做逆 DFT：
  P_c[l] = sum_j X[f_j, c] * e^{+j2pi (f_j - f_c) tau_l}，tau_l = l / (N_zp * df)。
  **原生 range bin** = 1/(F*df)：本批 F=17、df=31.25 MHz 得 **1.8823529411764705 ns（双程 0.2822 m）**；
  无模糊窗 = 1/df = **32 ns**；物理分辨率尺度 = 1/((F-1)*df) = 2.0 ns。
  核用 (f_j - f_c) 基带形式，与绝对 f_j 只差逐 bin 公共相位 e^{+j2pi f_c tau_l}，对 |P| 与跨阵元比值
  无影响（自测 §4-2 已验证）。
* **A2 首阵元到原点补偿（只加一次）**：式(6) 的峰取自**第一个阵元**，其到达时延
  tau_ref = (2r - v_x,ref sin theta)/v_prop + system_delay，故
  **r = v_prop*(tau_ref - system_delay)/2 + v_x,ref*sin(theta_hat)/2**。
  本批 v_x,ref = -0.038 m：漏加 / 重复加的偏差分别约 +14.55 mm / -14.55 mm（自测 §4-4 精确量化）。
* **A3 fc 窄带空间相位与符号**：本机 theta = atan2(x, -y)（视轴 -Y），绝对虚拟坐标 v = tx + rx；
  相邻阵元比值 xi 的相角 = **+**2pi f_c dv_x sin theta / v_prop（与原文 pi sin theta **同号**）；
  当 dv_x = lambda_c/2 时 sin theta = angle(xi)/pi 与式(14) 完全一致。**相位绕回唯一性判据**：
  |dv_x sin theta| < lambda_c/2 等价于 |theta| < theta_wrap = asin(lambda_c/(2 dv_x))。
  本批 dv_x = 0.019 m、lambda_c/2 = 0.018767 m，故 theta_wrap = **81.0189 度**；strict 共同域几何
  半角 = atan2(2.2, 0.55) = **75.9638 度 < 81.0189 度**，故**共同域内相位唯一**：
  **不因 d > lambda_c/2 判 NA**，而是按限定共同域声明辨识（phase_uniqueness 块）。
  超域时仍按式(14) 主值解并申报 principal-branch 限制（不静默）。
* **A4 S（发射谱）**：原文式(5) 是**去斜后**的复指数模型（发射波形已移除）。故声明**非平坦** S 时按
  **反卷积**（X 列除以 S_f）处理；S 含零分量则**显式 failed**。**DOA 比值与 S 无关**（S 为逐频点
  公共因子，在跨阵元比值中约掉；自测 §4-8 验证相位差 < 1e-9 rad）。S = 1 是**共同纯相位假设**。
* **A5 多快照 = 式(8) 的统计扩展（声明）**：R_m = (1/L) sum_t P_m^(t)(P_m^(t))^H；L=1 时**逐字**回到
  P_m P_m^H；无 FB、无对角加载、无平滑；单源时仍 rank 1（慢时间随机相位被平均消去，自测 §4-6）。
  时延谱峰值检测用**逐快照功率的平均**（对随机相位稳健）。
* **A6 时延 32 ns 周期与整周期 unwrap**：搜索窗不是固定 [0, 1/df)，而是**物理 ref-delay 窗**
  [原点环 +- 首阵元双程余量 max|v_x|/v_prop]（含 system_delay），要求**窗宽 < 周期**，窗内 bin 用
  **整周期 unwrap** 的绝对时延标签（l 可小于 0 或大于等于周期 bin 数）。因此**周期端点不作为不可测域**、
  非零 system_delay 造成的跨周期部分**不静默丢失**；窗宽大于等于周期时显式 not_applicable。
  固有周期模糊（tau 与 tau + k/df 同值，端点与 DC 混叠）在 delay_periodicity 与
  selected_bins_period_alias_of_dc 如实申报，不做额外去混叠。
* **A7 平面 x-y 近似（声明）**：本批 v = tx + rx 的 v_y 全 0、v_z 恒 -0.02 m（**公共 z 偏移**）。
  原文式(12) 为 1-D ULA；本适配在 x-y 平面建模，**公共 z 不进入角度相位**（只作公共标量/公共距离偏移）。
  若 v_y 或 v_z 在阵元间**不一致**则 not_applicable（不做变体替换）。
* **A8 共同域（strict）**：bbox、距离环、FoV 一律只用 DOMAIN_EPS = 1e-9 m 的**数值容差**，
  **不做** half-cell / 半格外延 —— 与 b1_deng.py L655ff 的 _in_domain **实际实现逐项一致**。
  说明（**不改** B1/B2 历史代码）：B1/B2 的 FROZEN_PARAMS 里 domain_tolerance = half_cell 只是
  **陈旧标签**，其代码与文档均为严格边界；本文档**不沿用该标签**，避免写成半格。半平面 y<0 按物理约束。
  域外候选**不静默丢弃**，逐条记入 dropped_candidates；**原生解域外可 ok + 零个**。
* **A9 共享后处理**：**直接调用** common/q6_contract.py::postprocess_native_candidates
  （0.10 m、分值降序、确定性、cap=K、不用真值）；**不套用**二维全谱 3x3 + 0.3 阈值取峰口径
  （C3 不是二维全谱方法）；score = 参考阵元时延谱在该 bin 的**快照平均功率**（原生检测统计量）。
* **A10 时延锚点模型（显式声明）**：本实现是**逐阵元时延锚点**
  tau_c = 2r/v_prop - v_x,c sin theta/v_prop + system_delay，频域相位 **exp(-j2pi f tau_c)**
  （与 case.model.geometry 及 B1/B2 流形一致），故式(6) 的峰落在**参考阵元**时延上，需要 A2 补偿。
  **另一锚点** exp(-j2pi f tau_origin)*exp(+j2pi f_c v_x sin theta/v) 的峰位即 tau_origin，
  **不应**再加该补偿；两种锚点**不可混用**（自测 §4-5 用另一锚点数据验证了可预测的 -14.55 mm 偏差，
  两类锚点测试分开，不混）。
* **A11 数值失败与无能量**：输入能量为零/非有限则 **failed**（拒绝捏造正常结果）；被选距离峰**全部**
  发生 ESPRIT 数值失败则 **failed**（记录 algorithm_failures，沿 runner failed 口径保留分母，
  **允许退化、不强 K**）；部分失败则 ok + algorithm_failures 明细。

## 4. 独立 oracle 与自测结果（10 PASS / 0 FAIL）

独立口径：数据由测试文件按**手写模型**生成；时延谱对照走**逐 (j,l) 显式双重循环**；比值到角度用
**独立构造的精确窄带向量**闭式核对；补偿用**独立几何反解**核对。

| # | 用例 | 实测 |
|---|---|---|
| 1 | 解析声明（原生 bin / 无模糊窗 / 物理窗 unwrap / strict 域 / wrap 唯一 / 锚点 / 公共 z） | native range bin 0.28215760752941177 m；window [1.8732456438247018, 32.12675435617529] ns，宽 30.2535 ns < 32 ns；period bin 17；unwrap 索引 [1,17]；wrap 81.0189 度 > 域内 75.9638 度 得 唯一 True；domain_tolerance = strict(1e-9 m) |
| 2 | 时延谱 vs 显式逆 DFT | max rel diff **1.266e-14**；功率谱 = 逐快照功率平均（逐点一致）；周期等价核一致 |
| 3 | 式(13)(14) 比值到角度闭式 | sin 误差 **0.0**（30 度 / -45 度 / 0 度）；sin=1 越界时主值镜像为 -0.9754795699434888（exact phase 3.1805873382731695 rad） |
| 4 | 首阵元到原点补偿（只加一次） | r_true = r_impl = 1.748930202639563 m；phase_center 参考给 1.7502589092058822 m（差 1.33 mm）；漏加 +0.014554844419260471 m；重复加 -0.014554844419260693 m |
| 5 | 锚点模型边界（另一锚点，单独测） | 观测偏差 -0.014554844419260471 m vs 预测 -0.014554844419260582 m（差 < 1e-4） |
| 6 | rank-1 结构 | L=1 首特征向量 = 归一化 P_m（逐元素 1e-10 内）；rank 比值 L=1 5.76e16、L=128 1.94e16 |
| 7 | 同距离多目标 | 真实 bin 上**只给 1 个候选**（19.9085 度）；第二个被选峰为**周期端点混叠 bin**（已标记 alias，候选被 strict 域丢弃） |
| 8 | 频率原始序 vs 排序置换；zeropad 仅插值 | 位置差 **0.0 m**；原生 17 bins vs zeropad16 = 258 bins；两网格 r 差 0.1234 m <= 原生 0.2822 m |
| 9 | 状态路径 | 缺 X 得 failed；NaN 得 failed；k=0 得 not_applicable；F=1 得 NA；非等间隔得 NA；非 ULA 得 NA；spatial_subspace_dim=2 得 failed（拒绝增强）；未知参数得 failed；**无能量得 failed**；域外得 ok + 零候选；NMS 来源 = common/q6_contract |
| 10 | 真实缓存前 3 窗接口烟测 | indoor_C2-1_N2_S1-S2_001 的 t10/t20/t30：均 ok，1/2/1 个候选，n_candidates_evaluated = 27，单窗 1.3-2.1 ms（**不是性能结论**） |

覆盖范围：解析 CFR 合成（K=1/2/3、左右角、0.3-4.8 m 环、无噪与随机慢时间相位、非平坦 S、
另一锚点边界）、状态路径、真实缓存前 3 窗。
**不覆盖**：全部 32 录制 / 214 窗、近场强曲率、真实发射谱、密集多径、低 SNR。

## 5. 冻结参数与理由（默认值写定，不按结果调整）

| 参数 | 默认 | 理由 |
|---|---|---|
| delay_transform | idft_over_sorted_uniform_inband_frequencies | 式(6) 的 DFT 要求等间隔有序；本机原始序非单调 |
| zeropad_factor | **1（原生网格）** | 保持原文原始 N 点 DFT 索引口径：**原生 range bin 0.2822 m 如实声明**，不做调参；zeropad>1 仅把同一 Dirichlet 核插值到更密网格（**不新增信息、不提高分辨率**），启用须显式传入并会申报 |
| reference_channel | first_element（v_x 升序的首个，即原文 first antenna array） | 保原文 first antenna array；另提供 phase_center 仅作交叉核对 |
| spatial_subspace_dim | 1 | 式(8) 单向量外积 rank 1；大于 1 直接 failed（拒绝增强） |
| covariance_extension | snapshot_mean_outer_product | 式(8) 的统计扩展（L=1 逐字回到原文） |
| delay_power_aggregation | snapshot_mean_power | 对慢时间随机相位稳健 |
| nms_min_dist_m | 0.10 | 公共合同 SHARED_NMS_MIN_DIST_M（调用前核对一致） |
| spatial_phase_model | narrowband_at_center_frequency | 原文式(12) 的 e^{j pi k sin theta} 是窄带模型；FOV 内相位唯一性见 A3 |
| bin_mask / domain_tolerance | ring_span_strict / strict_1e-9 | 整周期 unwrap 的物理窗 + strict 共同域（A6/A8） |
| s_response | None（S=1 假设） | 非平坦 S 需显式声明并反卷积（A4） |

## 6. 已知局限（保留真实结果，不调参掩盖）

1. **固有周期模糊**：tau 与 tau + k/df（本批 32 ns）的 bin **同值**；本批环上限 32.0 ns 恰为周期端点，
   与 DC/低时延混叠。已按物理窗标签申报（delay_periodicity、selected_bins_period_alias_of_dc），
   不做额外去混叠。
2. **原生 TOA 量化**：默认网格步长即原生 range bin（双程 0.2822 m）；原文自述 1D-DFT 的 TOA 性能
   已饱和。本实现如实保留该量化，不靠 zeropad 制造精度。
3. **同距离多目标**：式(8) rank 1，每个距离峰只给 1 个角度；同距离多目标**不分离**（原限制）。
4. **窄带 fc 空间近似**：阵元相位取 f_c；当 dv_x = 0.019 m 略大于 lambda_c/2 = 0.018767 m 时，
   共同域内唯一（81.02 度 > 75.96 度）；若共同域几何半角超过 theta_wrap，则按原式给主值解并申报限制。
5. **公共 z 偏移**：v_z = -0.02 m 不进入角度相位（声明近似）。
6. **S 未声明时的偏差**：若实际发射谱非平坦而按 S=1 处理，时延峰位置会偏（角阶段不敏感）。
7. 本模块**未经独立方法审核**；本文件不构成验收记录。214 窗性能比较**未运行**。

## 7. 准确运行命令与产物哈希

工作目录 = 工作区根 D:/论文/小论文修改；BLAS/OMP 线程 = 1。命令（PowerShell）：

    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    & .venv/Scripts/python.exe -B 大修执行_20260923/04_Q6/hardware/methods/tests/test_c3_kim.py --json 大修执行_20260923/04_Q6/hardware/logs/C3_selfcheck_20260924.json --log 大修执行_20260923/04_Q6/hardware/logs/C3_selfcheck_20260924.log

日志与 JSON 由测试自身以 **UTF-8** 直接写盘（--log / --json），不经 PowerShell 重定向，
避免控制台代码页导致的中文乱码；控制台显示乱码不影响文件内容。

PDF 渲染核查件用 .venv 的 PyMuPDF 1.28.2 以 400/700 dpi 导出到 hardware/logs/C3_pdf_render/。

| 文件 | SHA256（2026-09-24） |
|---|---|
| hardware/methods/c3_kim.py | 9fc26e28852a88fe3d2cd1f449d52dbde4b39b2b54a1d346595c7070daadd694 |
| hardware/methods/tests/test_c3_kim.py | 1f71455babdca846832a995c9fbd5cd58040eacd209579c0a0f6815bf29f4384 |
| hardware/logs/C3_selfcheck_20260924.log | a7f81791a78ea03f6516ce5fff7af9ee77b00900c5053d8efbcaf87154417890 |
| hardware/logs/C3_selfcheck_20260924.json | 67ccd08a82e358742342ff813c108cbb091318267ffc54056e20d0e101f47032 |

## 8. 范围声明

* 本模块按本批口径实现与验证：**17 带内频点、4 虚拟通道、K <= 3、L = 128 快照、ring
  [0.299792458, 4.796679328] m、system_delay = 0、pair (0,2)(0,3)(1,2)(1,3)、
  v_x = [-0.038, -0.019, 0, 0.019] m、v_y = 0、v_z = -0.02 m**。不扩更多功能、不做跨数据集泛化声明。
* **只本地归档**：未改论文、未跑 214 窗正式性能、未动旧 B0/B1-B4/缓存/真值/P0b。
