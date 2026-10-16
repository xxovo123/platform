# 从原始CIR构建case并运行JSF-MUSIC

以下路径相对仓库根；模块是公开包内文件，不需要工作区原目录或远程服务器。

| 接口/输入 | 可读路径 |
|---|---|
| `dataset()`, `features(entry)`, `make_case(entry, contexts)` | `hardware_reproduction/reproduce_hardware.py` |
| 全205段索引与原始CIR | `hardware_reproduction/data/raw/index.json`、`data/raw/segments/*.npz` |
| 31份录制设置 | `hardware_reproduction/data/raw/recordings/*.json` |
| 每段开始前连续EMA状态 | `hardware_reproduction/data/ema_states/<segment_id>.npz` |
| 几何、频轴、校准、网格和方法配置 | `hardware_reproduction/data/contexts.json` |
| `estimate(case)`、`music_image_from_X(case, X)` | `hardware_reproduction/source/大修执行_20260923/04_Q6/common/q6_methods_this_method.py` |
| 协议/统一取峰 | `hardware_reproduction/source/大修执行_20260923/01_投稿口径/p0b/protocol/knownk_protocol.json` 与 `driver/knownk_eval/peaks.py` |

在 `hardware_reproduction/` 中运行：

```python
import reproduce_hardware as hw

entry = hw.dataset()[0]  # 保持完整索引中的segment_id，勿按配置聚合替代
contexts = hw.read(hw.DATA / 'contexts.json')
case = hw.make_case(entry, contexts)
method = hw.module('public_hardware_this_method',
                   hw.COMMON / 'q6_methods_this_method.py')
result = method.estimate(case)
if result['status'] != 'ok':
    raise RuntimeError(result['diagnostics'])
spectrum = result['spectrum']
predictions = result['predictions']  # 每项x/y/score
assert spectrum.shape == (len(case['grid_y']), len(case['grid_x']))
```

`features(entry)` 从 `[128,8,32]` complex64原始CIR出发，恢复段前complex128背景状态与逐通道首次样本掩码，按录制元数据逐帧EMA更新、裁抽头、complex128 FFT、选频点/通道及数据侧相位校准。返回 `case['X']` 为 `[128,68]` complex128；列序为 `f*C+channel`，频点保持原FFT索引顺序。它与论文列快照表示互为排列约定，不应再转置传入估计器，也不要再次扣背景、校相位或增加时延补偿。205段均须从自己的状态恢复，不能将独立段的128帧视为录制起点。

`make_case` 提供 `X`, `k`, `grid_x/y`, `frequencies_hz`, `pairs`, `tx_positions`, `rx_positions`, `config`, `model`, `search` 及身份元数据；`snapshots=None`、`snapshots_available=False`。目标数K是已知输入，真值坐标不进入case。估计器拒绝真值字段。`estimate` 完成协方差、FB、加载、EVD、双程几何谱和统一取峰，返回 `status`, `predictions`, **未归一化完整网格 `spectrum`**、`diagnostics`。只需谱时可调用 `image, k, info = method.music_image_from_X(case, case['X'])`。不要调用需要时域 `snapshots` 的旧参考接口 `hardware_spectrum(case)`。

谱数组按 `[y,x]` 排列，网格单位m，正前方 `y<0`。角度为 `atan2(x,-y)`，径向/FOV掩膜外谱值为0。Q4核验应按每个谱自身最大值归一化，保持网格/有效点掩膜、零响应、segment_id与原对应表一致，不能根据谱非零值临时改变有效点定义。谱峰和真值应在估计完成后单独用于配对。Q4的混合域统计与绘图由主代理负责。

可直接导出全部205份硬件谱（不运行仿真）：

```text
python -B export_hardware_spectra.py --out generated/raw_spectra
```

先做一段接口检查可加 `--limit 1`。每份NPZ包含 `spectrum`, `grid_x`, `grid_y`, `predictions`, `truth`, `segment_id`, `recording_id`, `config_id`, `environment`；真值仅在估计后附入导出文件，不能回传估计器。导出脚本核对全部预测与公开冻结结果，失败即中断。`manifest.json`记录逐段raw、EMA、X和输出文件哈希。导出谱是新计算结果，不宣称已通过Q4逐格谱匹配；该核验由主代理执行。
