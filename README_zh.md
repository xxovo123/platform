# 仿真代码与原始 CIR 数据

本仓库对应论文《Joint Range-Azimuth Estimation for Single-Anchor IR-UWB Radar: Simulation, Algorithm, and Experimental Validation》的大修版本。

代码为本次大修实际使用的仿真平台与 JSF-MUSIC 处理链。数据包括精选原始场样例、全部小体积正式仿真 CIR、完整正式仿真结果，以及论文使用的全部 **205 段硬件原始 CIR**。硬件数据来自采集文件，保留背景扣除、截窗、FFT 和相位校准之前的复数抽头；没有用处理后快照替代原始数据。硬件采集及实验处理程序不另行发布。

## 目录

| 目录 | 用途 |
|---|---|
| `src/` | 远程正式仿真源码及其数值依赖 |
| `data/cache/fdtd/` | 14 组正式 FDTD CIR，各有 128 个真实状态 |
| `data/cache/analytic/` | 29 组解析模型 CIR |
| `data/field_samples/` | S1、S2、S5 的空场和初始目标状态，共 6 个原始场样例 |
| `data/results/runs/` | 六组实验的全部 24,150 条正式记录 |
| `data/q4/` | 仿真—实测对应数据、已保存谱及参数化对照 |
| `data/hardware/` | 完整 205 段硬件原始 CIR、段索引、真值及采集参数 |
| `data/settings/` | 正式协议、环境及源脉冲参考 |
| `data/tables/` | 正式统计及采用论文 SNR 标注的表格 |

数组维度、单位、文件关联和读取例子见 [data/README.md](data/README.md)；硬件数据另见 [data/hardware/README.md](data/hardware/README.md)。

## 使用

在仓库根目录安装 `requirements.txt` 中的依赖后：

```bash
python reproduce.py verify
python reproduce.py tables --out outputs/tables
python reproduce.py figures --out outputs/figures
python reproduce.py replay --layout S2
python q4_reproduce.py summarize
python q4_reproduce.py replay --config G-420 --trial 0
```

依次完成源文件及数据核验、从保存预测重算统计、再生三幅仿真曲线、复现一条正式定位试验、重算仿真—实测对应汇总，以及复现一个固定种子的参数化样例。这些命令不启动 FDTD 求解；输出与原数据分开保存。

需要从头生成 FDTD 时，在 Linux 上按 `environment.yml` 建立 Meep 环境，再执行：

```bash
python regenerate_fdtd.py --clean-id cln_de385a3b2982 --out outputs/new_fdtd_S2 --workers 1
```

完整生成会进行实际场求解。`--max-states` 可用于缩短演示，但不能把其输出当作论文的完整 128 状态数据。

## 对照口径

原始运行记录的 `snr_db` 是生成参数，论文使用的期望 CIR 信噪比为它加 `3.0102999566 dB`；实际有限样本信噪比另在每次记录中保存。保留这一换算及原实现，才能对上论文曲线和固定种子结果。

定位 RMSE 对已配对误差池化计算，不用距离门限删去大误差。漏出的真值不进入 RMSE；检出率采用另一次 0.30 m 门限匹配。噪声试验共享各自的无噪序列，快照试验使用同一序列的真实前缀。

本次发布按要求精选了大型中间场数据，未复制远程全部场缓存；小体积正式 CIR 和数值结果完整保留。硬件数据完整覆盖论文的 205 段评估集，室内 103 段、室外 102 段。硬件实验处理代码不在发布范围，因此本仓库不声称包含硬件五基线与消融的完整自动执行流程。
