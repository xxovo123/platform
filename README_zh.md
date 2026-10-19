# 单锚IR-UWB雷达论文复现仓库

**完整复现必须在Linux环境运行。** 根据 `environment.yml` 创建独立Conda环境，原生Windows不是完整流程的支持环境。当前标识为 `revision-20261019-candidate`，压缩包名为 `platform_validation_20261019.zip`。本轮问题已修复，未重新复验；原包条件数值复现通过，证据范围见[验证状态](docs/validation_status_20261019.md)。

英文[README](README.md)说明平台与全部命令；[Linux验证指南](docs/linux_validation.md)规定环境、顺序、比对对象、容差和报告；[图表对应表](docs/manuscript_reproduction.md)逐项映射图1–8及表I–IX。

- 硬件原始数据：完整205段、31次记录、15种配置、338个目标实例；连续EMA的段前状态、标定和真值齐全。
- 硬件运行：6种方法和20组消融，共205×26=5330次；稳定性使用固定30组×4段=120段。
- 噪声/快照：4个非间距实验族、150条件、16500条记录；E4_A为保留的参数化参考，正文图5使用FDTD E4_B。
- 图4：仅FDTD，3方向×8间距×100次=24条件/2400次；间距0.15、0.20、0.25、0.30、0.40、0.50、0.75、1.00 m，搜索网格0.05 m。
- 对应实验：15份FDTD CIR、205份实测谱、300次参数化参照及4100条参照—实测比较。
- 参数化目标间距实验不在本包。正文噪声定位和对应实验的参数化模型仍保留。

收到新ZIP后，在全新工作目录中解压：

```bash
unzip -O UTF-8 platform_validation_20261019.zip
cd platform
```

若 `unzip` 不支持 `-O`，改用 `python3 -m zipfile -e platform_validation_20261019.zip .`，再进入 `platform/`。保留UTF-8文件名；旧验收中裸 `unzip` 曾导致中文文件名乱码。

从仓库根目录按[Linux验证指南](docs/linux_validation.md)创建并激活新环境。Conda包精确固定为服务器此前条件运行实际使用的 **`pymeep=1.30.1=nompi_py312hfc0f314_104`**，其模块自报 **`meep.__version__=1.30.0`**，两种版本号须分别记录；Python **3.12.11**、NumPy **2.3.0**保持不变。设置单线程BLAS后执行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
python validation/run_validation.py --mode saved --out generated_validation/saved
python validation/run_validation.py --mode replay --out generated_validation/replay --workers 4
python validation/regenerate_inputs.py --out generated_validation/fields --workers 4
python validation/run_validation.py --mode replay --out generated_validation/from_fields --workers 4 \
  --generated-index generated_validation/fields/generated_index.json
```

第一步重算已有记录与图表；第二步从发布的CIR重跑全部估计器；第三步新生成5份参数化CIR和44份FDTD序列；第四步强制用新仿真输入重跑下游。四步分别报告，重画图不能算实验重跑，CIR重放不能算电磁场求解。断点恢复用 `--resume`，不得导入旧科研缓存。保留失败项、日志和环境，不能为通过验收调阈值、换参考结果或删条件。

硬件预检使用 `hardware_reproduction/data/preprocessing_reference.npz` 中冻结的全部205段FFT数组。归档时已确认205个数组均与既有 `X_sha256` 逐字节一致，此步骤未执行估计器。预检逐元素比较使用 `max_abs<=1e-12`、`rtol=0`。现有raw文件、连续EMA状态及source哈希仍严格核对，原 `X_hash` 保留为诊断。该修复不改变论文指标或估计器。

图4衡量最终取峰前的真值辅助双峰对应率，不是最终定位成功率；原输出与谱峰诊断分别保存。预期数值见 [expected_results.json](validation/expected_results.json)。PDF不要求跨环境字节一致，但8张图须与 `validation/reference/figures/` 核对坐标、数据、标记、图例和裁切。照片及图1为静态源文件，其余示意图有生成代码。
