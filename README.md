# DCS Analysis

面向 7.5--11.5 um 长波红外双梳光谱（LWIR-DCS）的 Python 数据处理基线。

当前实现覆盖：

- digitizer 导出的 NPZ/HDF5（HDF5 为可选依赖）数据接入；
- 两个 CW 参考波长的相对相位/时序估计；
- 按 interferogram 的时序和载波相位校正；
- 相干平均、RF 到光学波数映射、样品/背景透过率与吸光度；
- m-FID 风格的参考谱辅助基线抑制和稳健多项式基线；
- 基于 Voigt 线形的多组分拟合；
- 合成数据、命令行入口和单元测试。

## 快速开始

```powershell
python -m pip install -e ".[dev,io]"
python -m dcs_analysis.cli simulate --output data/demo.npz
python -m dcs_analysis.cli process data/demo.npz `
  --config configs/reference_lwir_dcs.toml `
  --output-dir outputs/demo
```

你的 80 MHz 系统使用独立的硬件配置模板：

```powershell
python -m dcs_analysis.cli inspect-config `
  --config configs/user_80mhz_ats9371_uhsm.toml
```

该模板已经写入 ATS9371（常用采样率 500 MS/s，硬件最高 1 GS/s、12 bit、±400 mV、50 Ω）和
UHSM-I-10.6（RF 输出 300 Hz--≥700 MHz、50 Ω）的参数。模板刻意不猜测
`delta_frep_hz`、`block_samples`、绝对波数锚点和两个 CW 参考波长；补齐这些参数前，真实数据处理会
给出明确错误，而不会沿用论文的 250 MHz / 1.45 kHz 数值。

如果只处理预先解调的参考相位，NPZ 至少包含：

```text
interferograms       shape = (n_interferograms, n_samples)
reference_phase_1    shape = (n_interferograms,)
reference_phase_2    shape = (n_interferograms,)
```

原始四路 CW beat note 也可以接入，格式和参数见
`src/dcs_analysis/io.py` 与 `configs/reference_lwir_dcs.toml`。

## 重要校准项

`optical_anchor_wavenumber_cm`、`rf_anchor_hz`、两个 CW 参考波长、beat-note 中心频率、采样率、`f_rep` 和 `Delta f_rep` 必须使用你实际系统的值。`reference_lwir_dcs.toml` 仅保留 Ahmad 等和 Moretti 等论文的参考架构；`user_80mhz_ats9371_uhsm.toml` 才是你的实机模板。

对于 80 MHz 重复频率，实值探测的一阶无混叠 RF 区间上限为
`f_rep / 2 = 40 MHz`，所以用户模板使用 39 MHz 作为带边保护。若希望单次覆盖
7.5--11.5 µm（约 13.9 THz），理论上要求 `Delta f_rep <= 约 230 Hz`；更大的
重复频率差需要缩窄光学带宽或分段测量。详见
[`docs/hardware_80mhz.md`](docs/hardware_80mhz.md)。

## 设计依据

详见 [`docs/architecture.md`](docs/architecture.md)。论文 PDF 只作为科研依据，不会被复制进仓库；代码也不会自动访问 HITRAN 或上传实验数据。

## 当前限制

这不是 ATS-SDK 的替代品。硬件触发、DMA 采集和 ATS9371 原始码格式仍需用你安装的 ATS-SDK 版本接入 `io.py`；当前代码处理已经导出的 NPZ/HDF5 数据。HITRAN 线表拟合需要用户自行准备线表 CSV，或安装并配置 HAPI。
