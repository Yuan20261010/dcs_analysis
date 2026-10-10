# LWIR-DCS 软件架构与科研依据

## 1. 目标

本项目实现的是“从 digitizer 数据到吸收谱/组分”的数字链路，而不是激光器、非线性晶体、MCT 探测器或采集卡的厂商控制程序。物理硬件由用户实验系统提供，软件通过 `Acquisition` 数据模型接收数据。

## 2. 与参考论文的对应关系

| 论文中的环节 | 软件模块 | 说明 |
|---|---|---|
| 两台 250 MHz Er:fiber comb + OP-GaP DFG | `io.py` 的输入契约 | DFG 光源本身不在离线软件中建模 |
| 500 MHz TE-cooled MCT 的时间域 IGM | `types.py`, `io.py` | 支持每个 interferogram 一行的二维数组 |
| 两个 CW 参考激光，各与两台 comb 产生 beat notes | `phase.py` | 四路原始 beat note 或已解调的两路相对相位 |
| 1.45 kHz repetition-rate offset | `spectrum.py` | RF 频率乘以 `f_rep / Delta f_rep` 映射到光学波数 |
| 相干平均 | `correction.py` | 先时序/载波相位校正，再复数平均 |
| m-FID / cepstral baseline 抑制 | `baseline.py` | 提供参考谱辅助的 m-FID 风格实现和稳健多项式基线兜底 |
| HITRAN 线逐线拟合 | `fit.py` | 用用户提供线表拟合，避免自动联网下载实验相关数据库 |

## 3. 关键物理映射

双梳的 RF beat frequency `f_RF` 与光学频率增量近似满足：

```text
Delta nu_opt = f_RF * f_rep / Delta f_rep
nu_opt       = nu_anchor + Delta nu_opt
wn_cm        = nu_opt / c_cm_s
```

因此 `f_rep = 250 MHz` 对应的光学 comb tooth spacing 约为 `0.00834 cm^-1`，与 Ahmad 等 2026 的 multispecies 测量一致。你的实机 `f_rep = 80 MHz`，齿间隔约为 `0.00267 cm^-1`。完整光学带宽与单次采集的瞬时带宽必须分开配置；`rf_max_hz` 只描述当前采集的瞬时 RF 带宽，并且本实现要求它不超过 `min(fs/2, f_rep/2, 模拟链路带宽)`。

## 4. 不能从论文自动确定的量

- `Delta f_rep`、触发相位和通道顺序；
- 四个 CW beat note 的中心频率、带宽、正负拍频约定；
- 绝对光频/波数锚点和 RF 正负号；
- MCT 的非线性、饱和和电子延迟；
- 你的气室长度、压力、温度、组分和 HITRAN 线表版本。

这些量在配置中显式保留，未校准时不应把输出当作定量结果。

## 5. 推荐的真实数据接入顺序

1. 先用 blank/background 数据验证中心 burst、采样率、`f_rep` 和 `Delta f_rep`。
2. 用单一气体和已知压力验证波数轴、线间距、基线和 Voigt 线形。
3. 再加入四路 CW beat note，比较校正前后的单个 IGM 相位残差。
4. 最后在多组分混合气体上拟合 mole fraction，并报告残差、压力、温度和线表版本。

## 参考资料

1. S. T. Ahmad et al., “Broadband long-wave infrared dual-comb spectroscopy for multi-species detection,” *Applied Optics* 65, 3961--3967 (2026), DOI: 10.1364/AO.592160.
2. L. Moretti et al., “Fast rate dual-comb spectrometer in the water-transparent 7.5--11.5 um region,” *Optics Letters* 49, 1844--1847 (2024), DOI: 10.1364/OL.515199.
3. I. Coddington, N. Newbury, and W. Swann, “Dual-comb spectroscopy,” *Optica* 3, 414--426 (2016).
4. D. Burghoff, N. Han, and J. H. Shin, “Generalized method for the computational phase correction of arbitrary dual comb signals,” *Optics Letters* 44, 2966--2969 (2019). The public `turbo-comb` repository is used only as an algorithmic reference, not as copied source.
