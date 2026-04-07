# 信号建模公式

## 使用的参数（详见 parameters.md）

- f_n：第n个信道中心频率 [Hz]
- g：信道间频率间隔 [Hz]
- P_0：单信道发射功率 [W]
- G_TX(f)：发射功率谱密度 [W/Hz]
- B_s：信号带宽 [Hz]
- T_0：信号周期 [s]，f_0 = 1/T_0 [Hz]

---

## 1. 离散信号建模（公式1.1.1）

假定信道全部功率集中于中心频率 f_n：

```
P(f) = P_0,   当 f = f_n
P(f) = 0,     当 f ≠ f_n
```

实现说明：信号用长度为 N_ch 的功率数组表示，每个元素对应一个信道。

---

## 2. 连续信号建模（公式1.2.1）

基于GN-Model，假设信号为复周期高斯白噪声过程，有平均PSD：

```
G_GN(f) = G_TX(f) × f_0 × Σ δ(f - i×f_0)
```

当周期 T_0 → ∞（即 f_0 → 0）时，离散谱线趋近于连续谱。

实现说明：实际计算中，G_TX(f) 用频率网格上的数组表示，每个元素是对应频率点的PSD值 [W/Hz]。

---

## 3. G_TX(f) 的三种建模方式

### 3.1 矩形信号谱

G_TX(f) 在信道带宽 B_s 内均匀分布：

```
对于第n个信道，中心频率 f_n，带宽 B_s：
G_TX(f) = P_ch / B_s,   当 |f - f_n| ≤ B_s/2
G_TX(f) = 0,             当 |f - f_n| > B_s/2
```

其中 P_ch 是该信道的总发射功率 [W]。

### 3.2 升余弦滚降谱

G_TX(f) 按升余弦滚降分布，滚降因子为 β_rolloff（0 ≤ β_rolloff ≤ 1）：

```
对于第n个信道，中心频率 f_n，符号速率 R_s = B_s：
Δf = |f - f_n|

当 Δf ≤ (1-β_rolloff)×R_s/2：
    G_TX(f) = P_ch / R_s

当 (1-β_rolloff)×R_s/2 < Δf ≤ (1+β_rolloff)×R_s/2：
    G_TX(f) = (P_ch / R_s) × 0.5 × (1 + cos(π/(β_rolloff×R_s) × (Δf - (1-β_rolloff)×R_s/2)))

当 Δf > (1+β_rolloff)×R_s/2：
    G_TX(f) = 0
```

注意：功率归一化条件 ∫G_TX(f)df = P_ch 需要验证。

### 3.3 OSA真实采样

直接读取OSA（光谱分析仪）返回的频谱采样数据作为 G_TX(f)。

**数据格式：** CSV文件，列为 `wavelength_nm, frequency_THz, power_dBm`。
示例数据位于 `data/osa/` 目录。

**加载流程：**
1. 读取CSV，提取 `frequency_THz` 和 `power_dBm` 列
2. 频率单位转换：f [Hz] = frequency_THz × 1e12
3. 功率单位转换：P_linear [W] = 10^(power_dBm / 10) × 1e-3
4. 计算PSD：G_osa [W/Hz] = P_linear / RBW（RBW为OSA分辨率带宽，需从仪器设置获取）
5. 使用 scipy.interpolate.interp1d 插值到计算频率网格

```
输入：CSV文件路径，RBW [Hz]（OSA分辨率带宽）
中间：f_osa [Hz]，G_osa [W/Hz]
输出：插值到计算频率网格上的 G_TX(f) [W/Hz]
```

---

## 4. 模型选择规则

- 如果信号建模采用离散模型，噪声计算也应选择离散模型
- 如果信号建模采用连续模型，噪声计算也应选择连续模型
- 离散模型在信道带宽 → 0 的极限下应退化为连续模型的特例

---

## 5. WDM频率网格生成

等间隔DWDM系统，N_ch 个信道，中心频率 f_center，间隔 g：

```python
f_channels = f_center + np.arange(-(N_ch-1)/2, (N_ch+1)/2) * g
```
