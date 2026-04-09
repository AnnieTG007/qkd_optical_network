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

$$
P(f) = P_0, \quad \text{当}\ f = f_n
$$

$$
P(f) = 0, \quad \text{当}\ f \neq f_n
$$

实现说明：信号用长度为 N_ch 的功率数组表示，每个元素对应一个信道。

**绘图说明（2026-04 修正）**：离散信道没有 PSD 概念，只有信道功率 P。
绘图时，在 f_n 处画高度为 P_ch 的竖线（stem），不乘以 1/Δf。
连续模型的绘图值为 PSD × Δf [W]（bin 功率），与离散信道的功率 [W] 量纲统一，可以直观比较。

---

## 2. 连续信号建模（公式1.2.1）

基于GN-Model，假设信号为复周期高斯白噪声过程，有平均PSD：

$$
G_{\text{GN}}(f) = G_{\text{TX}}(f) \times f_0 \times \sum_i \delta(f - i \times f_0)
$$

当周期 T_0 → ∞（即 f_0 → 0）时，离散谱线趋近于连续谱。

实现说明：实际计算中，G_TX(f) 用频率网格上的数组表示，每个元素是对应频率点的PSD值 [W/Hz]。

---

## 3. G_TX(f) 的两种建模方式

### 3.1 升余弦滚降谱（矩形谱为 β=0 的特例）

G_TX(f) 按升余弦滚降分布，滚降因子为 β_rolloff（0 ≤ β_rolloff ≤ 1）：

$$
\text{对于第n个信道，中心频率}\ f_n,\ \text{符号速率}\ R_s = B_s：
$$

$$
\Delta f = |f - f_n|
$$

**当** $\Delta f \leq (1-\beta_{\text{rolloff}}) \times R_s / 2$：

$$
G_{\text{TX}}(f) = \frac{P_{\text{ch}}}{R_s}
$$

**当** $(1-\beta_{\text{rolloff}}) \times R_s / 2 < \Delta f \leq (1+\beta_{\text{rolloff}}) \times R_s / 2$：

$$
G_{\text{TX}}(f) = \frac{P_{\text{ch}}}{R_s} \times 0.5 \times \left[1 + \cos\left(\frac{\pi}{\beta_{\text{rolloff}} \times R_s} \times \left(\Delta f - \frac{(1-\beta_{\text{rolloff}}) \times R_s}{2}\right)\right)\right]
$$

**当** $\Delta f > (1+\beta_{\text{rolloff}}) \times R_s / 2$：

$$
G_{\text{TX}}(f) = 0
$$

**矩形谱**：当 β_rolloff = 0 时，上式退化为矩形谱，即 $G_{\text{TX}}(f) = P_{\text{ch}} / B_s$（$|f-f_n| \leq B_s/2$）。

**峰值高度**：
  - 升余弦谱的峰值 PSD = P_ch / B_s（与滚降系数 β 无关）
  - 不同 β 值的差异体现在频谱形状：平坦区宽度由 $(1-\beta) \times B_s/2$ 变为（β=0 时完整 B_s），
    滚降区宽度由 0 变为 $\beta \times B_s/2$（总占用带宽 $(1+\beta) \times B_s/2$）
  - 矩形谱（β=0）和升余弦谱（β>0）在中心频率处具有相同的峰值 P_ch/B_s

注意：功率归一化条件 $\int G_{\text{TX}}(f) df = P_{\text{ch}}$ 需要验证。

### 3.2 OSA真实采样

直接读取OSA（光谱分析仪）返回的频谱采样数据作为 G_TX(f)。

**数据格式：** CSV文件，列为 `wavelength_nm, frequency_THz, power_dBm`。
示例数据位于 `data/osa/` 目录。

**加载流程：**
1. 读取CSV，提取 `frequency_THz` 和 `power_dBm` 列
2. 频率单位转换：$f\ [\text{Hz}] = \text{frequency\_THz} \times 10^{12}$
3. 功率单位转换：$P_{\text{linear}}\ [\text{W}] = 10^{\text{power\_dBm} / 10} \times 10^{-3}$
4. 计算PSD：$G_{\text{osa}}\ [\text{W/Hz}] = P_{\text{linear}} / \text{RBW}$（RBW为OSA分辨率带宽，需从仪器设置获取）
5. 使用 scipy.interpolate.interp1d 插值到计算频率网格

---

## 4. 模型选择规则

- 如果信号建模采用离散模型，噪声计算也应选择离散模型
- 如果信号建模采用连续模型，噪声计算也应选择连续模型
- 离散模型在信道带宽 → 0 的极限下应退化为连续模型的特例

---

## 5. WDM频率网格生成

等间隔DWDM系统，N_ch 个信道，中心频率 f_center，间隔 g：

$$
f_{\text{channels}} = f_{\text{center}} + \text{np.arange}\left(-\frac{N_{\text{ch}}-1}{2}, \frac{N_{\text{ch}}+1}{2}\right) \times g
$$
