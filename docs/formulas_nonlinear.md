# GN-Model 经典信道非线性干扰（NLI）噪声公式

## 使用的参数（详见 parameters.md）

核心参数：
- α：光纤衰减系数 [1/m]（C 波段近似为常数）
- γ：非线性系数 [1/(W·m)]
- β₂：二阶色散系数 [s²/m]（与 D_c 的关系见下文）
- L：光纤长度 [m]
- G_TX(f)：信道发射功率谱密度 [W/Hz]
- S：瑞利散射捕获因子 [1/m²]
- α_R：瑞利散射衰减 [1/m]
- S·α_R：瑞利散射系数（fiber.rayleigh_coeff）[1/m³]

---

## 1. 物理背景

GN-model（Gaussian Noise Model，Poggiolini et al., arXiv:1209.0394）将光纤 Kerr 非线性视为等效加性高斯噪声源。

非线性干扰（NLI）的本质是 FWM（四波混频）效应在 WDM 信道重叠频谱上的累积：

$$
G_{\text{NLI}}(f) = \frac{16}{27} \gamma^2 \iint G_{\text{TX}}(f_1) G_{\text{TX}}(f_2) G_{\text{TX}}(f_1 + f_2 - f) \left|\mu(f, f_1, f_2)\right|^2 df_1 df_2
$$

**与量子信道 FWM 噪声的关系**：
- 物理机制相同（四波混频）
- 目标信道不同：NLI 作用于经典信道自身；量子信道 FWM 作用于量子信道
- 积分形式相同，但目标频率 f 是经典信道频率而非量子信道

---

## 2. FWM 效率因子 |μ|²（单跨段，等损耗近似）

### 2.1 一般形式

$$
\left|\mu\right|^2 = \frac{e^{-2\alpha L} - 2e^{-\alpha L}\cos(\Delta\beta L) + 1}{\alpha^2 + \Delta\beta^2}
$$

**与离散 FWM η 的关系**：
在 C 波段近似（所有信道衰减相同 = α）下：

$$
\Delta\alpha = \alpha_k + \alpha_i + \alpha_j - \alpha_1 = 4\alpha - 2\alpha = 2\alpha
$$

代入公式 2.1.1（formulas_fwm.md）：

$$
\eta = \frac{e^{-2\alpha L} - 2e^{-\alpha L}\cos(\Delta\beta L) + 1}{(\alpha)^2 + \Delta\beta^2} = |\mu|^2
$$

因此 $|\mu|^2 =$ `_fwm_efficiency(Δα=2α, Δβ, L)`，可直接复用现有代码。

### 2.2 相位失配 Δβ（β₂ 近似）

在 GN-model 中，假设所有频率接近（窄带近似），使用二阶 Taylor 展开：

$$
\Delta\beta = 4\pi^2 \beta_2 (f_1 - f)(f_2 - f)
$$

其中 β₂ [s²/m] 由色散系数 D_c 反推：

$$
D_c = -\frac{2\pi}{c}\frac{d^2\beta}{d\omega^2} \Rightarrow \beta_2 = \frac{d^2\beta}{d\omega^2} = -\frac{D_c \lambda^2}{2\pi c}
$$

参考频率取 C 波段中心（193.5 THz ≈ 1549.3 nm）。

**与 formulas_fwm.md 2.2.3 节公式的关系**：
- GN-model β₂ 近似：忽略色散斜率项，仅保留 β₂
- formulas_fwm.md 使用完整 Taylor 展开（含 D_slope 项），适用于大频差场景
- 对于 C 波段内 WDM（信道间隔 50 GHz，Δf/f < 3%），两者差异 < 5%

---

## 3. 前向 NLI PSD（公式 4.1）

$$
G_{\text{NLI}}(f) = \frac{16}{27} \gamma^2 \iint_{-\infty}^{+\infty} G_{\text{TX}}(f_1) G_{\text{TX}}(f_2) G_{\text{TX}}(f_1 + f_2 - f) \left|\mu(f, f_1, f_2)\right|^2 df_1 df_2
$$

**系数 (16/27) 的来源**：
Poggiolini Eq.1 中包含 16/27 的系数，来自对 D²·η 在所有 (f₁, f₂) 对上的统计平均。

---

## 4. 后向 NLI PSD（弱瑞利近似）

后向 NLI 由瑞利散射重分配产生（与量子信道后向 FWM 相同）：

$$
G_{\text{NLI},\text{bwd}}(f) = S \cdot \alpha_R \int_0^L G_{\text{NLI}}(f, l) e^{-\alpha l} dl
$$

弱瑞利近似下（G_NLI 在光纤内变化缓慢）：

$$
G_{\text{NLI},\text{bwd}}(f) \approx S \cdot \alpha_R \cdot L_{\text{eff}} \cdot G_{\text{NLI}}(f)
$$

其中 $L_{\text{eff}} = (1 - e^{-\alpha L})/\alpha$ [m]。

---

## 5. 信道积分 NLI 功率

若需要求解落在频率范围为 $[f_c - B_s/2, f_c + B_s/2]$ 的信道 NLI 功率 [W]：

$$
P_{\text{NLI},\text{ch}} = \int_{f_c - B_s/2}^{f_c + B_s/2} G_{\text{NLI}}(f) df
$$

其中 $f_c$ 是信道中心频率，$B_s$ 是信号带宽。

---

## 6. 单信道闭式解（公式 4.2，Eq.120）

对于单个矩形谱信道（G_TX = P_ch/B_s，|f-f_c| ≤ B_s/2），NLI 可积分为闭式：

$$
P_{\text{NLI}} = \frac{8}{27} \gamma^2 G^3 L_{\text{eff}}^2 \frac{\text{asinh}\left(\frac{\pi^2 |\beta_2| L_{\text{eff,a}} B_s^2}{2}\right)}{\pi |\beta_2|}
$$

其中：
- $G = P_{\text{ch}}/B_s$ [W/Hz]
- $L_{\text{eff}} = (1 - e^{-\alpha L})/\alpha$ [m]
- $L_{\text{eff,a}} = 1/(2\alpha)$ [m]（渐近有效长度）

**物理意义**：
- asinh 项：体现相位失配的累积效应
- 当 $|\beta_2| L_{\text{eff,a}} B_s^2 \gg 1$（宽带信道），$\text{asinh}(x) \approx \ln(2x)$，NLI ∝ $B_s^2 \ln(B_s)$
- 当 $|\beta_2| L_{\text{eff,a}} B_s^2 \ll 1$（窄带信道），$\text{asinh}(x) \approx x$，NLI ∝ $B_s^3$

---

## 7. 数值积分实现

### 7.1 离散求和近似

使用均匀网格黎曼和：

$$
G_{\text{NLI}}(f_k) \approx \frac{16}{27} \gamma^2 \sum_{i,j} G_{\text{TX}}(f_i) G_{\text{TX}}(f_j) G_{\text{TX}}(f_i + f_j - f_k) |\mu|^2 \cdot \Delta f^2
$$

其中 Δf 是频率网格步长。

### 7.2 有效频率对筛选

G_TX(f) > 0 的频点为"有效"泵浦点。设有效点数为 N_a，则泵浦对数为 N_a²。

### 7.3 内存优化

当 N_a² × N_f 过大时（> 50M），使用分块处理：
- 每次处理 N_f 的子块（chunk_size = 10-50）
- float32 替代 float64 以节省 50% 内存

---

## 8. 交叉验证关系

- **与离散 FWM 的极限关系**：当所有信道为 SINGLE_FREQ（极窄带宽）时，NLI 积分退化为离散 FWM 求和
- **功率三次方关系**：NLI ∝ P_ch³（G_TX 的三次方来自双重积分）
- **与 Eq.120 的关系**：单信道数值积分结果应比 Eq.120 高 10-30%（Eq.120 是近似）
- **对称性**：G_NLI(f) 关于信道中心对称

---

## 9. 多跨段扩展

对于 N 跨段系统，总 NLI PSD 为各跨段非相干累加：

$$
G_{\text{NLI},\text{total}} = N \times G_{\text{NLI},\text{single}}
$$

注意：各跨段噪声功率直接相加（而非功率相加），因为各跨段噪声不相关。

---

## 10. 参考文献

- Poggiolini et al., "The GN Model of Fiber Nonlinear Propagation", arXiv:1209.0394, 2012
- Carena et al., "GN Model of Fiber-Optic Nonlinearities", J. Lightw. Technol., vol. 30, no. 4, 2012
- Second edition (2024): arXiv:1211.5516
