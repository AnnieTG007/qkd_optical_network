# FWM（四波混频）噪声建模公式

## 使用的参数（详见 parameters.md）

核心参数：
- α₁：信号光频率 f₁ 处的光纤衰减 [1/m]
- α₂, α₃, α₄：参与FWM的泵浦光频率对应衰减 [1/m]
- γ：非线性系数 [1/(W·m)]
- D：简并因子（ω₃=ω₄ 时 D=3，ω₃≠ω₄ 时 D=6）
- Δα：衰减失配 = α₃+α₄+α₂-α₁ [1/m]
- Δβ：相位失配 [rad/m]
- P_0：单信道发射功率 [W]
- S：瑞利散射捕获因子 [1/m²]
- α_R：瑞利散射衰减 [1/m]
- L：光纤长度 [m]

离散模型额外参数：
- P_2(0), P_3(0), P_4(0)：泵浦光发射功率 [W]

连续模型额外参数：
- G_{TX}(f)：发射功率谱密度 [W/Hz]

---

## 1. FWM效率因子（公式2.1.1）

FWM效率因子为：

$$
\eta = \frac{e^{-\Delta\alpha \cdot z} - 2e^{-\frac{\Delta\alpha}{2}z}\cos(\Delta\beta \cdot z) + 1}{\frac{(\Delta\alpha)^2}{4} + (\Delta\beta)^2}
$$

其中相位匹配因子 Δβ 的二阶展开近似为：

$$
\Delta\beta = \frac{2\pi\lambda^2}{c} |f_3 - f_2| \cdot |f_4 - f_2| \left[ D_c + \frac{\lambda^2}{2c}(|f_3 - f_2| + |f_4 - f_2|)\frac{dD_c}{d\lambda} \right]
$$

其中：
- $f_2, f_3, f_4$：参与FWM过程的三个信道频率 [Hz]
- $D_c$：色散系数 [s/m²]
- $\frac{dD_c}{d\lambda}$：色散斜率 [s/m³]

---

## 2. 离散FWM噪声模型

### 2.1 前向FWM噪声（公式2.2.1）

对于频率 ω₁ 处的前向FWM噪声功率，有：

$$
P_{f,1}(z) = \frac{e^{-\alpha_1 z}\gamma^2}{9} \sum_{\omega_3}\sum_{\omega_4} \eta \cdot D^2 \cdot P_2(0) \cdot P_3(0) \cdot P_4(0)
$$

约束条件：$\omega_2 \neq \omega_3$ 且 $\omega_2 \neq \omega_4$，即排除SPM/XPM项。

$P_{2,(3,4)}(0)$ 表示泵浦光的发射功率 [W]。

对于前向FWM噪声，最后应取光纤接收端功率，即 $P_{f,1}(L)$ [W]。

### 2.2 后向FWM噪声（公式2.2.4）

后向FWM噪声被视为前向FWM噪声经瑞利散射的再分布：

$$
P_{b,1}(z) = S \cdot \alpha_R \int_z^L P_{f,1}(l) e^{-\alpha(l-z)}\ dl
$$

其中 $S$ 为瑞利散射捕获因子 [1/m²]，$\alpha_R$ 为瑞利散射衰减 [1/m]，两者的乘积 $S\alpha_R$ 称为瑞利散射系数 [1/m³]。

对上式积分整理得（公式2.2.5）：

$$
P_{b,1}(z) = \frac{S\alpha_R \gamma^2}{9} \sum_{\omega_3}\sum_{\omega_4} D^2 P_2(0) P_3(0) P_4(0) \cdot [F(L) - F(z)]
$$

其中原函数 $F(l)$ 表示为：

$$
F(l) = \frac{e^{\alpha_1z}}{\frac{(\Delta\alpha)^2}{4}+(\Delta\beta)^2}[-\frac{e^{-Al}}{A}-\frac{e^{-Bl}}{B^2+\Delta\beta^2}(-Bcos(\beta l)+\beta sin(\beta l))-\frac{e^{-Cl}}{C}]
$$

辅助变量定义：

$$
\begin{aligned}
A &= \Delta\alpha + 2\alpha_1 \\
B &= \frac{\Delta\alpha}{2} + 2\alpha_1 \\
C &= 2\alpha_1
\end{aligned}
$$

**注意**：分子中的 $z_{\text{obs}}$ 为观测位置（噪声评估位置），不是积分变量。当计算 $P_{b,1}(0)$ 时，$z_{\text{obs}} = 0$，$e^{\alpha_1 \cdot 0} = 1$ 自动消去。

对于后向FWM噪声，最后应取光纤发射端功率，即 $P_{b,1}(0)$ [W]。

---

## 3. 连续FWM噪声模型

### 3.1 前向FWM噪声（公式2.3.1）

对于积分区域 $(f_3, f_4)$ 而言，简并项只是区域中满足 $f_3 = f_4$ 的一条对角线，在连续积分中会被忽略，无需额外考虑简并项贡献。求解 ω₁ 处的FWM噪声PSD，有：

$$
G_{f,1}(z) = 4 \cdot \frac{\gamma^2 e^{-\alpha_1 z}}{9} \iint_{-\infty}^{+\infty} D^2 \cdot \eta \cdot G_{\text{TX}}(f_i) G_{\text{TX}}(f_j) G_{\text{TX}}(f_k)\ df_i df_j
$$

其中 $G_{\text{TX}}(f)$ 表示频率 $f$ 处的发射功率谱密度 [W/Hz]。

对于前向FWM噪声，最后应取光纤接收端功率，即 $G_{f,1}(L)$ [W/Hz]。

### 3.2 后向FWM噪声（公式2.3.2/2.3.3）

同理，后向FWM噪声为：

$$
G_{b,1}(z) = S\alpha_R \int_z^L G_{f,1}(l) e^{-\alpha(l-z)}\ dl
$$

整理得：

$$
G_{b,1}(z) = \frac{\gamma^2}{9} \iint_{-\infty}^{+\infty} D^2 G_{\text{TX}}(f_i) G_{\text{TX}}(f_j) G_{\text{TX}}(f_k) \cdot [F(L) - F(z)]\ df_i df_j
$$

其中原函数 $F(l)$ 与离散模型形式相同（见公式2.2.6）。

对于后向FWM噪声，最后应取光纤发射端功率谱密度，即 $G_{b,1}(0)$ [W/Hz]。

### 3.3 信道积分噪声（公式2.3.6）

若需要求解落在频率范围为 $[B_l, B_r]$ 某信道的噪声功率 [W]，应有：

$$
P_{\text{noise}} = \int_{B_l}^{B_r} G_{\text{noise}}(f)\ df
$$

其中 $P_{\text{noise}}$ 代表落在信道频率范围内的噪声功率 [W]，$G_{\text{noise}}$ 代表频率为 $f$ 处的噪声功率密度 [W/Hz]。

---

## 4. 交叉验证关系

离散模型和连续模型的关系：
- 当信道带宽 $B_s \to 0$ 时，连续模型的积分退化为离散模型的求和
- 测试方法：用极窄矩形谱的连续模型对比等功率的离散模型

数值验证检查项：
- FWM噪声功率应随泵浦功率的平方增长（$P^2$ 关系）
- FWM效率 η 随地距长度增加而下降
- 简并FWM（D=3）的贡献约为非简并FWM（D=6）的一半

---

## 5. 推导来源

主要参考：
- Poggiolini et al., "The GN Model of Fiber Nonlinear Propagation", arXiv:1209.0394, Eq.120+123
- Gao et al., JLT 2025, doi:10.1109/JLT.2025.3610854（后向FWM积分原函数 F(l) 推导）
