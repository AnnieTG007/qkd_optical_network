# BB84 QKD 安全码率公式

## 使用的参数（详见 parameters.md）

| 符号 | 说明 | 单位 |
|------|------|------|
| $\eta_{\text{spd}}$ | SPD 量子效率 | — |
| $IL$ | 插入损耗（线性值） | — |
| $\alpha$ | 光纤衰减系数 | 1/m |
| $z$ | 光纤传输距离 | m |
| $p_{\text{dark}}$ | 暗计数概率/脉冲 | — |
| $p_{\text{noise}}$ | 噪声光子计数概率/脉冲（SpRS 等） | — |
| $\mu$ | 信号态平均光子数 | — |
| $\nu$ | 诱骗态平均光子数 | — |
| $e_{\text{Det}}$ | 探测器本征误码率 | — |
| $f_e$ | 纠错效率 | — |
| $q$ | 筛选效率（BB84 = 0.5） | — |
| $R_{\text{rep}}$ | 脉冲重复率 | Hz |
| $p_\mu$ | 信号态发送概率 | — |
| $p_\nu$ | 诱骗态发送概率 | — |
| $N_{\text{pulse}}$ | 总脉冲数 | — |
| $\gamma_{ks}$ | Gaussian 置信倍数 | — |

---

## 公共信道量（三种模型共用）

### 信道透过率（公式 10）

$$
\eta = \eta_{\text{spd}} \cdot e^{-\alpha z} \cdot IL
$$

### 真空态产率（公式 4）

$$
Y_0 = 1 - (1 - p_{\text{dark}} - p_{\text{noise}})^2
$$

### 给定强度 $\mu$ 的总增益与误码率（公式 8, 9）

$$
Q_\mu = Y_0 + 1 - e^{-\eta\mu}
$$

$$
E_\mu = \frac{e_0 Y_0 + e_{\text{Det}}(1 - e^{-\eta\mu})}{Q_\mu}, \quad e_0 = 0.5
$$

---

## 1. 无限长密钥模型

在无限密钥长度假设下，不考虑统计涨落修正。

单光子产率近似：

$$
Y_1 \approx Y_0 + \eta
$$

$$
Q_1 = Y_1 \cdot \mu \cdot e^{-\mu}, \quad
e_1 = \frac{e_0 Y_0 + e_{\text{Det}} \cdot \eta}{Y_1}
$$

安全码率（bit/pulse）：

$$
R = q \left[ -Q_\mu f_e H_2(E_\mu) + Q_1 (1 - H_2(e_1)) \right]
$$

码率（bit/s）：$R_{\text{bps}} = R \cdot R_{\text{rep}}$

其中 $H_2(x) = -x\log_2 x - (1-x)\log_2(1-x)$ 为二元香农熵。

---

## 2. 近似有限长模型（三态诱骗：信号 + 诱骗 + 真空）

基于文献 [15] 的 Gaussian 统计涨落修正，显式发送三种强度：信号态 $\mu$、诱骗态 $\nu$、真空态 0。

### 步骤 1：计算诱骗态观测量

使用公式 (8)(9) 代入 $\nu$ 得 $Q_\nu$、$E_\nu$。

### 步骤 2：有限长修正（公式 6, 7）

对**诱骗态**增益和误差积施加 Gaussian 置信区间修正：

$$
Q_\nu^L = Q_\nu \left(1 - \frac{\gamma_{ks}}{\sqrt{p_\nu Q_\nu N_{\text{pulse}}/2}}\right)
$$

$$
(E_\nu Q_\nu)^U = E_\nu Q_\nu \left(1 + \frac{\gamma_{ks}}{\sqrt{p_\nu E_\nu Q_\nu N_{\text{pulse}}/2}}\right)
$$

其中 $N_{\text{pulse}}/2$ 表示仅一半脉冲在基矢筛选后有效。

### 步骤 3：单光子产率下界（公式 5）

$$
Y_1^L = \frac{\mu}{\mu\nu - \nu^2}
\left( Q_\nu^L e^\nu - \frac{\nu^2}{\mu^2} Q_\mu e^\mu - \frac{\mu^2 - \nu^2}{\mu^2} Y_0 \right)
$$

### 步骤 4：单光子增益下界与误码率上界（公式 2, 3）

$$
Q_1^L = Y_1^L \cdot \mu \cdot e^{-\mu}
$$

$$
e_1^U = \frac{(E_\nu Q_\nu)^U e^\nu - e_0 Y_0}{\nu \cdot Y_1^L}
$$

### 步骤 5：安全码率（公式 1）

$$
R = p_\mu \cdot q \left[ -Q_\mu f_e H_2(E_\mu) + Q_1^L (1 - H_2(e_1^U)) \right]
$$

码率（bit/s）：$R_{\text{bps}} = R \cdot R_{\text{rep}}$

---

## 3. 严格有限长模型（1-decoy BB84，X/Z 双基矢分离）

基于 Wiesemann et al. (arXiv:2405.16578)，显式发送两种强度：信号态 $\mu_1$、诱骗态 $\mu_2$，使用 Hoeffding 或 Azuma 浓度不等式，密钥在 **X 基**生成，相位误差在 **Z 基**估计。

### 参数说明

| 符号 | 说明 |
|------|------|
| $\mu_1 > \mu_2$ | 信号强度 > 诱骗强度 |
| $P_{X,A}$, $P_{X,B}$ | Alice/Bob 选 X 基概率 |
| $\varepsilon_{\text{cor}}$, $\varepsilon_{\text{sec}}$ | 正确性/保密性安全参数 |
| $N_A$ | Alice 发送总脉冲数（块长） |
| $R_0 = R_{\text{rep}}$ | 信号发射率 = 脉冲重复率 [Hz] |

### Bayes 先验权重

$$
\tau_m = \sum_{k \in \{\mu_1, \mu_2\}} p_k \frac{e^{-k} k^m}{m!}
$$

### Hoeffding 浓度不等式偏差

$$
\delta(n, \varepsilon) = \sqrt{\frac{n \ln(1/\varepsilon)}{2}}
$$

### 检测数与误码数（X 基）

$$
n_{X,\mu_i} = N_A \cdot P_{XX} \cdot p_{\mu_i} \cdot P_{\text{det},\mu_i}
$$

其中 $P_{XX} = P_{X,A} \cdot P_{X,B}$，
$P_{\text{det},\mu} = 1 - e^{-\mu \eta_{\text{sys}}} (1 - P_{DC})$，
$P_{DC} = \text{DCR} / R_{\text{rep}}$。

Z 基同理。

### 统计边界

利用 Hoeffding 不等式对各统计量取置信上/下界（$\pm\delta$ 修正）：
- $n_{X,\mu_1}^+$，$n_{X,\mu_2}^-$：检测数上/下界
- $c_{X,\mu_i}^+$，$c_{X,\mu_i}^-$：误码数上/下界

### 真空事件边界

$$
s_{X,0}^- = \frac{\tau_0}{\mu_1 - \mu_2}
\left( \mu_1 e^{\mu_2} \frac{n_{X,\mu_2}^-}{p_{\mu_2}} - \mu_2 e^{\mu_1} \frac{n_{X,\mu_1}^+}{p_{\mu_1}} \right)
$$

### 单光子事件下界

$$
s_{X,1}^- = \frac{\mu_1 \tau_1}{\mu_2(\mu_1-\mu_2)}
\left( e^{\mu_2}\frac{n_{X,\mu_2}^-}{p_{\mu_2}}
- \frac{\mu_2^2}{\mu_1^2} e^{\mu_1} \frac{n_{X,\mu_1}^+}{p_{\mu_1}}
- \frac{\mu_1^2-\mu_2^2}{\mu_1^2 \tau_0} s_{X,0}^+ \right)
$$

### 单光子误码数上界与 QBER 上界

$$
v_{Z,1}^+ = \frac{\tau_1}{\mu_1-\mu_2}
\left( e^{\mu_1}\frac{c_{Z,\mu_1}^+}{p_{\mu_1}}
- e^{\mu_2}\frac{c_{Z,\mu_2}^-}{p_{\mu_2}} \right)
$$

$$
\lambda_{Z}^+ = \frac{v_{Z,1}^+}{s_{Z,1}^-}
$$

利用 Serfling 不等式修正 X 基相位误差：

$$
\lambda_X^+ = \lambda_Z^+ + \gamma(\varepsilon_0, \lambda_Z^+, s_{X,1}^-, s_{Z,1}^-)
$$

其中 $\gamma(a,b,c,d) = \sqrt{\dfrac{(c+d)(1-b)b}{cd\ln 2} \log_2 \dfrac{c+d}{cd(1-b)b \cdot a^2}}$。

### 纠错泄漏

$$
\text{leak}_{EC} = n_X \cdot f_e \cdot H_2(c_X / n_X)
$$

### 最终密钥长度

$$
l_{\max} = s_{X,0}^- + s_{X,1}^- [1 - H_2(\lambda_X^+)]
- \text{leak}_{EC}
- \log_2\frac{2}{\varepsilon_{\text{cor}}}
- 4\log_2\frac{15}{\varepsilon_{\text{sec}} \cdot 2^{1/4}}
$$

码率（bit/s）：$l_{\max} / t_{\text{integration}}$，其中 $t_{\text{integration}} = N_A / R_{\text{rep}}$。

---

## 参考文献

- [15] Ma et al., "Practical decoy state for quantum key distribution," *PRA* 72, 012326 (2005)
- Wiesemann et al., "Rigorous finite-size security proof for 1-decoy BB84," arXiv:2405.16578 (2024)
