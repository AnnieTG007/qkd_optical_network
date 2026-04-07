# 公式修订记录

本文档用于记录物理层噪声模型公式的修订历史。用户可在此直接输入新公式或修改现有公式。

## 修订历史

| 日期         | 修订内容      | 修订人   | 状态   |
|------------|-----------|-------|--------|
| 2026-03-25 | 初始版本     | 项目初始化 | 已实现   |
| 2026-03-26 | FWM 噪声公式 | 王雨婷   | 已更新   |

---

## 当前实现的公式

### 1. 四波混频 (FWM)

#### 1.1 相位失配因子

$$
\Delta\beta = \frac{2\pi\lambda^2}{c} |f_i - f_k| |f_j - f_k| \left[ D + \frac{\lambda^2}{2c} (|f_i - f_k| + |f_j - f_k|) S \right] \tag{1}
$$

- **来源**: 光纤光学标准推导
- **假设**: 忽略高阶色散（β₄及以上）
- **参数**:
  - $\lambda$: 四波混频产物波长 [m]
  - $f_i, f_j, f_k$: 参与 FWM 的三个信道频率 [Hz]
  - $D$: 色散系数 [s/m²]
  - $S$: 色散斜率 [s/m³]
  - $c$: 光速 [m/s]

#### 1.2 FWM 效率

$$
\eta = \frac{e^{-\Delta\alpha z} - 2e^{-\frac{\Delta\alpha}{2}z}\cos(\Delta\beta z) + 1}{\frac{(\Delta\alpha)^2}{4} + (\Delta\beta)^2} \tag{2}
$$

其中：

$$
\Delta\alpha = \alpha_i + \alpha_j + \alpha_k - \alpha_{ijk} \tag{3}
$$

- **来源**: 光纤光学标准推导
- **参数**:
  - $\alpha_i, \alpha_j, \alpha_k$: 参与 FWM 的三个信道的衰减系数 [m⁻¹]
  - $\alpha_{ijk}$: FWM 产物的衰减系数 [m⁻¹]
  - $\Delta\beta$: 相位失配因子 [rad/m]
  - $z$: 传输距离 [m]

#### 1.3 FWM 效应功率谱密度
**原公式**：

$$
P_{fwm} = \frac{\eta D^2 \gamma^2 P_i P_j P_k e^{-\alpha_{ijk} L}}{9} \tag{4}
$$

- **来源**: 光纤光学标准推导
- **参数**:
  - $D$: 简并因子（$D=3$ 当 $f_i=f_j$，$D=6$ 当 $f_i \neq f_j$）
  - $\gamma$: 非线性系数 [W⁻¹·m⁻¹]
  - $P_i, P_j, P_k$: 参与 FWM 的三个信道功率 [W]
  - $\alpha_{ijk}$: FWM 产物的衰减系数 [m⁻¹]
  - $L$: 光纤长度 [m]

**修订后**: 
$$
G_{fwm}(z)=\frac{\gamma^2e^{-\alpha_{ijk}z}}{9}\int_{-\infty}^{+\infty}D^2G_{TX}(f_i)G_{TX}(f_j)G_{TX}(f_k)\eta df_idf_j
$$
- **来源**: GN-model和FWM波动方程推导
- **参数**:
  - $D$: 简并因子（$D=3$ 当 $f_i=f_j$，$D=6$ 当 $f_i \neq f_j$）
  - $\gamma$: 非线性系数 [W⁻¹·m⁻¹]
  - $G_{TX}(f_i), G_{TX}(f_i), G_{TX}(f_i)$: 参与 FWM 的三个信道功率谱密度 [W/Hz]
  - $\alpha_{ijk}$: FWM 产物的衰减系数 [m⁻¹]
  - $z$: 所在光纤位置 [m]
**修订理由**: 统一公式的规则，给出噪声功率谱随z的函数。

**修订日期**: 2026-3-26

---

### 2. 拉曼散射

#### 2.1 Bose-Einstein 光子数分布

$$
n_{th}(\Delta f) = \frac{1}{\exp\left(\frac{h \Delta f}{k T}\right) - 1} \tag{5}
$$

- **来源**: Mandelbaum 2003, Eq.(5)
- **参数**:
  - $h$: 普朗克常数 [J·s]
  - $k$: 玻尔兹曼常数 [J/K]
  - $T$: 温度 [K]
  - $\Delta f$: 泵浦光与信号光频率差 [Hz]

#### 2.2 自发拉曼散射截面

**Stokes 过程** ($\Delta f < 0$):

$$
\sigma_{spont} = 2 h \nu_s g_R (1 + n_{th}) |\Delta f| \tag{6}
$$

**anti-Stokes 过程** ($\Delta f > 0$):

$$
\sigma_{spont} = 2 h \nu_s g_R n_{th} \frac{\nu_s}{\nu_p} |\Delta f| \tag{7}
$$

- **来源**: Mandelbaum 2003, Eq.(5)
- **参数**:
  - $\nu_s$: 散射光频率 [Hz]
  - $\nu_p$: 泵浦光频率 [Hz]
  - $g_R$: 拉曼增益系数 [m/W]

#### 2.3 前向自发拉曼功率（解析积分）

**当 $\alpha_c = \alpha_q$**:

$$
P_{fwd} = P_{pump} \sigma_{spont} e^{-\alpha L} L \tag{8}
$$

**当 $\alpha_c \neq \alpha_q$**:

$$
P_{fwd} = P_{pump} \sigma_{spont} e^{-\alpha_q L} \frac{1 - e^{-(\alpha_c - \alpha_q)L}}{\alpha_c - \alpha_q} \tag{9}
$$

- **参数**:
  - $P_{pump}$: 泵浦光功率 [W]
  - $\sigma_{spont}$: 自发拉曼散射截面 [m⁻¹]
  - $\alpha_c$: 泵浦光衰减系数 [m⁻¹]
  - $\alpha_q$: 散射光衰减系数 [m⁻¹]
  - $L$: 光纤长度 [m]

#### 2.4 后向自发拉曼功率（解析积分）

$$
P_{bwd} = P_{pump} \sigma_{spont} \frac{1 - e^{-(\alpha_c + \alpha_q)L}}{\alpha_c + \alpha_q} \tag{10}
$$

- **参数**: 同上

---

### 3. WDM 信道信号模型

#### 3.1 信道带宽定义

$$
B = \begin{cases}
0, & \text{SINGLE\_FREQ} \\
R_s, & \text{RECTANGULAR} \\
(1 + \alpha) R_s, & \text{RAISED\_COSINE}
\end{cases} \tag{25}
$$

其中：
- $R_s$: 符号率 (baudrate) [Hz]
- $\alpha$: 滚降系数 (roll_off)，$0 \leq \alpha \leq 1$

- **来源**: GN-Model 标准定义
- **实现**: `WDMChannel.bandwidth` 属性

#### 3.2 升余弦滚降谱

$$
H(f) = \begin{cases}
1, & |f| \leq \frac{(1-\alpha)R_s}{2} \\
\frac{1}{2}\left[1 + \cos\left(\frac{\pi}{\alpha R_s}\left(|f| - \frac{(1-\alpha)R_s}{2}\right)\right)\right], & \frac{(1-\alpha)R_s}{2} < |f| \leq \frac{(1+\alpha)R_s}{2} \\
0, & |f| > \frac{(1+\alpha)R_s}{2}
\end{cases} \tag{26}
$$

- **来源**: 数字通信理论，升余弦滤波器
- **实现**: `WDMChannel._get_psd_raised_cosine()`

#### 3.3 PSD 归一化条件

$$
\int_{-\infty}^{+\infty} \text{PSD}(f) \, df = P_{\text{total}} \tag{27}
$$

解析归一化系数：
$$
\text{PSD}(f) = \frac{P_{\text{total}}}{R_s} H(f) \tag{28}
$$

- **说明**: 由于升余弦谱的积分 $\int H(f) df = R_s$，因此峰值 PSD = $P_{\text{total}} / R_s$
- **实现**: `WDMChannel.get_psd()`

---

### 4. QAM 调制与信道容量

#### 4.1 香农容量（理想调制）

$$
C = B \log_2(1 + \text{SNR}) \tag{29}
$$

其中：
- $B$: 信道带宽 [Hz]
- SNR: 信噪比（线性单位）

#### 4.2 M-QAM 频谱效率

$$
\eta_{\text{SE}} = \log_2(M) \cdot (1 - \alpha) \quad \text{[bps/Hz]} \tag{30}
$$

其中：
- $M$: 调制阶数（如 16QAM → $M=16$，64QAM → $M=64$）
- $\alpha$: 滚降系数

#### 4.3 实际信道容量

$$
C = R_s \cdot \log_2(M) \cdot (1 - \alpha) \quad \text{[bps]} \tag{31}
$$

- **实现**: `WDMChannel.get_capacity()`（预留接口）

---

### 5. 色散系数

#### 5.1 色散系数（线性近似）

$$
D(\lambda) = D_0 + S_0 \times (\lambda - \lambda_0) \tag{32}
$$

其中：
- $D_0$: 参考波长 $\lambda_0$ 处的色散系数 [s/m²]
- $S_0$: 色散斜率 [s/m³]
- $\lambda_0$: 参考波长 [m]（C 波段取 1550 nm）
- $\lambda$: 当前波长 [m]

- **来源**: 光纤光学标准推导（一阶近似）
- **适用范围**: C 波段（1530~1565 nm）
- **参数来源**: ITU-T G.652 标准

#### 5.2 β₂ 与 D 的关系

$$
\beta_2 = -\frac{\lambda^2}{2\pi c} D \tag{33}
$$

- **来源**: 群速度色散与色散系数的关系

---

### 6. 有效长度

$$
L_{eff} = \frac{1 - e^{-\alpha L}}{\alpha} \tag{11}
$$

- **参数**:
  - $\alpha$: 光纤衰减系数 [m⁻¹]
  - $L$: 光纤长度 [m]

---

### 4. QKD 密钥率（诱骗态 BB84）

#### 4.1 基本参数

$$
Y_0 = \text{dark\_count} + \text{noise\_after\_spd} \tag{12}
$$

$$
\eta_{sys} = \eta_{spd} \exp(-\alpha L) 10^{-0.1 \cdot IL} \tag{13}
$$

$$
Y_1 = Y_0 + \eta_{sys} \tag{14}
$$

$$
Q_1 = Y_1 \mu e^{-\mu} \tag{15}
$$

$$
Q_{ave} = Y_0 + 1 - e^{-\eta_{sys} \mu} \tag{16}
$$

- **参数**:
  - $Y_0$: 背景计数率
  - $\eta_{spd}$: 单光子探测器效率
  - $IL$: 插入损耗 [dB]
  - $\mu$: 平均光子数

#### 4.2 误码率

$$
e_1 = \frac{e_0 Y_0 + e_{opt} \eta_{sys}}{Y_1} \tag{17}
$$

$$
e_{ave} = \frac{e_0 Y_0 + e_{opt} (1 - e^{-\eta_{sys} \mu})}{Q_{ave}} \tag{18}
$$

- **参数**:
  - $e_0$: 背景误码率
  - $e_{opt}$: 光学系统误码率

#### 4.3 密钥率（无限长）

$$
R_{key} = \eta_{BB84} \left[ -Q_{ave} f_{EC} H_2(e_{ave}) + Q_1 (1 - H_2(e_1)) \right] \tag{19}
$$

其中二元熵函数：

$$
H_2(x) = -x \log_2(x) - (1-x) \log_2(1-x) \tag{20}
$$

- **参数**:
  - $\eta_{BB84}$: BB84 协议效率因子
  - $f_{EC}$: 纠错效率

#### 4.4 噪声光子数转换

$$
n_{noise} = \frac{P_{noise}}{h \nu} \cdot t_{gate} \cdot \eta_{spd} \cdot 10^{-0.1 \cdot IL} \tag{21}
$$

- **参数**:
  - $P_{noise}$: 噪声功率 [W]
  - $\nu$: 光频率 [Hz]
  - $t_{gate}$: 门控时间 [s]

---

### 5. 协同度指标

#### 5.1 经典性能指标

$$
I_{classical} = 0.5 \cdot (1 - 10 \cdot P_{block}) + 0.5 \cdot \left(\frac{C_{total}}{C_{max}}\right) \tag{22}
$$

- **参数**:
  - $P_{block}$: 业务阻塞率（归一化，最大值 0.1）
  - $C_{total}$: 链路总香农容量 [bps]
  - $C_{max}$: 单信道最大容量 × 信道数 [bps]

#### 5.2 量子性能指标

$$
I_{quantum} = \frac{R_{key}}{R_{key}^{no\_noise}} \tag{23}
$$

- **参数**:
  - $R_{key}$: 实际密钥率 [bps]
  - $R_{key}^{no\_noise}$: 无噪声时的理论最大密钥率 [bps]

#### 5.3 协同度

$$
\text{协同度} = \sqrt{I_{classical} \cdot I_{quantum}} \tag{24}
$$

---

## 公式修订模板

### 新增公式

```markdown
#### X.X 公式名称

$$
公式内容（LaTeX 格式）
$$

- **来源**: 论文/书籍引用
- **假设**: 适用条件和简化假设
- **参数**:
  - 符号：物理意义 [单位]
```

### 修改现有公式

**原公式**: （复制原公式）

**修订后**: （新公式）

**修订理由**: （说明原因）

**修订日期**: YYYY-MM-DD

---

## 待讨论/待实现公式

| 公式名称           | 状态     | 备注            |
|----------------|--------|---------------|
| 受激拉曼（泵浦耗尽）   | 待实现    | 当前为泵浦不耗尽近似    |
| TF-QKD 密钥率     | 不计划实现  | -             |
| 芯间 FWM         | 待实现    | 多芯光纤场景        |
| 芯间拉曼          | 待实现    | 多芯光纤场景        |
