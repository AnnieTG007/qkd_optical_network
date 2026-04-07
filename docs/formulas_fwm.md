## 参数定义


### 2.2 离散四波混频噪声模型
#### 2.2.1 前向四波混频噪声
如果信号建模采用1.1所示的离散信号建模，那么在计算四波混频噪声功率时，也应采用离散四波混频噪声模型。对于前向四波混频噪声功率，有
$$
P_{f,1}(z)=\frac{e^{-\alpha_1z}\gamma^2}{9}\sum_{\omega_3}\sum_{\omega_4}\eta D^2P_2(0)P_3(0)P_4(0),\tag{2.2.1}
$$
要求$\omega_2\neq\omega_3$且$\omega_2\neq\omega_4$。$P_{2,(3,4)}(0)$表示泵浦光的发射功率[W]。
式(2.2.1)中求解$\omega_1$处的前向四波混频噪声功率，因此要求参与四波混频过程的泵浦光$\omega_2,\omega_3,\omega_4$均为前向传播的光信号，其中的四波混频效率为
$$
\eta=\frac{e^{-\Delta\alpha z}-2e^{-\frac{\Delta\alpha}{2}z}cos(\beta z)+1}{\frac{(\Delta\alpha)^2}{4}+(\Delta\beta)^2},\tag{2.2.2}
$$
其中相位匹配因子$\Delta \beta$的二阶展开近似为
$$  
\Delta\beta = \frac{2\pi\lambda^2}{c} |f_3 - f_2| |f_4 - f_k| \left[ D_c + \frac{\lambda^2}{2c} (|f_3 - f_2| + |f_4 - f_2|) \frac{dD_c}{d\lambda} \right], \tag{2.2.3}  
$$
其中$f_2, f_3, f_4$表示参与四波混频过程的三个信道频率[Hz]，$D_c$表示色散系数[s/m^2]，$\frac{dD_c}{d\lambda}$表示色散斜率[s/m^3]。
对于前向四波混频噪声，最后应取光纤接收端功率，即$P_{f,1}(L)$[W]，其中$L$代表光纤长度。
#### 2.2.2 后向四波混频噪声
后向四波混频噪声被视为前向四波混频噪声经瑞利散射的再分布，于是有
$$
P_{b,1}(z)=Sa_R\int_z^{L}P_{f,1}(l)e^{-\alpha(l-z)}dl\tag{2.2.4},
$$
其中$S$代表后向瑞利散射捕获因子[1/m^2]，$\alpha_R$代表瑞利散射衰减[1/m]，两者的乘积$S\alpha_R$也称为瑞利散射系数[1/m^3]。
式(2.2.4)可整理得
$$
P_{b,1}(z)=\frac{S\alpha_R\gamma^2}{9}\sum_{\omega_3}\sum_{\omega_4}D^2P_2(0)P_3(0)P_4(0)[F(L)-F(z)]\tag{2.2.5},
$$
其中有原函数$F(l)$表示为
$$
F(l)=\frac{e^{\alpha_1z}}{\frac{(\Delta\alpha)^2}{4}+(\Delta\beta)^2}[-\frac{e^{-Al}}{A}-\frac{e^{-Bl}}{B^2}-\frac{e^{-Cl}}{C}]\tag{2.2.6},
$$
其中有
$$
\begin{equation}
\left\{
\begin{array}{lr}
A=\Delta\alpha+2\alpha_1,\\
B=\Delta\alpha/2+2\alpha_1,\\
C=2\alpha_1,\tag{2.2.7}
\end{array}
\right.
\end{equation}
$$
对于后向四波混频噪声，最后应取光纤发射端功率，即$P_{b,1}(0)$[W]。
### 2.3 模拟四波混频噪声模型
#### 2.3.1 前向四波混频噪声
对于积分区域$(f_3,f_4)$而言，简并项只是区域中满足$f_3=f_4$的一条对角线，在连续积分中会被忽略。所以无需额外考虑简并项贡献，求解$\omega_1$处的FWM噪声PSD，有
$$  
G_{f,1}(z)=4\frac{\gamma^2e^{-\alpha_1z}}{9}\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}D^2\eta G_{TX}(f_i)G_{TX}(f_j)G_{TX}(f_k) df_idf_j\tag{2.3.1}.
$$
对于前向四波混频噪声，最后应取光纤接收端功率，即$G_{f,1}(L)$[W/Hz]，其中$L$代表光纤长度[m]。
同理，对于后向四波混频噪声，有
$$
G_{b,1}(z)=Sa_R\int_z^{L}G_{f,1}(l)e^{-\alpha(l-z)}dl\tag{2.3.2}.
$$
整理得到
$$  
G_{b,1}(z)=\frac{\gamma^2}{9}\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}D^2 G_{TX}(f_i)G_{TX}(f_j)G_{TX}(f_k)[F(L)-F(z)] df_idf_j\tag{2.3.3},
$$
其中$G_{TX}(f_i)$表示频率$f_i$处的发射功率谱密度[W/Hz]，原函数$F(l)$表示为
$$
F(l)=\frac{e^{\alpha_1z}}{\frac{(\Delta\alpha)^2}{4}+(\Delta\beta)^2}[-\frac{e^{-Al}}{A}-\frac{e^{-Bl}}{B^2}-\frac{e^{-Cl}}{C}]\tag{2.3.4},
$$
其中有
$$
\begin{equation}
\left\{
\begin{array}{lr}
A=\Delta\alpha+2\alpha_1,\\
B=\Delta\alpha/2+2\alpha_1,\\
C=2\alpha_1,\tag{2.3.5}
\end{array}
\right.
\end{equation}
$$
#### 2.3.2 后向四波混频噪声
对于后向四波混频噪声，最后应取光纤发射端功率谱密度，即$G_{b,1}(0)$[W/Hz]。
对于连续噪声模型，若需要求解落在频率范围为$[B_l,B_r]$某信道的噪声功率[Hz]，应有
$$
P_{noise}=\int_{B_l}^{B_r}G_{noise}(f)df\tag{2.3.6},
$$
其中$P_{nosie}$代表落在信道频率范围内的噪声功率[W]，$G_{noise}$代表频率为$f$处的噪声功率密度[W/Hz]。