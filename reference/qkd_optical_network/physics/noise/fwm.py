"""
FWM（四波混频）噪声模型

计算由四波混频效应产生的非线性噪声。

实现的公式（参见 FORMULAS_REVISION.md）：
- 公式 (1): 相位失配因子 Δβ（在 fiber.py 中计算）
- 公式 (2): FWM 效率 η
- 公式 (3): 衰减系数差 Δα = α_i + α_j + α_k - α_ijk
- 公式 (4): FWM 产物功率 P_fwm

References
----------
- tool.py 中的 FWMSolver 类
- MCF.py 中的 get_four_wave_mixing 等方法
- Fiber optics standard derivation
- FORMULAS_REVISION.md

Notes
-----
当前版本实现单芯光纤的芯内 FWM（intra-core FWM）。
芯间 FWM 请在 inter_core.py 中实现（预留接口）。
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.constants import c

from physics.signal import WDMChannel
from physics.fiber import Fiber


def compute_fwm_noise(
    fiber: Fiber,
    channels: List[WDMChannel],
    compute_at_length: bool = True
) -> np.ndarray:
    """
    计算 FWM 产生的噪声功率

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    compute_at_length : bool, optional
        如果 True，只计算光纤末端 (z=L) 的噪声
        如果 False，返回沿光纤的噪声分布（当前版本不支持）

    Returns
    -------
    noise_powers : np.ndarray
        各信道的 FWM 噪声功率 [W]，shape: (n_channels,)

    Notes
    -----
    算法流程：
    1. 枚举所有有效的 (fi, fj, fk) 组合，满足 f_fwm = fi + fj - fk
    2. 计算每个组合的相位失配 Δβ
    3. 计算 FWM 效率 η
    4. 聚合到目标频率

    优化策略：
    - 使用矩阵运算代替嵌套 for 循环
    - 使用 np.add.at 进行聚合
    """
    if not compute_at_length:
        raise NotImplementedError(
            "Distributed FWM noise calculation is not supported yet. "
            "Only compute_at_length=True is available."
        )

    n_channels = len(channels)

    if n_channels < 3:
        # 少于 3 个信道无法产生 FWM
        return np.zeros(n_channels, dtype=np.float64)

    # 提取频率和功率数组
    freq_array = np.array([ch.center_freq for ch in channels], dtype=np.float64)
    power_array = np.array([ch.power for ch in channels], dtype=np.float64)

    # 计算 FWM 噪声
    noise_powers = _compute_fwm_power_vectorized(
        fiber=fiber,
        freq_array=freq_array,
        power_array=power_array,
        length=fiber.length
    )

    return noise_powers


def _compute_fwm_power_spectral_density(
    fiber: Fiber,
    freq_array: np.ndarray,
    power_array: np.ndarray,
    channel_spacing: float,
    length: float
) -> np.ndarray:
    """
    使用 GN-model 积分形式计算 FWM 噪声功率谱密度

    基于修订后的公式 (4)：
    G_fwm(z) = (γ²e^(-α_ijk*z)/9) ∫∫ D² G_TX(fi) G_TX(fj) G_TK(fk) η dfi dfj
    其中 fk = fi + fj - f_fwm

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    freq_array : np.ndarray
        信道频率数组 [Hz]，shape: (n_channels,)
    power_array : np.ndarray
        信道功率数组 [W]，shape: (n_channels,)
    channel_spacing : float
        信道频率间隔 [Hz]，用于将离散功率转换为功率谱密度
    length : float
        光纤长度 [m]

    Returns
    -------
    noise_psd : np.ndarray
        FWM 噪声功率谱密度 [W/Hz]，shape: (n_channels,)

    Notes
    -----
    离散信道近似：将连续积分转化为离散求和
    G_TX(f) ≈ P / Δf_channel，其中Δf_channel 为信道带宽（此处用信道间隔近似）
    """
    n = len(freq_array)

    # 将离散信道功率转换为功率谱密度 [W/Hz]
    # G_TX = P / channel_spacing（假设矩形频谱，带宽等于信道间隔）
    psd_array = power_array / channel_spacing

    # 生成所有可能的 (i, j, k) 组合
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n),
        indexing='ij'
    )

    # 展平为二维数组 shape: (n³, 3)
    combinations = np.stack([i_idx.ravel(), j_idx.ravel(), k_idx.ravel()], axis=1)

    # 计算 FWM 产物频率
    freq_i = freq_array[combinations[:, 0]]
    freq_j = freq_array[combinations[:, 1]]
    freq_k = freq_array[combinations[:, 2]]
    freq_fwm = freq_i + freq_j - freq_k

    # 筛选：FWM 产物频率必须接近某个信道频率
    # 容差：信道间隔的 10%
    spacing_tolerance = channel_spacing * 0.1
    valid_mask = np.zeros(len(freq_fwm), dtype=bool)
    fwm_to_channel = np.full(len(freq_fwm), -1, dtype=int)

    for ch_idx, ch_freq in enumerate(freq_array):
        match = np.abs(freq_fwm - ch_freq) < spacing_tolerance
        valid_mask |= match
        fwm_to_channel[match] = ch_idx

    # 进一步筛选：排除无效组合
    # - fk 不能等于 fi 或 fj（否则没有能量转移）
    # - FWM 产物不能与输入频率相同
    valid_mask &= (combinations[:, 2] != combinations[:, 0])
    valid_mask &= (combinations[:, 2] != combinations[:, 1])

    # 获取有效组合
    valid_combinations = combinations[valid_mask]
    valid_fwm_to_channel = fwm_to_channel[valid_mask]

    if len(valid_combinations) == 0:
        return np.zeros(n, dtype=np.float64)

    # 获取有效组合对应的索引
    i_idx = valid_combinations[:, 0]
    j_idx = valid_combinations[:, 1]
    k_idx = valid_combinations[:, 2]

    # 获取有效组合对应的频率
    freq_i_valid = freq_array[i_idx]
    freq_j_valid = freq_array[j_idx]
    freq_k_valid = freq_array[k_idx]
    freq_fwm_valid = freq_i_valid + freq_j_valid - freq_k_valid

    # 获取功率谱密度
    psd_i = psd_array[i_idx]
    psd_j = psd_array[j_idx]
    psd_k = psd_array[k_idx]

    # 计算简并因子 D
    # D = 3 if fi == fj, else D = 6
    d_factor = np.where(np.abs(freq_i_valid - freq_j_valid) < 1e6, 3, 6)

    # 计算相位失配
    delta_beta = np.array([
        fiber.get_phase_mismatch(freq_i_valid[m], freq_j_valid[m], freq_k_valid[m])
        for m in range(len(valid_combinations))
    ])

    # 获取衰减系数（波长依赖性）
    # Δα = α_i + α_j + α_k - α_ijk
    alpha_i = fiber.get_loss_array(freq_i_valid)
    alpha_j = fiber.get_loss_array(freq_j_valid)
    alpha_k = fiber.get_loss_array(freq_k_valid)
    alpha_fwm = fiber.get_loss_array(freq_fwm_valid)
    delta_alpha = alpha_i + alpha_j + alpha_k - alpha_fwm

    # 计算 FWM 效率
    eta = _compute_fwm_efficiency(delta_alpha, delta_beta, length)

    # 获取 FWM 产物信道的衰减（用于 exp(-α_ijk * z) 项）
    alpha_ijk = alpha_fwm

    # 计算每个组合的 FWM 功率谱密度贡献
    # 修订后公式 (4): G_fwm = (γ²e^(-α_ijk*z)/9) × D² × G_TX(fi) × G_TX(fj) × G_TX(fk) × η
    gamma = fiber.nonlinear_coff
    fwm_psd_contrib = (
        eta
        * (d_factor * gamma) ** 2
        * psd_i * psd_j * psd_k
        * np.exp(-alpha_ijk * length)
        / 9.0
    )

    # 离散求和近似连续积分：乘以 dfi × dfj（频率分辨率的平方）
    # 这里 dfi = dfj = channel_spacing
    df = channel_spacing
    fwm_psd_contrib *= df * df

    # 聚合到目标信道
    noise_psd = np.zeros(n, dtype=np.float64)
    np.add.at(noise_psd, valid_fwm_to_channel, fwm_psd_contrib)

    return noise_psd


def _compute_fwm_power_vectorized(
    fiber: Fiber,
    freq_array: np.ndarray,
    power_array: np.ndarray,
    length: float
) -> np.ndarray:
    """
    使用矩阵运算计算 FWM 噪声功率（向量化版本）

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    freq_array : np.ndarray
        信道频率数组 [Hz]，shape: (n_channels,)
    power_array : np.ndarray
        信道功率数组 [W]，shape: (n_channels,)
    length : float
        光纤长度 [m]

    Returns
    -------
    noise_powers : np.ndarray
        FWM 噪声功率 [W]，shape: (n_channels,)
    """
    n = len(freq_array)

    # 计算信道间隔（用于功率谱密度转换）
    if n >= 2:
        channel_spacing = np.min(np.diff(np.sort(freq_array)))
    else:
        channel_spacing = 50e9  # 默认 50 GHz

    # 使用 GN-model 积分形式计算
    noise_psd = _compute_fwm_power_spectral_density(
        fiber, freq_array, power_array, channel_spacing, length
    )

    # 将功率谱密度积分得到总功率：P = G × Δf
    # 对于每个信道，噪声功率 = PSD × 信道带宽（用信道间隔近似）
    noise_powers = noise_psd * channel_spacing

    return noise_powers


def _compute_fwm_efficiency(
    delta_alpha: np.ndarray,
    delta_beta: np.ndarray,
    length: float
) -> np.ndarray:
    """
    计算 FWM 效率 η

    公式 (2)：
    $$
    \\eta = \\frac{e^{-\\Delta\\alpha L} - 2e^{-\\frac{\\Delta\\alpha}{2}L}\\cos(\\Delta\\beta L) + 1}
    {\\frac{(\\Delta\\alpha)^2}{4} + (\\Delta\\beta)^2}
    $$

    其中：
    - Δα = α_i + α_j + α_k - α_ijk [公式 (3)]
    - Δβ: 相位失配因子 [公式 (1)]

    Parameters
    ----------
    delta_alpha : np.ndarray
        衰减系数差数组 [m⁻¹]，shape: (n_combinations,)
        Δα = α_i + α_j + α_k - α_ijk
    delta_beta : np.ndarray
        相位失配数组 [rad/m]，shape: (n_combinations,)
    length : float
        光纤长度 [m]

    Returns
    -------
    eta : np.ndarray
        FWM 效率数组，shape: (n_combinations,)

    References
    ----------
    - FORMULAS_REVISION.md 公式 (2)
    """
    # 指数项
    exp_neg_full = np.exp(-delta_alpha * length)
    exp_neg_half = np.exp(-delta_alpha * length / 2)

    # 余弦项 cos(Δβ·L)
    cos_term = np.cos(delta_beta * length)

    # 分子：e^(-Δα·L) - 2·e^(-Δα·L/2)·cos(Δβ·L) + 1
    numerator = exp_neg_full - 2 * exp_neg_half * cos_term + 1

    # 分母：(Δα)²/4 + (Δβ)²
    denominator = (delta_alpha ** 2) / 4 + delta_beta ** 2

    # FWM 效率
    eta = numerator / denominator

    # 处理分母接近 0 的情况（Δα = 0 且 Δβ = 0 时）
    # 此时 η = L²（通过洛必达法则）
    singular_mask = denominator < 1e-20
    if np.any(singular_mask):
        eta[singular_mask] = length ** 2

    return eta


def _compute_fwm_efficiency_backward(
    alpha: float,
    delta_beta: np.ndarray,
    length: float
) -> np.ndarray:
    """
    计算后向 FWM 效率（预留接口）

    后向 FWM 涉及瑞利散射捕获效应，公式更复杂。
    当前版本未实现。

    Parameters
    ----------
    alpha : float
        光纤衰减系数 [m⁻¹]
    delta_beta : np.ndarray
        相位失配数组 [rad/m]
    length : float
        光纤长度 [m]

    Returns
    -------
    eta : np.ndarray
        后向 FWM 效率数组

    Raises
    ------
    NotImplementedError
        当前版本未实现
    """
    raise NotImplementedError(
        "Backward FWM efficiency calculation is not implemented yet."
    )
