"""
拉曼噪声计算 - 连续带宽模型（优化版）

基于连续 PSD 的拉曼噪声计算。
考虑信道带宽内的功率分布，而非仅使用中心频率。

公式（自发拉曼散射）：
对于信号频率 fs 和泵浦频率 fp：
  P_raman(fs) = ∫∫ P_pump(fp) × σ_spont(fp, fs) × L_eff × dfp

其中 σ_spont 包含了频率分辨率（带宽）因子。

使用方法:
--------
from physics.noise.raman_continuous import compute_raman_noise_vectorized

noise_powers, noise_spectrum = compute_raman_noise_vectorized(fiber, channels, return_spectrum=True)
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.constants import h, k, c

from physics.signal import WDMChannel, SpectrumType
from physics.fiber import Fiber


def compute_raman_noise_continuous(
    fiber: Fiber,
    channels: List[WDMChannel],
    freq_resolution: float = 1e9,
    compute_at_length: bool = True,
    return_grid: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    原始连续模型计算拉曼噪声

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    freq_resolution : float
        频率积分分辨率 [Hz]
    compute_at_length : bool
        如果 True，只计算光纤末端噪声
    return_grid : bool
        如果 True，同时返回频率网格 [Hz]

    Returns
    -------
    noise_powers : np.ndarray
        各信道的拉曼噪声功率 [W]
    freq_array : np.ndarray, optional
        频率网格 [Hz]（仅在 return_grid=True 时返回）
    """
    if not compute_at_length:
        raise NotImplementedError("Only compute_at_length=True is supported")

    n_channels = len(channels)
    if n_channels < 2:
        return np.zeros(n_channels, dtype=np.float64), None

    # 构建频率采样网格
    freq_array, psd_array, power_array, channel_masks = _build_spectrum_grid(
        channels, freq_resolution
    )
    n_freq = len(freq_array)
    df = freq_array[1] - freq_array[0]

    print(f"  Raman continuous: {n_freq} frequency points, df = {df/1e9:.2f} GHz")

    # 获取衰减系数
    alpha_array = fiber.get_loss_array(freq_array)

    # 初始化噪声数组
    noise_powers = np.zeros(n_channels, dtype=np.float64)

    # 对每个目标信道计算噪声
    for ch_idx, ch in enumerate(channels):
        mask_s = channel_masks[ch_idx]
        if not np.any(mask_s):
            continue

        # 对每个泵浦信道
        for pump_idx, pump_ch in enumerate(channels):
            if pump_idx == ch_idx:
                continue  # 跳过自身

            mask_p = channel_masks[pump_idx]
            if not np.any(mask_p):
                continue

            # 计算该泵浦对信号的贡献
            noise_contrib = _compute_raman_contrib(
                fiber, freq_array, power_array, alpha_array,
                mask_p, mask_s, df, ch_idx, pump_idx
            )
            noise_powers[ch_idx] += noise_contrib

    if return_grid:
        return noise_powers, freq_array
    else:
        return noise_powers


def _build_spectrum_grid(
    channels: List[WDMChannel],
    freq_resolution: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    构建频谱网格

    Parameters
    ----------
    channels : List[WDMChannel]
        信道列表
    freq_resolution : float
        频率分辨率 [Hz]

    Returns
    -------
    freq_array : np.ndarray
        频率数组
    psd_array : np.ndarray
        总 PSD 数组
    power_array : np.ndarray
        每个频率点的功率数组（PSD × df）
    channel_masks : List[np.ndarray]
        每个信道的频率掩码列表
    """
    # 找出频率范围
    freq_min = float('inf')
    freq_max = -float('inf')

    for ch in channels:
        if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
            freq_min = min(freq_min, ch.center_freq - 1e9)
            freq_max = max(freq_max, ch.center_freq + 1e9)
        else:
            freq_min = min(freq_min, ch.center_freq - ch.bandwidth/2)
            freq_max = max(freq_max, ch.center_freq + ch.bandwidth/2)

    # 生成频率数组
    n_points = int((freq_max - freq_min) / freq_resolution) + 1
    freq_array = np.linspace(freq_min, freq_max, n_points)

    # 计算每个信道的 PSD 和功率
    psd_array = np.zeros(len(freq_array), dtype=np.float64)
    power_array = np.zeros(len(freq_array), dtype=np.float64)
    channel_masks = []

    df = freq_array[1] - freq_array[0]

    for ch in channels:
        ch_psd = ch.get_psd(freq_array)
        psd_array += ch_psd
        # 功率 = PSD × df
        ch_power = ch_psd * df
        power_array += ch_power

        # 创建信道掩码
        if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
            mask = np.abs(freq_array - ch.center_freq) < freq_resolution
        else:
            mask = (freq_array >= ch.center_freq - ch.bandwidth/2) & \
                   (freq_array <= ch.center_freq + ch.bandwidth/2)
        channel_masks.append(mask)

    return freq_array, psd_array, power_array, channel_masks


def _compute_raman_contrib(
    fiber: Fiber,
    freq_array: np.ndarray,
    power_array: np.ndarray,
    alpha_array: np.ndarray,
    mask_p: np.ndarray,
    mask_s: np.ndarray,
    df: float,
    ch_idx: int,
    pump_idx: int
) -> float:
    """
    计算单个泵浦对单个信号的拉曼噪声贡献

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    freq_array : np.ndarray
        频率数组
    power_array : np.ndarray
        每个频率点的功率数组 [W]
    alpha_array : np.ndarray
        衰减系数数组
    mask_p : np.ndarray
        泵浦频率掩码
    mask_s : np.ndarray
        信号频率掩码
    df : float
        频率间隔
    ch_idx : int
        信号信道索引
    pump_idx : int
        泵浦信道索引

    Returns
    -------
    noise_power : float
        拉曼噪声功率 [W]
    """
    L = fiber.length
    T = fiber.temperature

    # 获取泵浦和信号频率
    freq_p = freq_array[mask_p]
    freq_s = freq_array[mask_s]
    power_p = power_array[mask_p]
    alpha_p = alpha_array[mask_p]
    alpha_s = alpha_array[mask_s]

    if len(freq_p) == 0 or len(freq_s) == 0:
        return 0.0

    # 为了控制计算量，进行子采样
    max_points = 50
    if len(freq_p) > max_points:
        indices = np.linspace(0, len(freq_p)-1, max_points, dtype=int)
        freq_p = freq_p[indices]
        power_p = power_p[indices]
        alpha_p = alpha_p[indices]

    if len(freq_s) > max_points:
        indices = np.linspace(0, len(freq_s)-1, max_points, dtype=int)
        freq_s = freq_s[indices]
        alpha_s = alpha_s[indices]

    # 对每个泵浦频率点计算贡献
    noise_total = 0.0

    for ip, fp in enumerate(freq_p):
        Pp = power_p[ip]
        ap = alpha_p[ip]

        # 对每个信号频率点
        for is_, fs in enumerate(freq_s):
            as_ = alpha_s[is_]

            # 计算频率差
            delta_f = fs - fp

            # 跳过频率差太小的情况
            if np.abs(delta_f) < 1e9:  # < 1 GHz
                continue

            # 拉曼增益系数
            gR = fiber.get_raman_gain_coefficient(fp, fs)

            # Bose-Einstein 光子数
            abs_delta_f = np.abs(delta_f)
            exponent = h * abs_delta_f / (k * T)
            if exponent > 1e-10:
                n_th = 1.0 / (np.exp(exponent) - 1)
            else:
                n_th = 0.0

            # 自发拉曼截面（谱密度，单位 m⁻¹/Hz）
            if delta_f < 0:  # Stokes
                sigma = 2 * h * fs * gR * (1 + n_th)
            else:  # anti-Stokes
                freq_ratio = fs / fp
                sigma = 2 * h * fs * gR * freq_ratio * n_th

            # 前向拉曼积分
            delta_alpha = ap - as_
            exp_s = np.exp(-as_ * L)

            if np.abs(delta_alpha) < 1e-12:
                # αs = αp
                noise_fwd = Pp * sigma * exp_s * L
            else:
                # αs ≠ αp
                exp_diff = np.exp(-delta_alpha * L)
                noise_fwd = Pp * sigma * exp_s * (1 - exp_diff) / delta_alpha

            # 后向拉曼积分
            alpha_sum = as_ + ap
            noise_bwd = Pp * sigma * (1 - np.exp(-alpha_sum * L)) / alpha_sum

            # 聚合（乘以 df 对信号频率积分）
            noise_total += (noise_fwd + noise_bwd) * df

    return noise_total


def compute_raman_noise_vectorized(
    fiber: Fiber,
    channels: List[WDMChannel],
    target_resolution: float = 100e6,
    compute_resolution: float = 1e9,
    compute_at_length: bool = True,
    return_grid: bool = False,
    return_spectrum: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    向量化加速版本的拉曼连续模型（粗网格 + 向量化）

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    target_resolution : float
        目标频率分辨率 [Hz]，默认 100 MHz
    compute_resolution : float
        计算用粗网格分辨率 [Hz]，默认 1 GHz
    compute_at_length : bool
        如果 True，只计算光纤末端噪声
    return_grid : bool
        如果 True，同时返回频率网格 [Hz]
    return_spectrum : bool
        如果 True，返回每个信道的噪声功率谱 (n_channels, n_freq)

    Returns
    -------
    noise_powers : np.ndarray
        各信道的拉曼总噪声功率 [W]
    freq_array : np.ndarray, optional
        频率网格 [Hz]（仅在 return_grid=True 时返回）
    noise_spectrum : np.ndarray, optional
        噪声功率谱 [W/frequency_point]，形状 (n_channels, n_freq)
            （仅在 return_spectrum=True 时返回）

    Notes
    -----
    计算复杂度：O(n_channels² × n_grid²)
    """
    if not compute_at_length:
        raise NotImplementedError("Only compute_at_length=True is supported")

    n_channels = len(channels)
    if n_channels < 2:
        return np.zeros(n_channels, dtype=np.float64), None, None

    print(f"  [Vectorized Raman] Using coarse grid: {compute_resolution/1e9:.1f} GHz, "
          f"target: {target_resolution/1e6:.1f} MHz")

    # 使用粗网格构建频谱
    freq_coarse, psd_coarse, power_coarse, channel_masks = _build_spectrum_grid(
        channels, compute_resolution
    )
    n_freq = len(freq_coarse)
    df = freq_coarse[1] - freq_coarse[0]

    print(f"  [Vectorized Raman] {n_freq} coarse frequency points")

    # 获取衰减系数
    alpha_coarse = fiber.get_loss_array(freq_coarse)

    # 使用向量化计算拉曼噪声和频谱
    noise_powers, noise_spectrum = _compute_raman_power_vectorized_with_spectrum(
        fiber, freq_coarse, power_coarse, alpha_coarse,
        channel_masks, df, channels
    )

    # 构建目标频率网格
    freq_result = freq_coarse
    if target_resolution < compute_resolution and return_grid:
        freq_min = freq_coarse[0]
        freq_max = freq_coarse[-1]
        n_target = int((freq_max - freq_min) / target_resolution) + 1
        freq_result = np.linspace(freq_min, freq_max, n_target)

    if return_spectrum:
        return noise_powers, freq_coarse, noise_spectrum
    elif return_grid:
        return noise_powers, freq_result
    else:
        return noise_powers


def _compute_raman_power_vectorized_with_spectrum(
    fiber: Fiber,
    freq_array: np.ndarray,
    power_array: np.ndarray,
    alpha_array: np.ndarray,
    channel_masks: List[np.ndarray],
    df: float,
    channels: List[WDMChannel]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化计算拉曼噪声功率和频谱

    Returns
    -------
    noise_powers : np.ndarray
        各信道的总拉曼噪声功率 [W]
    noise_spectrum : np.ndarray
        噪声功率谱 [W/frequency_point]，形状 (n_channels, n_freq)
    """
    L = fiber.length
    T = fiber.temperature
    n_channels = len(channels)
    n_freq = len(freq_array)

    # 初始化频谱数组：每信道一个频谱
    noise_spectrum = np.zeros((n_channels, n_freq), dtype=np.float64)

    # 对于每个目标信道
    for ch_idx, ch in enumerate(channels):
        mask_s = channel_masks[ch_idx]
        if not np.any(mask_s):
            continue

        freq_s = freq_array[mask_s]
        alpha_s = alpha_array[mask_s]

        # 对于每个泵浦信道
        for pump_idx, pump_ch in enumerate(channels):
            if pump_idx == ch_idx:
                continue

            mask_p = channel_masks[pump_idx]
            if not np.any(mask_p):
                continue

            freq_p = freq_array[mask_p]
            power_p = power_array[mask_p]
            alpha_p = alpha_array[mask_p]

            # === 向量化计算所有 (fp, fs) 组合 ===
            Fp = freq_p[:, np.newaxis]
            Fs = freq_s[np.newaxis, :]
            Pp = power_p[:, np.newaxis]
            Ap = alpha_p[:, np.newaxis]
            As = alpha_s[np.newaxis, :]

            delta_f = Fs - Fp
            valid_mask = np.abs(delta_f) >= 1e9
            if not np.any(valid_mask):
                continue

            # 平均增益系数近似
            avg_delta_f = np.mean(np.abs(delta_f[valid_mask]))
            gR_avg = fiber.get_raman_gain_coefficient(
                freq_array[0] + avg_delta_f, freq_array[0]
            )

            # Bose-Einstein 因子
            abs_delta_f = np.abs(delta_f)
            exponent = h * abs_delta_f / (k * T)
            n_th = np.where(
                exponent > 1e-10,
                1.0 / (np.exp(np.minimum(exponent, 700)) - 1),
                0.0
            )

            # 拉曼截面 σ
            sigma = np.zeros_like(delta_f)

            stokes_mask = (delta_f < 0) & valid_mask
            valid_i, valid_j = np.where(stokes_mask)
            if len(valid_i) > 0:
                sigma[stokes_mask] = 2 * h * Fs[0, valid_j] * gR_avg * (1 + n_th[stokes_mask])

            anti_stokes_mask = (delta_f > 0) & valid_mask
            valid_i, valid_j = np.where(anti_stokes_mask)
            if len(valid_i) > 0:
                freq_ratio = Fs[0, valid_j] / Fp[valid_i, 0]
                sigma[anti_stokes_mask] = 2 * h * Fs[0, valid_j] * gR_avg * freq_ratio * n_th[anti_stokes_mask]

            # 有效长度积分
            delta_alpha = Ap - As
            exp_s = np.exp(-As * L)
            exp_diff = np.exp(-delta_alpha * L)

            noise_fwd = np.where(
                np.abs(delta_alpha) < 1e-12,
                Pp * sigma * exp_s * L,
                Pp * sigma * exp_s * (1 - exp_diff) / delta_alpha
            )

            alpha_sum = As + Ap
            noise_bwd = Pp * sigma * (1 - np.exp(-alpha_sum * L)) / alpha_sum

            # 聚合：shape (n_p,) -> 需要对 n_p 求和得到 (n_s,)
            noise_contribution = np.sum((noise_fwd + noise_bwd) * df, axis=0)

            # 添加到频谱
            s_indices = np.where(mask_s)[0]
            for idx, local_idx in enumerate(s_indices):
                noise_spectrum[ch_idx, local_idx] += noise_contribution[idx]

    # 对频谱求和得到总功率
    noise_powers = np.array([np.sum(noise_spectrum[i]) for i in range(n_channels)])

    return noise_powers, noise_spectrum


def compute_raman_noise_discrete(
    fiber: Fiber,
    channels: List[WDMChannel]
) -> np.ndarray:
    """
    单频离散模型计算拉曼噪声（对比基准）
    """
    from physics.noise.raman import compute_raman_noise

    original_types = [ch.spectrum_type for ch in channels]
    for ch in channels:
        ch.spectrum_type = SpectrumType.SINGLE_FREQ

    noise = compute_raman_noise(fiber, channels)

    for ch, orig_type in zip(channels, original_types):
        ch.spectrum_type = orig_type

    return noise
