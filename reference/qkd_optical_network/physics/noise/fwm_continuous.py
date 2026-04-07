"""
FWM 噪声计算 - 连续带宽模型

基于 GN-model 的连续频谱积分形式计算 FWM 噪声。
与离散信道模型不同，本模块考虑信道带宽内的功率分布。

公式：
G_fwm(f, z) = (γ²e^(-α_ijk*z)/9) ∫∫ D²·G_TX(fi)·G_TX(fj)·G_TX(fk)·η dfi dfj

其中 fk = fi + fj - f

使用方法:
--------
from physics.noise.fwm_continuous import compute_fwm_noise_continuous

noise = compute_fwm_noise_continuous(fiber, channels, freq_resolution=1e9)
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.constants import c

from physics.signal import WDMChannel, SpectrumType
from physics.fiber import Fiber


def compute_fwm_noise_continuous(
    fiber: Fiber,
    channels: List[WDMChannel],
    freq_resolution: float = 1e9,
    compute_at_length: bool = True,
    return_grid: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    计算基于连续 PSD 的 FWM 噪声功率

    对每个信道的带宽进行积分，而非仅使用中心频率。

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    freq_resolution : float
        频率积分分辨率 [Hz]，默认 1 GHz
    compute_at_length : bool
        如果 True，只计算光纤末端噪声
    return_grid : bool
        如果 True，同时返回频率网格 [Hz]

    Returns
    -------
    noise_powers : np.ndarray
        各信道的 FWM 噪声功率 [W]
    freq_array : np.ndarray, optional
        频率网格 [Hz]（仅在 return_grid=True 时返回）

    Notes
    -----
    计算复杂度：O(n_channels³ × (B/freq_resolution)²)
    对于大带宽和高分辨率，计算时间可能很长。
    """
    if not compute_at_length:
        raise NotImplementedError("Only compute_at_length=True is supported")

    n_channels = len(channels)
    if n_channels < 3:
        return np.zeros(n_channels, dtype=np.float64), None

    # 生成频率采样网格
    freq_array, channel_psds, channel_masks = _build_continuous_spectrum_per_channel(
        channels, freq_resolution
    )
    n_freq = len(freq_array)

    # 频率间隔
    df = freq_array[1] - freq_array[0]

    print(f"  FWM continuous: {n_freq} frequency points, df = {df/1e9:.2f} GHz")

    # 计算 FWM 噪声
    noise_powers = _compute_fwm_power_continuous(
        fiber, freq_array, channel_psds, channel_masks, df, channels
    )

    if return_grid:
        return noise_powers, freq_array
    else:
        return noise_powers


def _build_continuous_spectrum_per_channel(
    channels: List[WDMChannel],
    freq_resolution: float
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    构建每个信道的连续频谱

    Parameters
    ----------
    channels : List[WDMChannel]
        信道列表
    freq_resolution : float
        频率分辨率 [Hz]

    Returns
    -------
    freq_array : np.ndarray
        频率数组 [Hz]
    channel_psds : np.ndarray
        每个信道的 PSD 数组 [W/Hz]，shape: (n_channels, n_freq)
    channel_masks : List[np.ndarray]
        每个信道的频率掩码
    """
    # 找出频率范围
    freq_min = float('inf')
    freq_max = -float('inf')

    for ch in channels:
        if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
            freq_min = min(freq_min, ch.center_freq)
            freq_max = max(freq_max, ch.center_freq)
        else:
            freq_min = min(freq_min, ch.center_freq - ch.bandwidth/2)
            freq_max = max(freq_max, ch.center_freq + ch.bandwidth/2)

    # 添加边界余量
    margin = 2 * freq_resolution
    freq_min -= margin
    freq_max += margin

    # 生成频率数组
    n_points = int((freq_max - freq_min) / freq_resolution) + 1
    freq_array = np.linspace(freq_min, freq_max, n_points)

    # 计算每个信道的 PSD 和掩码
    n_channels = len(channels)
    channel_psds = np.zeros((n_channels, len(freq_array)), dtype=np.float64)
    channel_masks = []

    for i, ch in enumerate(channels):
        channel_psds[i] = ch.get_psd(freq_array)

        # 创建信道掩码
        if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
            mask = np.abs(freq_array - ch.center_freq) < freq_resolution
        else:
            mask = (freq_array >= ch.center_freq - ch.bandwidth/2) & \
                   (freq_array <= ch.center_freq + ch.bandwidth/2)
        channel_masks.append(mask)

    return freq_array, channel_psds, channel_masks


def _compute_fwm_power_continuous(
    fiber: Fiber,
    freq_array: np.ndarray,
    channel_psds: np.ndarray,
    channel_masks: List[np.ndarray],
    df: float,
    channels: List[WDMChannel]
) -> np.ndarray:
    """
    连续模型计算 FWM 噪声功率

    对于每个目标频率 f_fwm，计算所有满足 fk = fi + fj - f_fwm 的组合的贡献。

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    freq_array : np.ndarray
        频率数组 [Hz]
    channel_psds : np.ndarray
        每个信道的 PSD 数组 [W/Hz]，shape: (n_channels, n_freq)
    channel_masks : List[np.ndarray]
        每个信道的频率掩码
    df : float
        频率间隔 [Hz]
    channels : List[WDMChannel]
        信道列表

    Returns
    -------
    noise_powers : np.ndarray
        各信道的 FWM 噪声功率 [W]
    """
    n_channels = len(channels)
    n_freq = len(freq_array)
    gamma = fiber.nonlinear_coff
    L = fiber.length

    # 获取衰减系数数组
    alpha_array = fiber.get_loss_array(freq_array)

    # 初始化噪声数组
    noise_powers = np.zeros(n_channels, dtype=np.float64)

    # 对于每个目标信道
    for ch_idx in range(n_channels):
        mask_fwm = channel_masks[ch_idx]
        freq_fwm_arr = freq_array[mask_fwm]

        if len(freq_fwm_arr) == 0:
            continue

        # 对于每个 f_fwm 频率点
        for f_fwm in freq_fwm_arr:
            # 获取 f_fwm 处的衰减
            alpha_fwm = np.interp(f_fwm, freq_array, alpha_array)

            # 遍历所有 (i, j, k) 信道组合
            for i in range(n_channels):
                mask_i = channel_masks[i]
                freq_i_arr = freq_array[mask_i]
                psd_i_arr = channel_psds[i][mask_i]
                alpha_i_arr = alpha_array[mask_i]

                for j in range(n_channels):
                    mask_j = channel_masks[j]
                    freq_j_arr = freq_array[mask_j]
                    psd_j_arr = channel_psds[j][mask_j]
                    alpha_j_arr = alpha_array[mask_j]

                    # 计算 fk = fi + fj - f_fwm
                    # 对于每个 fi 和 fj，检查 fk 是否落在某个信道内
                    for fi_idx, fi in enumerate(freq_i_arr):
                        Gi = psd_i_arr[fi_idx]
                        ai = alpha_i_arr[fi_idx]

                        for fj_idx, fj in enumerate(freq_j_arr):
                            Gj = psd_j_arr[fj_idx]
                            aj = alpha_j_arr[fj_idx]

                            # 计算 fk
                            fk = fi + fj - f_fwm

                            # 找到 fk 对应的信道 k
                            for k in range(n_channels):
                                # 物理约束：k 不能等于 i 或 j
                                # 否则 f_fwm = fi + fj - fk = fj 或 fi，无新频率产生
                                if k == i or k == j:
                                    continue

                                mask_k = channel_masks[k]
                                fk_center = channels[k].center_freq
                                fk_bandwidth = channels[k].bandwidth

                                # 检查 fk 是否在信道 k 内
                                if channels[k].spectrum_type == SpectrumType.SINGLE_FREQ:
                                    if np.abs(fk - fk_center) > freq_resolution:
                                        continue
                                else:
                                    if not (fk_center - fk_bandwidth/2 <= fk <= fk_center + fk_bandwidth/2):
                                        continue

                                # 获取 fk 处的 PSD（插值）
                                Gk = np.interp(fk, freq_array, channel_psds[k])

                                # 获取 fk 处的衰减
                                ak = np.interp(fk, freq_array, alpha_array)

                                # 计算简并因子
                                fi_center = channels[i].center_freq
                                fj_center = channels[j].center_freq
                                D = 3 if np.abs(fi_center - fj_center) < 1e6 else 6

                                # 计算相位失配
                                fk_center = channels[k].center_freq
                                delta_beta = fiber.get_phase_mismatch(fi_center, fj_center, fk_center)

                                # 计算衰减系数差
                                delta_alpha = ai + aj + ak - alpha_fwm

                                # 计算 FWM 效率
                                eta = _compute_eta_scalar(delta_alpha, delta_beta, L)

                                # FWM 功率贡献
                                # 三重积分 dfi × dfj × df_fwm
                                # 注意：fk 由 fi + fj - f_fwm 确定，不是独立变量
                                contribution = (
                                    (gamma ** 2 / 9.0) *
                                    (D ** 2) *
                                    Gi * Gj * Gk *
                                    eta *
                                    np.exp(-alpha_fwm * L) *
                                    (df ** 3)  # 三重积分
                                )

                                # 聚合到目标信道
                                noise_powers[ch_idx] += contribution

    return noise_powers


def _compute_eta_scalar(delta_alpha: float, delta_beta: float, L: float) -> float:
    """
    计算 FWM 效率（标量版本）

    Parameters
    ----------
    delta_alpha : float
        衰减系数差 [m⁻¹]
    delta_beta : float
        相位失配 [rad/m]
    L : float
        光纤长度 [m]

    Returns
    -------
    eta : float
        FWM 效率
    """
    exp_full = np.exp(-delta_alpha * L)
    exp_half = np.exp(-delta_alpha * L / 2)
    cos_term = np.cos(delta_beta * L)
    numerator = exp_full - 2 * exp_half * cos_term + 1

    denominator = (delta_alpha ** 2) / 4 + delta_beta ** 2

    # 处理奇异点
    if denominator < 1e-20:
        return L ** 2

    return numerator / denominator


def compute_fwm_noise_discrete(
    fiber: Fiber,
    channels: List[WDMChannel]
) -> np.ndarray:
    """
    单频离散模型计算 FWM 噪声（对比基准）

    仅使用信道中心频率和总功率计算。

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        信道列表

    Returns
    -------
    noise_powers : np.ndarray
        FWM 噪声功率 [W]
    """
    # 从现有的 fwm 模块导入
    from physics.noise.fwm import compute_fwm_noise

    # 临时设置所有信道为单频模型
    original_types = [ch.spectrum_type for ch in channels]
    for ch in channels:
        ch.spectrum_type = SpectrumType.SINGLE_FREQ

    noise = compute_fwm_noise(fiber, channels)

    # 恢复原始类型
    for ch, orig_type in zip(channels, original_types):
        ch.spectrum_type = orig_type

    return noise


def compute_fwm_noise_vectorized(
    fiber: Fiber,
    channels: List[WDMChannel],
    target_resolution: float = 100e6,
    compute_resolution: float = 1e9,
    compute_at_length: bool = True,
    return_grid: bool = False,
    return_spectrum: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    向量化加速版本的 FWM 连续模型（方案 A：向量化 + 粗网格插值）

    使用粗网格计算，内部采用向量化运算代替 Python 循环，
    然后通过插值获得目标分辨率的噪声功率谱。

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
        如果 True，返回每个信道的噪声频谱 (n_channels, n_freq)

    Returns
    -------
    noise_powers : np.ndarray
        各信道的 FWM 噪声功率 [W]
    freq_array : np.ndarray, optional
        频率网格 [Hz]（仅在 return_grid=True 时返回）
    noise_spectrum : np.ndarray, optional
        噪声频谱 [W/frequency_point]，形状 (n_channels, n_freq)
            （仅在 return_spectrum=True 时返回）

    Notes
    -----
    计算复杂度：O(n_channels³ × n_grid²)，其中 n_grid = B/Δf_coarse
    比原始连续模型快约 (Δf_fine/Δf_coarse)² 倍
    """
    if not compute_at_length:
        raise NotImplementedError("Only compute_at_length=True is supported")

    n_channels = len(channels)
    if n_channels < 3:
        return np.zeros(n_channels, dtype=np.float64), None, None

    # 使用粗网格构建频谱
    print(f"  [Vectorized] Using coarse grid: {compute_resolution/1e9:.1f} GHz, "
          f"target: {target_resolution/1e6:.1f} MHz")

    freq_coarse, channel_psds, channel_masks = _build_continuous_spectrum_per_channel(
        channels, compute_resolution
    )
    n_freq = len(freq_coarse)
    df = freq_coarse[1] - freq_coarse[0]

    print(f"  [Vectorized] {n_freq} coarse frequency points")

    # 使用向量化计算 FWM 噪声和频谱
    noise_powers, noise_spectrum = _compute_fwm_power_vectorized(
        fiber, freq_coarse, channel_psds, channel_masks, df, channels
    )

    # 如果需要更细的网格，构建目标网格并插值
    freq_result = freq_coarse
    if target_resolution < compute_resolution and not return_spectrum:
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


def _compute_fwm_power_vectorized(
    fiber: Fiber,
    freq_array: np.ndarray,
    channel_psds: np.ndarray,
    channel_masks: List[np.ndarray],
    df: float,
    channels: List[WDMChannel]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化计算 FWM 噪声功率和频谱

    Returns
    -------
    noise_powers : np.ndarray
        各信道的总 FWM 噪声功率 [W]
    noise_spectrum : np.ndarray
        噪声功率谱 [W/frequency_point]，形状 (n_channels, n_freq)
    """
    n_channels = len(channels)
    n_freq = len(freq_array)
    gamma = fiber.nonlinear_coff
    L = fiber.length

    # 获取衰减系数数组
    alpha_array = fiber.get_loss_array(freq_array)

    # 初始化噪声数组和频谱
    noise_powers = np.zeros(n_channels, dtype=np.float64)
    noise_spectrum = np.zeros((n_channels, n_freq), dtype=np.float64)

    # 预计算信道中心频率（用于简并因子和相位失配）
    center_freqs = np.array([ch.center_freq for ch in channels])

    # 对于每个目标信道
    for ch_idx in range(n_channels):
        mask_fwm = channel_masks[ch_idx]
        freq_fwm_arr = freq_array[mask_fwm]
        alpha_fwm_arr = alpha_array[mask_fwm]

        if len(freq_fwm_arr) == 0:
            continue

        # 对于每个 (i, j, k) 组合
        for i in range(n_channels):
            mask_i = channel_masks[i]
            if not np.any(mask_i):
                continue
            freq_i = freq_array[mask_i]
            psd_i = channel_psds[i][mask_i]
            alpha_i = alpha_array[mask_i]

            for j in range(n_channels):
                if j == i:
                    continue  # i ≠ j

                mask_j = channel_masks[j]
                if not np.any(mask_j):
                    continue
                freq_j = freq_array[mask_j]
                psd_j = channel_psds[j][mask_j]
                alpha_j = alpha_array[mask_j]

                # 计算简并因子
                D = 3 if np.abs(center_freqs[i] - center_freqs[j]) < 1e6 else 6

                # 向量化计算所有 (fi, fj) 组合
                # 使用广播创建网格: shape (len(freq_i), len(freq_j))
                Fi = freq_i[:, np.newaxis]  # column vector
                Fj = freq_j[np.newaxis, :]  # row vector
                Gi = psd_i[:, np.newaxis]
                Gj = psd_j[np.newaxis, :]
                Ai = alpha_i[:, np.newaxis]
                Aj = alpha_j[np.newaxis, :]

                # 计算 fk = fi + fj - f_fwm 对于每个 f_fwm
                for f_idx, (f_fwm, alpha_fwm) in enumerate(zip(freq_fwm_arr, alpha_fwm_arr)):
                    Fk = Fi + Fj - f_fwm

                    # 检查 fk 落在哪个信道 k 中 (k ≠ i, k ≠ j)
                    for k in range(n_channels):
                        if k == i or k == j:
                            continue

                        mask_k = channel_masks[k]
                        if not np.any(mask_k):
                            continue

                        # 找到 fk 在信道 k 范围内的索引
                        fk_min = channels[k].center_freq - channels[k].bandwidth/2
                        fk_max = channels[k].center_freq + channels[k].bandwidth/2

                        # 创建掩码：哪些 (fi, fj) 组合产生的 fk 落在信道 k 内
                        fk_mask = (Fk >= fk_min) & (Fk <= fk_max)

                        if not np.any(fk_mask):
                            continue

                        # 计算相位失配（使用中心频率近似）
                        delta_beta = fiber.get_phase_mismatch(
                            center_freqs[i], center_freqs[j], center_freqs[k]
                        )

                        # 获取对应 alpha_k（使用信道 k 中心频率）
                        alpha_k = np.interp(channels[k].center_freq, freq_array, alpha_array)

                        # 使用 np.where 获取有效的索引位置
                        valid_i, valid_j = np.where(fk_mask)

                        # 获取对应的值（展平后索引）
                        Ai_valid = Ai[valid_i, 0]
                        Aj_valid = Aj[0, valid_j]
                        Gi_valid = Gi[valid_i, 0]
                        Gj_valid = Gj[0, valid_j]

                        # 计算衰减系数差
                        delta_alpha = Ai_valid + Aj_valid + alpha_k - alpha_fwm

                        # 计算 FWM 效率（向量化）
                        eta = _compute_eta_vectorized(delta_alpha, delta_beta, L)

                        # 获取 Gk（使用信道 k 的平均 PSD 近似）
                        Gk_avg = np.mean(channel_psds[k][mask_k])

                        # 向量化计算贡献
                        contributions = (
                            (gamma ** 2 / 9.0) *
                            (D ** 2) *
                            Gi_valid * Gj_valid * Gk_avg *
                            eta *
                            np.exp(-alpha_fwm * L) *
                            (df ** 3)
                        )

                        # 聚合：shape (n_i, n_j) -> 对所有 (i, j) 求和得到标量
                        fwm_power = np.sum(contributions)

                        # 添加到频谱（均匀分布假设）
                        s_indices = np.where(mask_fwm)[0]
                        for idx, local_idx in enumerate(s_indices):
                            noise_spectrum[ch_idx, local_idx] += fwm_power / len(freq_fwm_arr)

    return noise_powers, noise_spectrum


def _compute_eta_vectorized(delta_alpha: np.ndarray, delta_beta: float, L: float) -> np.ndarray:
    """向量化版本的 FWM 效率计算"""
    exp_full = np.exp(-delta_alpha * L)
    exp_half = np.exp(-delta_alpha * L / 2)
    cos_term = np.cos(delta_beta * L)
    numerator = exp_full - 2 * exp_half * cos_term + 1

    denominator = (delta_alpha ** 2) / 4 + delta_beta ** 2

    # 处理奇异点
    eta = np.where(denominator < 1e-20, L ** 2, numerator / denominator)

    return eta