"""
拉曼散射噪声模型

计算由拉曼散射效应产生的噪声：
- 自发拉曼散射（Spontaneous Raman Scattering）
- 受激拉曼散射（Stimulated Raman Scattering，泵浦不耗尽近似）

基于 Mandelbaum 2003 Eq.(5) 的公式实现。
使用 GNpy 拉曼增益系数表进行插值。

References
----------
- Mandelbaum 2003, "Raman amplifier model in single-mode optical fiber", Eq.(5)
- tool.py 中的 RamanSolver 类
- GNpy 拉曼系数表

Notes
-----
当前版本实现：
- 泵浦不耗尽近似（适合经典光功率 << 泵浦阈值的场景）
- 单芯光纤的芯内拉曼散射

未来扩展：
- 泵浦耗尽效应
- 芯间拉曼散射（见 inter_core.py）
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.constants import h, k, c

from physics.signal import WDMChannel
from physics.fiber import Fiber


def compute_raman_noise(
    fiber: Fiber,
    channels: List[WDMChannel],
    compute_at_length: bool = True
) -> np.ndarray:
    """
    计算拉曼散射产生的噪声功率

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
        各信道的拉曼噪声功率 [W]，shape: (n_channels,)

    Notes
    -----
    算法流程：
    1. 对每个 (pump, signal) 对，计算频率差 Δf
    2. 从拉曼系数表插值得到 gR(Δf)
    3. 计算 Bose-Einstein 光子数 n_th
    4. 计算自发拉曼截面 σ_spont
    5. 解析积分得到总噪声功率

    物理过程：
    - Stokes 散射 (Δf < 0)：泵浦光 → 低频散射光 + 声子
    - anti-Stokes 散射 (Δf > 0)：泵浦光 + 声子 → 高频散射光
    """
    if not compute_at_length:
        raise NotImplementedError(
            "Distributed Raman noise calculation is not supported yet. "
            "Only compute_at_length=True is available."
        )

    n_channels = len(channels)

    if n_channels < 2:
        # 少于 2 个信道无法产生拉曼散射
        return np.zeros(n_channels, dtype=np.float64)

    # 提取频率和功率数组
    freq_array = np.array([ch.center_freq for ch in channels], dtype=np.float64)
    power_array = np.array([ch.power for ch in channels], dtype=np.float64)

    # 信道间隔（用于拉曼截面计算）
    # 注意：这里使用固定的 50 GHz 作为标准 WDM 信道间隔
    # 而不是动态计算相邻信道的频率差，因为测试时可能只包含部分信道
    channel_spacing = 50e9  # 50 GHz，标准 WDM 信道间隔

    # 计算拉曼噪声（前向 + 后向）
    noise_powers_fwd = _compute_raman_power_forward(
        fiber=fiber,
        pump_freq=freq_array,
        pump_power=power_array,
        signal_freq=freq_array,
        temperature=fiber.temperature,
        channel_spacing=channel_spacing
    )

    noise_powers_bwd = _compute_raman_power_backward(
        fiber=fiber,
        pump_freq=freq_array,
        pump_power=power_array,
        signal_freq=freq_array,
        temperature=fiber.temperature,
        channel_spacing=channel_spacing
    )

    # 总噪声 = 前向 + 后向
    # 对角线（自噪声）设为 0
    np.fill_diagonal(noise_powers_fwd, 0)
    np.fill_diagonal(noise_powers_bwd, 0)

    # 每个信道的总噪声（对所有泵浦求和）
    total_noise = np.sum(noise_powers_fwd + noise_powers_bwd, axis=1)

    return total_noise


def _compute_spontaneous_raman_cross_section(
    pump_freq: np.ndarray,
    signal_freq: np.ndarray,
    raman_gain: np.ndarray,
    temperature: float,
    channel_spacing: float = 50e9
) -> np.ndarray:
    """
    计算自发拉曼散射截面（Mandelbaum 2003, Eq.5）

    基于 tool.py 的 RamanSolver._get_sprs_coff 实现：
    - Stokes: σ = 2·h·νs · (gR · Δf_channel) · (1 + n_th)
    - anti-Stokes: σ = 2·h·νs · (gR · Δf_channel · νs/νp) · n_th

    Parameters
    ----------
    pump_freq : np.ndarray
        泵浦光频率 [Hz]，shape: (n_pumps,)
    signal_freq : np.ndarray
        信号光频率 [Hz]，shape: (n_signals,)
    raman_gain : np.ndarray
        拉曼增益系数 gR [m/W]，shape: (n_signals, n_pumps)
    temperature : float
        温度 [K]
    channel_spacing : float
        信道频率间隔 [Hz]，用于将拉曼增益系数转换为单位长度截面

    Returns
    -------
    sigma_spont : np.ndarray
        自发拉曼散射截面 [m⁻¹]，shape: (n_signals, n_pumps)

    Notes
    -----
    tool.py 中的实现：
    - stoke_coff = df_stoke_mask * gamma_raman * channel_spacing
    - anti_stoke_coff = df_anti_stoke_mask * gamma_raman * (νs/νp) * channel_spacing
    - stoke_section = 2 * h * νs * stoke_coff * (1 + eta)
    - anti_stoke_section = 2 * h * νs * anti_stoke_coff * eta

    其中：
    - Stokes 散射 (Δf < 0)：泵浦光 → 低频散射光 + 声子
    - anti-Stokes 散射 (Δf > 0)：泵浦光 + 声子 → 高频散射光
    - n_th = 1 / (exp(h·|Δf| / k·T) - 1) 为 Bose-Einstein 光子数分布
    """
    n_signals = len(signal_freq)
    n_pumps = len(pump_freq)

    # 频率差矩阵 shape: (n_signals, n_pumps)
    # Δf = νs - νp
    delta_f = signal_freq[:, np.newaxis] - pump_freq[np.newaxis, :]

    # Bose-Einstein 光子数分布
    # n_th = 1 / (exp(h·|Δf| / k·T) - 1)
    abs_delta_f = np.abs(delta_f)
    exponent = h * abs_delta_f / (k * temperature)

    # 避免 exp 溢出
    n_th = np.zeros_like(delta_f)
    valid_mask = exponent > 1e-10
    n_th[valid_mask] = 1.0 / (np.exp(exponent[valid_mask]) - 1)

    # Stokes/anti-Stokes 掩码
    stokes_mask = delta_f < 0
    anti_stokes_mask = delta_f > 0

    # 信号频率矩阵 shape: (n_signals, n_pumps)
    signal_freq_matrix = np.tile(signal_freq[:, np.newaxis], (1, n_pumps))
    pump_freq_matrix = np.tile(pump_freq[np.newaxis, :], (n_signals, 1))

    # 频率比 νs/νp（用于 anti-Stokes 补偿）
    freq_ratio = signal_freq_matrix / pump_freq_matrix

    # 拉曼系数矩阵（已包含 channel_spacing 因子）
    # 参考 tool.py: stoke_coff = gamma_raman * channel_spacing
    raman_coeff = raman_gain * channel_spacing

    # 自发拉曼散射截面
    sigma_spont = np.zeros_like(delta_f)

    # Stokes 过程 (Δf < 0)
    # σ = 2 * h * νs * (gR * channel_spacing) * (1 + n_th)
    sigma_spont[stokes_mask] = (
        2 * h * signal_freq_matrix[stokes_mask]
        * raman_coeff[stokes_mask]
        * (1 + n_th[stokes_mask])
    )

    # anti-Stokes 过程 (Δf > 0)
    # σ = 2 * h * νs * (gR * channel_spacing * νs/νp) * n_th
    sigma_spont[anti_stokes_mask] = (
        2 * h * signal_freq_matrix[anti_stokes_mask]
        * raman_coeff[anti_stokes_mask]
        * freq_ratio[anti_stokes_mask]
        * n_th[anti_stokes_mask]
    )

    return sigma_spont


def _compute_raman_power_forward(
    fiber: Fiber,
    pump_freq: np.ndarray,
    pump_power: np.ndarray,
    signal_freq: np.ndarray,
    temperature: float,
    channel_spacing: float = 50e9
) -> np.ndarray:
    """
    计算前向自发拉曼散射噪声功率

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    pump_freq : np.ndarray
        泵浦光频率 [Hz]，shape: (n_pumps,)
    pump_power : np.ndarray
        泵浦光功率 [W]，shape: (n_pumps,)
    signal_freq : np.ndarray
        信号光频率 [Hz]，shape: (n_signals,)
    temperature : float
        温度 [K]
    channel_spacing : float
        信道频率间隔 [Hz]

    Returns
    -------
    noise_power : np.ndarray
        前向拉曼噪声功率 [W]，shape: (n_signals, n_pumps)
    """
    n_signals = len(signal_freq)
    n_pumps = len(pump_freq)

    # 获取拉曼增益系数矩阵 gR(|νs - νp|)
    raman_gain = np.zeros((n_signals, n_pumps), dtype=np.float64)
    for i, sf in enumerate(signal_freq):
        for j, pf in enumerate(pump_freq):
            raman_gain[i, j] = fiber.get_raman_gain_coefficient(pf, sf)

    # 计算自发拉曼截面
    sigma_spont = _compute_spontaneous_raman_cross_section(
        pump_freq, signal_freq, raman_gain, temperature, channel_spacing
    )

    # 获取衰减系数（考虑波长依赖性）
    alpha_pump = fiber.get_loss_array(pump_freq)  # shape: (n_pumps,)
    alpha_signal = fiber.get_loss_array(signal_freq)  # shape: (n_signals,)

    # 计算衰减矩阵 shape: (n_signals, n_pumps)
    alpha_p = alpha_pump[np.newaxis, :]  # shape: (1, n_pumps)
    alpha_s = alpha_signal[:, np.newaxis]  # shape: (n_signals, 1)

    # 前向拉曼积分公式
    # 当 αs ≠ αp:
    #   P_fwd = P_pump · σ · exp(-αs·L) · (1-exp(-(αp-αs)·L)) / (αp-αs)
    # 当 αs = αp:
    #   P_fwd = P_pump · σ · exp(-α·L) · L

    # 计算功率矩阵 shape: (n_signals, n_pumps)
    power_matrix = np.tile(pump_power[np.newaxis, :], (n_signals, 1))  # shape: (n_signals, n_pumps)

    # 衰减差
    delta_alpha = alpha_p - alpha_s  # shape: (n_signals, n_pumps)

    # 指数项（需要广播到 shape: (n_signals, n_pumps)）
    exp_s = np.tile(np.exp(-alpha_s * fiber.length), (1, n_pumps))  # shape: (n_signals, n_pumps)
    exp_diff = np.exp(-delta_alpha * fiber.length)  # shape: (n_signals, n_pumps)

    # 前向噪声功率
    noise_power = np.zeros((n_signals, n_pumps), dtype=np.float64)

    # αs ≠ αp 的情况
    diff_mask = np.abs(delta_alpha) > 1e-12
    noise_power[diff_mask] = (
        power_matrix[diff_mask]
        * sigma_spont[diff_mask]
        * exp_s[diff_mask]  # 广播后索引
        * (1 - exp_diff[diff_mask])
        / delta_alpha[diff_mask]
    )

    # αs = αp 的情况（极限）
    same_mask = ~diff_mask
    if np.any(same_mask):
        noise_power[same_mask] = (
            power_matrix[same_mask]
            * sigma_spont[same_mask]
            * exp_s[same_mask]  # 广播后索引
            * fiber.length
        )

    return noise_power


def _compute_raman_power_backward(
    fiber: Fiber,
    pump_freq: np.ndarray,
    pump_power: np.ndarray,
    signal_freq: np.ndarray,
    temperature: float,
    channel_spacing: float = 50e9
) -> np.ndarray:
    """
    计算后向自发拉曼散射噪声功率

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    pump_freq : np.ndarray
        泵浦光频率 [Hz]，shape: (n_pumps,)
    pump_power : np.ndarray
        泵浦光功率 [W]，shape: (n_pumps,)
    signal_freq : np.ndarray
        信号光频率 [Hz]，shape: (n_signals,)
    temperature : float
        温度 [K]
    channel_spacing : float
        信道频率间隔 [Hz]

    Returns
    -------
    noise_power : np.ndarray
        后向拉曼噪声功率 [W]，shape: (n_signals, n_pumps)
    """
    n_signals = len(signal_freq)
    n_pumps = len(pump_freq)

    # 获取拉曼增益系数矩阵
    raman_gain = np.zeros((n_signals, n_pumps), dtype=np.float64)
    for i, sf in enumerate(signal_freq):
        for j, pf in enumerate(pump_freq):
            raman_gain[i, j] = fiber.get_raman_gain_coefficient(pf, sf)

    # 计算自发拉曼截面
    sigma_spont = _compute_spontaneous_raman_cross_section(
        pump_freq, signal_freq, raman_gain, temperature, channel_spacing
    )

    # 获取衰减系数
    alpha_pump = fiber.get_loss_array(pump_freq)
    alpha_signal = fiber.get_loss_array(signal_freq)

    alpha_p = alpha_pump[np.newaxis, :]
    alpha_s = alpha_signal[:, np.newaxis]

    # 后向拉曼积分公式
    # 当 αs ≠ αp:
    #   P_bwd = P_pump · σ · (1-exp(-(αs+αp)·L)) / (αs+αp)
    # 当 αs = αp:
    #   P_bwd = P_pump · σ · (1-exp(-2α·L)) / (2α)

    power_matrix = np.tile(pump_power[np.newaxis, :], (n_signals, 1))
    alpha_sum = alpha_s + alpha_p

    # 后向噪声功率
    noise_power = np.zeros((n_signals, n_pumps), dtype=np.float64)

    # 一般情况
    noise_power = (
        power_matrix
        * sigma_spont
        * (1 - np.exp(-alpha_sum * fiber.length))
        / alpha_sum
    )

    return noise_power


def get_stimulated_raman_gain(
    fiber: Fiber,
    pump_freq: np.ndarray,
    pump_power: np.ndarray,
    signal_freq: np.ndarray,
    freq: Optional[float] = None
) -> np.ndarray:
    """
    计算受激拉曼增益（泵浦不耗尽近似）

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    pump_freq : np.ndarray
        泵浦光频率 [Hz]，shape: (n_pumps,)
    pump_power : np.ndarray
        泵浦光功率 [W]，shape: (n_pumps,)
    signal_freq : np.ndarray
        信号光频率 [Hz]，shape: (n_signals,)
    freq : float, optional
        计算有效长度时使用的频率 [Hz]。如果为 None，使用 193.4 THz。

    Returns
    -------
    gain : np.ndarray
        受激拉曼增益（线性倍数，非 dB），shape: (n_signals,)

    Notes
    -----
    公式（泵浦不耗尽近似）：
    G = exp(gR · P_pump · L_eff / A_eff)

    其中 L_eff = (1 - exp(-α·L)) / α
    """
    if freq is None:
        freq = 193.4e12  # 默认 C 波段中心频率

    n_signals = len(signal_freq)
    n_pumps = len(pump_freq)

    # 有效长度
    L_eff = fiber.get_effective_length(freq)

    # 增益矩阵
    gain = np.ones(n_signals, dtype=np.float64)

    for i, sf in enumerate(signal_freq):
        for j, pf in enumerate(pump_freq):
            gR = fiber.get_raman_gain_coefficient(pf, sf)

            # 只考虑 Stokes 过程（信号频率 < 泵浦频率）
            if sf < pf:
                # 小信号增益
                gain[i] *= np.exp(gR * pump_power[j] * L_eff / fiber.effective_area)

    return gain
