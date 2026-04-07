"""
噪声模型分派器

根据信道的 SpectrumType 自动选择噪声计算模型：
- 全部 SINGLE_FREQ  → 离散模型（fwm.py, raman.py）
- 含连续类型        → 连续模型（fwm_continuous.py, raman_continuous.py）

设计原则
--------
- 纯函数，无状态
- 不重写物理模型，只做路由和结果封装
- 调用方只需传入 channels，模型选择由 SpectrumType 决定
- freq_resolution 仅在连续模式下有效，作用域限定在 compute_noise() 调用参数

References
----------
- physics/noise/fwm.py               离散 FWM 模型
- physics/noise/raman.py             离散 Raman 模型
- physics/noise/fwm_continuous.py    连续 FWM 模型
- physics/noise/raman_continuous.py  连续 Raman 模型
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

from physics.fiber import Fiber
from physics.signal import SpectrumType, WDMChannel


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class NoiseResult:
    """
    统一噪声计算结果容器。

    两种模式下均提供每信道噪声功率；连续模式下额外提供频谱数据。

    Attributes
    ----------
    mode : {'discrete', 'continuous'}
        计算模式，由 channels[*].spectrum_type 决定。
    fwm_power : ndarray, shape (n_channels,), [W]
        每信道收到的 FWM 噪声总功率。
        - 离散模式：在中心频率处的噪声功率。
        - 连续模式：fwm_spectrum[i, :] 对 df 积分的结果，
                    即 sum(fwm_spectrum[i, :]) * df。
    raman_power : ndarray, shape (n_channels,), [W]
        每信道收到的 Raman 噪声总功率（同上）。
    freq_array : ndarray or None, shape (n_freq,), [Hz]
        连续模式下的频率采样点。离散模式为 None。
    df : float or None, [Hz]
        连续模式下频率网格间隔（均匀网格）。离散模式为 None。
    fwm_spectrum : ndarray or None, shape (n_channels, n_freq), [W/Hz]
        连续模式下每信道在各采样频率处的 FWM 噪声功率谱密度。
        离散模式为 None，或 return_spectrum=False 时为 None。
    raman_spectrum : ndarray or None, shape (n_channels, n_freq), [W/Hz]
        连续模式下每信道在各采样频率处的 Raman 噪声功率谱密度。
        离散模式为 None，或 return_spectrum=False 时为 None。

    Notes
    -----
    功率与 PSD 的关系（连续模式）：
        fwm_power[i] = sum(fwm_spectrum[i, :]) * df   [W/Hz * Hz = W]

    若需要全频带总噪声 PSD，使用 fwm_spectrum_total 属性，它对信道维度求和：
        fwm_spectrum_total[f] = sum_i fwm_spectrum[i, f]   [W/Hz]
    """

    mode: Literal['discrete', 'continuous']

    # ── 两种模式均有 ──────────────────────────────────────────────────────────
    fwm_power: np.ndarray     # shape (n_channels,) [W]
    raman_power: np.ndarray   # shape (n_channels,) [W]

    # ── 仅连续模式 ────────────────────────────────────────────────────────────
    freq_array: Optional[np.ndarray] = None   # shape (n_freq,) [Hz]
    df: Optional[float] = None                # [Hz]
    fwm_spectrum: Optional[np.ndarray] = None  # shape (n_channels, n_freq) [W/Hz]
    raman_spectrum: Optional[np.ndarray] = None  # shape (n_channels, n_freq) [W/Hz]

    # ── 辅助属性 ──────────────────────────────────────────────────────────────

    @property
    def total_power(self) -> np.ndarray:
        """
        各信道总噪声功率 = FWM + Raman。

        Returns
        -------
        ndarray, shape (n_channels,), [W]
        """
        return self.fwm_power + self.raman_power

    @property
    def fwm_spectrum_total(self) -> Optional[np.ndarray]:
        """
        全频带 FWM 噪声 PSD（对信道维度求和）。

        Returns
        -------
        ndarray, shape (n_freq,), [W/Hz]，或 None（离散模式 / return_spectrum=False）。
        """
        if self.fwm_spectrum is None:
            return None
        return np.sum(self.fwm_spectrum, axis=0)

    @property
    def raman_spectrum_total(self) -> Optional[np.ndarray]:
        """
        全频带 Raman 噪声 PSD（对信道维度求和）。

        Returns
        -------
        ndarray, shape (n_freq,), [W/Hz]，或 None（离散模式 / return_spectrum=False）。
        """
        if self.raman_spectrum is None:
            return None
        return np.sum(self.raman_spectrum, axis=0)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _determine_noise_mode(
    channels: List[WDMChannel],
) -> Literal['discrete', 'continuous']:
    """
    根据信道的 SpectrumType 确定噪声计算模式。

    规则
    ----
    - 全部 SINGLE_FREQ  → 'discrete'
    - 全部为连续类型    → 'continuous'
      （RECTANGULAR、RAISED_COSINE、OSA_SAMPLED 均视为连续类型）
    - 混合类型          → 抛出 ValueError

    Parameters
    ----------
    channels : list of WDMChannel
        WDM 信道列表，不得为空。

    Returns
    -------
    mode : {'discrete', 'continuous'}

    Raises
    ------
    ValueError
        若 channels 为空，或各信道 spectrum_type 不一致。
    """
    if not channels:
        raise ValueError("channels 列表不得为空。")

    types = {ch.spectrum_type for ch in channels}

    if len(types) > 1:
        raise ValueError(
            f"Inconsistent spectrum types: {{{', '.join(t.value for t in types)}}}. "
            "All channels must use the same SpectrumType."
        )

    spectrum_type = next(iter(types))

    if spectrum_type is SpectrumType.SINGLE_FREQ:
        return 'discrete'
    else:
        return 'continuous'


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def compute_noise(
    fiber: Fiber,
    channels: List[WDMChannel],
    enable_fwm: bool = True,
    enable_raman: bool = True,
    freq_resolution: float = 1e9,
    return_spectrum: bool = False,
) -> NoiseResult:
    """
    统一噪声计算入口。

    根据 channels[*].spectrum_type 自动选择噪声模型并返回 NoiseResult。

    选择规则
    --------
    - 全部 SINGLE_FREQ  → 离散模型（physics/noise/fwm.py, raman.py）
    - 含连续类型        → 连续模型（physics/noise/fwm_continuous.py,
                          raman_continuous.py）

    Parameters
    ----------
    fiber : Fiber
        光纤对象。
    channels : list of WDMChannel
        WDM 信道列表。所有信道必须具有相同的 SpectrumType。
    enable_fwm : bool, optional
        是否计算 FWM 噪声，默认 True。
    enable_raman : bool, optional
        是否计算 Raman 噪声，默认 True。
    freq_resolution : float, optional
        连续模式下的频率网格分辨率 [Hz]，默认 1e9（1 GHz）。
        在离散模式下此参数无效。
        此参数仅存在于本函数调用，不存储在任何对象中。
    return_spectrum : bool, optional
        仅连续模式有效。True 时在 NoiseResult 中填充
        freq_array / fwm_spectrum / raman_spectrum；
        False 时这些字段为 None（节省内存，调用更快的路径）。
        在离散模式下此参数无效，频谱字段始终为 None。

    Returns
    -------
    NoiseResult
        统一结果容器。详见 NoiseResult 文档。

    Raises
    ------
    ValueError
        若 channels 为空，或各信道 spectrum_type 不一致。

    Examples
    --------
    离散模式（SINGLE_FREQ）::

        channels = [WDMChannel(..., spectrum_type=SpectrumType.SINGLE_FREQ), ...]
        result = compute_noise(fiber, channels)
        print(result.fwm_power)   # shape (n_channels,) [W]

    连续模式（RAISED_COSINE），含频谱输出::

        channels = [WDMChannel(..., spectrum_type=SpectrumType.RAISED_COSINE), ...]
        result = compute_noise(fiber, channels, freq_resolution=1e9,
                               return_spectrum=True)
        print(result.fwm_spectrum)  # shape (n_channels, n_freq) [W/Hz]
        print(result.freq_array)    # shape (n_freq,) [Hz]
    """
    mode = _determine_noise_mode(channels)

    if mode == 'discrete':
        return _compute_noise_discrete(fiber, channels, enable_fwm, enable_raman)
    else:
        return _compute_noise_continuous(
            fiber, channels, enable_fwm, enable_raman,
            freq_resolution, return_spectrum
        )


# ---------------------------------------------------------------------------
# 离散路径
# ---------------------------------------------------------------------------

def _compute_noise_discrete(
    fiber: Fiber,
    channels: List[WDMChannel],
    enable_fwm: bool,
    enable_raman: bool,
) -> NoiseResult:
    """
    离散噪声计算路径（SINGLE_FREQ 信道）。

    直接调用 compute_fwm_noise / compute_raman_noise，
    返回各信道中心频率处的噪声功率。

    Parameters
    ----------
    fiber : Fiber
    channels : list of WDMChannel
        所有信道的 spectrum_type 必须为 SINGLE_FREQ。
    enable_fwm : bool
    enable_raman : bool

    Returns
    -------
    NoiseResult
        mode='discrete'；freq_array / df / fwm_spectrum / raman_spectrum 均为 None。
    """
    # 延迟导入，避免循环依赖
    from physics.noise.fwm import compute_fwm_noise
    from physics.noise.raman import compute_raman_noise

    n_channels = len(channels)

    fwm_power = np.zeros(n_channels, dtype=np.float64)
    raman_power = np.zeros(n_channels, dtype=np.float64)

    if enable_fwm and n_channels >= 3:
        fwm_power = compute_fwm_noise(fiber=fiber, channels=channels)

    if enable_raman and n_channels >= 2:
        raman_power = compute_raman_noise(fiber=fiber, channels=channels)

    return NoiseResult(
        mode='discrete',
        fwm_power=fwm_power,
        raman_power=raman_power,
        # 离散模式无频谱数据
        freq_array=None,
        df=None,
        fwm_spectrum=None,
        raman_spectrum=None,
    )


# ---------------------------------------------------------------------------
# 连续路径
# ---------------------------------------------------------------------------

def _compute_noise_continuous(
    fiber: Fiber,
    channels: List[WDMChannel],
    enable_fwm: bool,
    enable_raman: bool,
    freq_resolution: float,
    return_spectrum: bool,
) -> NoiseResult:
    """
    连续噪声计算路径（RECTANGULAR / RAISED_COSINE / OSA_SAMPLED 信道）。

    性能优化：
    - return_spectrum=False 时，直接调用 vectorized 函数获取总功率，
      不构建 freq_array / fwm_spectrum / raman_spectrum
    - return_spectrum=True 时，才构建完整频谱数据

    Parameters
    ----------
    fiber : Fiber
    channels : list of WDMChannel
        所有信道的 spectrum_type 必须为连续类型。
    enable_fwm : bool
    enable_raman : bool
    freq_resolution : float
        连续模式频率网格分辨率 [Hz]
    return_spectrum : bool
        是否返回频谱数据（PSD）

    Returns
    -------
    NoiseResult
        mode='continuous'
    """
    # 延迟导入，避免循环依赖
    from physics.noise.fwm_continuous import compute_fwm_noise_vectorized
    from physics.noise.raman_continuous import compute_raman_noise_vectorized

    n_channels = len(channels)
    fwm_power = np.zeros(n_channels, dtype=np.float64)
    raman_power = np.zeros(n_channels, dtype=np.float64)

    # ── 频谱数据（仅 return_spectrum=True 时构建）─────────────────────────────
    freq_array: Optional[np.ndarray] = None
    df: Optional[float] = None
    fwm_spectrum: Optional[np.ndarray] = None
    raman_spectrum: Optional[np.ndarray] = None

    # ── FWM 噪声 ──────────────────────────────────────────────────────────────
    if enable_fwm and n_channels >= 3:
        if return_spectrum:
            # 完整路径：返回频谱
            fwm_powers, freq_array, fwm_spectrum = compute_fwm_noise_vectorized(
                fiber=fiber,
                channels=channels,
                target_resolution=freq_resolution,
                compute_resolution=freq_resolution,
                return_grid=True,
                return_spectrum=True
            )
            fwm_power = fwm_powers
            df = freq_array[1] - freq_array[0] if len(freq_array) > 1 else freq_resolution
        else:
            # 快速路径：只返回总功率，不构建频谱
            fwm_power = compute_fwm_noise_vectorized(
                fiber=fiber,
                channels=channels,
                target_resolution=freq_resolution,
                compute_resolution=freq_resolution,
                return_grid=False,
                return_spectrum=False
            )

    # ── Raman 噪声 ────────────────────────────────────────────────────────────
    if enable_raman and n_channels >= 2:
        if return_spectrum:
            # 完整路径：返回频谱
            raman_powers, freq_raman, raman_spectrum = compute_raman_noise_vectorized(
                fiber=fiber,
                channels=channels,
                target_resolution=freq_resolution,
                compute_resolution=freq_resolution,
                return_grid=True,
                return_spectrum=True
            )
            raman_power = raman_powers
            # 确保 freq_array 一致（取 FWM 的网格，若 FWM 未启用则用 Raman 的）
            if freq_array is None and freq_raman is not None:
                freq_array = freq_raman
                df = freq_array[1] - freq_array[0] if len(freq_array) > 1 else freq_resolution
        else:
            # 快速路径：只返回总功率，不构建频谱
            raman_power = compute_raman_noise_vectorized(
                fiber=fiber,
                channels=channels,
                target_resolution=freq_resolution,
                compute_resolution=freq_resolution,
                return_grid=False,
                return_spectrum=False
            )

    return NoiseResult(
        mode='continuous',
        fwm_power=fwm_power,
        raman_power=raman_power,
        freq_array=freq_array,
        df=df,
        fwm_spectrum=fwm_spectrum,
        raman_spectrum=raman_spectrum,
    )
