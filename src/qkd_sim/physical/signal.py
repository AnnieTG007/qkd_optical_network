"""WDM 信号建模：频谱类型、信道、网格与 G_TX 构建。

公式来源: docs/formulas_signal.md
支持四种频谱类型:
  - SINGLE_FREQ: 离散单频（功率集中于中心频率）
  - RECTANGULAR: 连续矩形谱（公式 3.1）
  - RAISED_COSINE: 连续升余弦滚降谱（公式 3.2）
  - OSA_SAMPLED: OSA 实测谱（公式 3.3）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from qkd_sim.config.schema import WDMConfig
from qkd_sim.utils.units import power_dBm_to_W


class SpectrumType(Enum):
    """信号频谱类型枚举。"""

    SINGLE_FREQ = "single_freq"
    RECTANGULAR = "rectangular"
    RAISED_COSINE = "raised_cosine"
    OSA_SAMPLED = "osa_sampled"


@dataclass
class WDMChannel:
    """单个 WDM 信道。

    Attributes
    ----------
    f_center : float
        信道中心频率 [Hz]
    power : float
        信道功率 [W]
    channel_type : str
        "classical" | "quantum"
    spectrum_type : SpectrumType
        频谱类型
    B_s : float
        信号带宽 [Hz]
    beta_rolloff : float
        升余弦滚降因子 [0, 1]
    osa_f : ndarray or None
        OSA 频率数组 [Hz]（仅 OSA_SAMPLED 类型使用）
    osa_psd : ndarray or None
        OSA 功率谱密度 [W/Hz]（仅 OSA_SAMPLED 类型使用）
    """

    f_center: float
    power: float
    channel_type: str
    spectrum_type: SpectrumType
    B_s: float
    beta_rolloff: float = 0.0
    osa_f: np.ndarray | None = field(default=None, repr=False)
    osa_psd: np.ndarray | None = field(default=None, repr=False)

    def get_psd(self, f_grid: np.ndarray) -> np.ndarray:
        """计算该信道在频率网格上的功率谱密度 G_TX(f)。

        Parameters
        ----------
        f_grid : ndarray, shape (N_f,)
            频率网格 [Hz]

        Returns
        -------
        ndarray, shape (N_f,)
            功率谱密度 [W/Hz]
        """
        assert f_grid.ndim == 1, f"f_grid must be 1D, got shape {f_grid.shape}"

        if self.spectrum_type == SpectrumType.SINGLE_FREQ:
            # 离散模型：在最近的频率点放置功率
            # 返回全零 PSD（离散模型不使用 PSD 表示）
            return np.zeros_like(f_grid)

        elif self.spectrum_type == SpectrumType.RECTANGULAR:
            return self._psd_rectangular(f_grid)

        elif self.spectrum_type == SpectrumType.RAISED_COSINE:
            return self._psd_raised_cosine(f_grid)

        elif self.spectrum_type == SpectrumType.OSA_SAMPLED:
            return self._psd_osa(f_grid)

        raise ValueError(f"Unknown spectrum type: {self.spectrum_type}")

    def _psd_rectangular(self, f_grid: np.ndarray) -> np.ndarray:
        """矩形谱 G_TX(f)。公式 3.1 (formulas_signal.md)。

        G_TX(f) = P_ch / B_s,  当 |f - f_n| ≤ B_s/2
        G_TX(f) = 0,            当 |f - f_n| > B_s/2
        """
        delta_f = np.abs(f_grid - self.f_center)
        psd = np.where(delta_f <= self.B_s / 2.0, self.power / self.B_s, 0.0)
        return psd

    def _psd_raised_cosine(self, f_grid: np.ndarray) -> np.ndarray:
        """升余弦滚降谱 G_TX(f)。公式 3.2 (formulas_signal.md)。

        R_s = B_s (符号速率)
        β = beta_rolloff

        区间1: |Δf| ≤ (1-β)R_s/2  → G = P_ch/R_s
        区间2: (1-β)R_s/2 < |Δf| ≤ (1+β)R_s/2
               → G = (P_ch/R_s) × 0.5 × (1 + cos(π/(βR_s) × (|Δf| - (1-β)R_s/2)))
        区间3: |Δf| > (1+β)R_s/2  → G = 0
        """
        R_s = self.B_s
        beta = self.beta_rolloff

        if beta < 1e-12:
            # β ≈ 0 退化为矩形谱
            return self._psd_rectangular(f_grid)

        delta_f = np.abs(f_grid - self.f_center)
        psd = np.zeros_like(f_grid)

        flat_edge = (1.0 - beta) * R_s / 2.0
        roll_edge = (1.0 + beta) * R_s / 2.0

        # 区间1: 平坦区
        mask_flat = delta_f <= flat_edge
        psd[mask_flat] = self.power / R_s

        # 区间2: 滚降区
        mask_roll = (delta_f > flat_edge) & (delta_f <= roll_edge)
        cos_arg = np.pi / (beta * R_s) * (delta_f[mask_roll] - flat_edge)
        psd[mask_roll] = (self.power / R_s) * 0.5 * (1.0 + np.cos(cos_arg))

        # 区间3: 已初始化为 0

        return psd

    def _psd_osa(self, f_grid: np.ndarray) -> np.ndarray:
        """OSA 实测谱插值。公式 3.3 (formulas_signal.md)。"""
        if self.osa_f is None or self.osa_psd is None:
            raise ValueError("OSA data (osa_f, osa_psd) not set for this channel")
        interpolator = interp1d(
            self.osa_f, self.osa_psd,
            kind="linear", bounds_error=False, fill_value=0.0,
        )
        return interpolator(f_grid)


@dataclass
class WDMGrid:
    """WDM 信道网格。

    Attributes
    ----------
    channels : list[WDMChannel]
        所有信道列表
    f_grid : ndarray or None
        精细频率网格 [Hz]（连续模型用）
    """

    channels: list[WDMChannel]
    f_grid: np.ndarray | None = field(default=None, repr=False)

    def get_classical_channels(self) -> list[WDMChannel]:
        """返回所有经典信道。"""
        return [ch for ch in self.channels if ch.channel_type == "classical"]

    def get_quantum_channels(self) -> list[WDMChannel]:
        """返回所有量子信道。"""
        return [ch for ch in self.channels if ch.channel_type == "quantum"]

    def get_channel_frequencies(self) -> np.ndarray:
        """返回所有信道中心频率数组。

        Returns
        -------
        ndarray, shape (N_ch,)
            频率 [Hz]
        """
        return np.array([ch.f_center for ch in self.channels])

    def get_channel_powers(self) -> np.ndarray:
        """返回所有信道功率数组。

        Returns
        -------
        ndarray, shape (N_ch,)
            功率 [W]
        """
        return np.array([ch.power for ch in self.channels])

    def get_total_psd(self) -> np.ndarray:
        """计算所有经典信道在频率网格上的总发射 PSD。

        仅对经典信道求和（量子信道功率 ≈ 0）。

        Returns
        -------
        ndarray, shape (N_f,)
            总功率谱密度 [W/Hz]

        Raises
        ------
        ValueError
            如果 f_grid 未设置
        """
        if self.f_grid is None:
            raise ValueError("f_grid not set; required for PSD calculation")
        total = np.zeros_like(self.f_grid)
        for ch in self.get_classical_channels():
            total += ch.get_psd(self.f_grid)
        return total


def build_wdm_grid(
    config: WDMConfig,
    spectrum_type: SpectrumType,
    f_grid: np.ndarray | None = None,
) -> WDMGrid:
    """根据 WDMConfig 构建 WDM 信道网格。

    频率网格公式 (formulas_signal.md 第5节):
        f_channels = f_center + np.arange(-(N_ch-1)/2, (N_ch+1)/2) * g

    Parameters
    ----------
    config : WDMConfig
        WDM 系统配置
    spectrum_type : SpectrumType
        所有信道使用的频谱类型
    f_grid : ndarray or None
        精细频率网格 [Hz]（连续模型需要）

    Returns
    -------
    WDMGrid
    """
    # 生成等间隔信道中心频率
    indices = np.arange(-(config.N_ch - 1) / 2, (config.N_ch + 1) / 2)
    f_channels = config.f_center + indices * config.channel_spacing

    quantum_set = set(config.quantum_channel_indices)

    channels = []
    for i, f_c in enumerate(f_channels):
        ch_type = "quantum" if i in quantum_set else "classical"
        # 量子信道：接收端功率 ≈ 0（不是发射机）
        power = 0.0 if ch_type == "quantum" else config.P0
        ch = WDMChannel(
            f_center=float(f_c),
            power=power,
            channel_type=ch_type,
            spectrum_type=spectrum_type,
            B_s=config.B_s,
            beta_rolloff=config.beta_rolloff,
        )
        channels.append(ch)

    return WDMGrid(channels=channels, f_grid=f_grid)


def build_frequency_grid(
    config: WDMConfig,
    resolution: float = 0.1e9,
    padding_factor: float = 1.5,
) -> np.ndarray:
    """构建连续模型使用的精细频率网格。

    网格范围覆盖所有信道 ± padding。

    Parameters
    ----------
    config : WDMConfig
        WDM 配置
    resolution : float
        频率分辨率 [Hz]，默认 0.1 GHz
    padding_factor : float
        网格两侧扩展为信道间隔的倍数

    Returns
    -------
    ndarray
        频率网格 [Hz]
    """
    half_span = (config.N_ch - 1) / 2 * config.channel_spacing
    padding = padding_factor * config.channel_spacing
    f_min = config.f_center - half_span - padding
    f_max = config.f_center + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / resolution)) + 1
    return np.linspace(f_min, f_max, n_points)


def load_osa_csv(
    csv_path: str | Path,
    rbw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从 CSV 文件加载 OSA 实测频谱数据。

    CSV 格式: wavelength_nm, frequency_THz, power_dBm
    (见 docs/formulas_signal.md 3.3节)

    Parameters
    ----------
    csv_path : str or Path
        CSV 文件路径
    rbw : float
        OSA 分辨率带宽 [Hz]

    Returns
    -------
    f_osa : ndarray
        频率 [Hz]
    G_osa : ndarray
        功率谱密度 [W/Hz]
    """
    import csv

    wavelengths = []
    frequencies = []
    powers_dBm = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wavelengths.append(float(row["wavelength_nm"]))
            frequencies.append(float(row["frequency_THz"]))
            powers_dBm.append(float(row["power_dBm"]))

    f_osa = np.array(frequencies) * 1e12  # THz → Hz
    P_linear = power_dBm_to_W(np.array(powers_dBm))  # dBm → W
    G_osa = P_linear / rbw  # W → W/Hz

    return f_osa, G_osa
