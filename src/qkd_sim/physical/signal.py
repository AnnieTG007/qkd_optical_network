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


def validate_uniform_frequency_grid(f_grid: np.ndarray) -> float:
    """验证 1D 近似均匀频率网格并返回频率步长。

    Parameters
    ----------
    f_grid : ndarray
        频率网格 [Hz]

    Returns
    -------
    float
        网格平均频率步长 df [Hz]

    Raises
    ------
    ValueError
        网格非 1D、长度不足、非严格递增或步长不均匀时
    """
    f_grid = np.asarray(f_grid, dtype=np.float64)
    if f_grid.ndim != 1 or f_grid.size < 2:
        raise ValueError(f"f_grid must be 1D with at least 2 points, got {f_grid.shape}")
    diffs = np.diff(f_grid)
    if not np.all(diffs > 0.0):
        raise ValueError("f_grid must be strictly increasing")
    df = float(np.mean(diffs))
    if not np.allclose(diffs, df, rtol=1e-6, atol=0.0):
        raise ValueError("f_grid must be approximately uniform")
    return df


def integrate_psd(f_grid: np.ndarray, psd: np.ndarray) -> float:
    """在均匀频率网格上用黎曼和积分 PSD。

    Parameters
    ----------
    f_grid : ndarray
        均匀频率网格 [Hz]
    psd : ndarray
        功率谱密度 [W/Hz]

    Returns
    -------
    float
        积分总功率 [W]
    """
    psd = np.asarray(psd, dtype=np.float64)
    if psd.size == 0:
        return 0.0
    df = validate_uniform_frequency_grid(f_grid)
    return float(np.sum(psd) * df)


def normalize_psd_to_power(
    f_grid: np.ndarray,
    psd: np.ndarray,
    target_power: float,
) -> np.ndarray:
    """缩放 PSD 使 sum(psd * df) == target_power。

    用于保证连续模型（矩形/升余弦/OSA）的总功率与离散模型一致。

    Parameters
    ----------
    f_grid : ndarray
        均匀频率网格 [Hz]
    psd : ndarray
        原始功率谱密度 [W/Hz]
    target_power : float
        目标总功率 [W]

    Returns
    -------
    ndarray
        归一化后的 PSD [W/Hz]
    """
    psd = np.asarray(psd, dtype=np.float64)
    if target_power <= 0.0:
        return np.zeros_like(psd)
    total = integrate_psd(f_grid, psd)
    if total <= 0.0:
        return np.zeros_like(psd)
    return psd * (target_power / total)


@dataclass
class WDMChannel:
    """单个 WDM 信道。

    Attributes
    ----------
    f_center : float
        信道中心频率 [Hz]
    power : float
        信道总发射功率 [W]
    channel_type : str
        "classical" | "quantum" | "inactive"
    spectrum_type : SpectrumType
        频谱类型
    B_s : float
        信号带宽 [Hz]
    beta_rolloff : float
        升余弦滚降因子 [0, 1]
    osa_f : ndarray or None
        OSA 频率数组 [Hz]（仅 OSA_SAMPLED 类型使用）
    osa_psd : ndarray or None
        OSA 功率谱密度 [W/Hz]（仅 OSA_SAMPLED 类型使用）。
        模板已对齐至本信道 f_center。
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
        """计算该信道在频率网格上的发射功率谱密度 G_TX(f)。

        对于连续模型（RECTANGULAR/RAISED_COSINE/OSA_SAMPLED），
        PSD 在 f_grid 上数值归一化，使 sum(G_TX*df) == self.power。

        Parameters
        ----------
        f_grid : ndarray, shape (N_f,)
            频率网格 [Hz]

        Returns
        -------
        ndarray, shape (N_f,)
            功率谱密度 [W/Hz]
        """
        f_grid = np.asarray(f_grid, dtype=np.float64)
        if f_grid.ndim != 1:
            raise ValueError(f"f_grid must be 1D, got shape {f_grid.shape}")

        # 非活跃信道无 PSD
        if self.channel_type != "classical" or self.power <= 0.0:
            return np.zeros_like(f_grid)

        if self.spectrum_type == SpectrumType.SINGLE_FREQ:
            return np.zeros_like(f_grid)

        if self.spectrum_type == SpectrumType.RECTANGULAR:
            raw = self._psd_rectangular(f_grid)
        elif self.spectrum_type == SpectrumType.RAISED_COSINE:
            raw = self._psd_raised_cosine(f_grid)
        elif self.spectrum_type == SpectrumType.OSA_SAMPLED:
            raw = self._psd_osa(f_grid)
        else:
            raise ValueError(f"Unknown spectrum type: {self.spectrum_type}")

        return normalize_psd_to_power(f_grid, raw, self.power)

    def _psd_rectangular(self, f_grid: np.ndarray) -> np.ndarray:
        """矩形谱 G_TX(f)。公式 3.1 (formulas_signal.md)。

        G_TX(f) = P_ch / B_s,  当 |f - f_n| ≤ B_s/2
        G_TX(f) = 0,            当 |f - f_n| > B_s/2
        """
        delta_f = np.abs(f_grid - self.f_center)
        return np.where(delta_f <= self.B_s / 2.0, self.power / self.B_s, 0.0)

    def _psd_raised_cosine(self, f_grid: np.ndarray) -> np.ndarray:
        """升余弦滚降谱 G_TX(f)。公式 3.2 (formulas_signal.md)。

        区间1: |Δf| ≤ (1-β)R_s/2  → G = P_ch/R_s
        区间2: (1-β)R_s/2 < |Δf| ≤ (1+β)R_s/2
               → G = (P_ch/R_s) × 0.5 × (1 + cos(...))
        区间3: |Δf| > (1+β)R_s/2  → G = 0
        """
        R_s = self.B_s
        beta = self.beta_rolloff

        if beta < 1e-12:
            return self._psd_rectangular(f_grid)

        delta_f = np.abs(f_grid - self.f_center)
        psd = np.zeros_like(f_grid)

        flat_edge = (1.0 - beta) * R_s / 2.0
        roll_edge = (1.0 + beta) * R_s / 2.0

        mask_flat = delta_f <= flat_edge
        psd[mask_flat] = self.power / R_s

        mask_roll = (delta_f > flat_edge) & (delta_f <= roll_edge)
        cos_arg = np.pi / (beta * R_s) * (delta_f[mask_roll] - flat_edge)
        psd[mask_roll] = (self.power / R_s) * 0.5 * (1.0 + np.cos(cos_arg))

        return psd

    def _psd_osa(self, f_grid: np.ndarray) -> np.ndarray:
        """OSA 实测谱插值。公式 3.3 (formulas_signal.md)。

        osa_f 已是移位对齐后的频率（峰值频率 == f_center）。
        """
        if self.osa_f is None or self.osa_psd is None:
            raise ValueError("OSA data (osa_f, osa_psd) not set for this channel")
        interpolator = interp1d(
            self.osa_f,
            self.osa_psd,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return np.asarray(interpolator(f_grid), dtype=np.float64)


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
    osa_csv_path: str | Path | None = None,
    osa_rbw: float | None = None,
    classical_channel_indices: list[int] | None = None,
) -> WDMGrid:
    """根据 WDMConfig 构建 WDM 信道网格。

    默认行为（classical_channel_indices=None）：
      - config.quantum_channel_indices 中的信道为量子信道
      - 其余所有信道为经典信道

    当 classical_channel_indices 指定时：
      - classical_channel_indices 中的信道为经典信道
      - config.quantum_channel_indices 中的信道为量子信道
      - 其余信道为 inactive（无功率，不参与噪声计算）

    频率网格公式 (formulas_signal.md 第5节):
        f_channels = f_center + np.arange(-(N_ch-1)/2, (N_ch+1)/2) * g

    OSA_SAMPLED 谱型：
      - 从 osa_csv_path 加载模板后，将峰值平移对齐到每个经典信道的 f_center
      - 再归一化到 P0（由 normalize_psd_to_power 保证 sum(G*df)=P0）

    Parameters
    ----------
    config : WDMConfig
        WDM 系统配置
    spectrum_type : SpectrumType
        所有信道使用的频谱类型
    f_grid : ndarray or None
        精细频率网格 [Hz]（连续模型需要）
    osa_csv_path : str, Path or None
        OSA CSV 文件路径（OSA_SAMPLED 类型必需）
    osa_rbw : float or None
        OSA 分辨率带宽 [Hz]（OSA_SAMPLED 类型必需）
    classical_channel_indices : list[int] or None
        显式指定经典信道索引（覆盖默认行为）

    Returns
    -------
    WDMGrid
    """
    all_indices_set = set(range(config.N_ch))
    quantum_set = set(config.quantum_channel_indices)

    if not quantum_set.issubset(all_indices_set):
        raise ValueError("quantum_channel_indices contains out-of-range values")

    if classical_channel_indices is None:
        classical_set = all_indices_set - quantum_set
    else:
        classical_set = set(classical_channel_indices)
        if not classical_set.issubset(all_indices_set):
            raise ValueError("classical_channel_indices contains out-of-range values")
        overlap = classical_set & quantum_set
        if overlap:
            raise ValueError(
                f"Classical and quantum channel sets overlap: {sorted(overlap)}"
            )

    indices = np.arange(-(config.N_ch - 1) / 2, (config.N_ch + 1) / 2)
    f_channels = config.f_center + indices * config.channel_spacing

    # OSA 模板预处理（峰值对齐至各信道 f_center）
    osa_offsets = None
    osa_template_psd = None
    if spectrum_type == SpectrumType.OSA_SAMPLED:
        if osa_csv_path is None or osa_rbw is None:
            raise ValueError("OSA_SAMPLED requires osa_csv_path and osa_rbw")
        osa_f_raw, osa_psd_raw = load_osa_csv(Path(osa_csv_path), osa_rbw)
        template_center = float(osa_f_raw[int(np.argmax(osa_psd_raw))])
        osa_offsets = osa_f_raw - template_center  # 相对偏移
        osa_template_psd = osa_psd_raw

    channels: list[WDMChannel] = []
    for i, f_c in enumerate(f_channels):
        if i in classical_set:
            ch_type = "classical"
            power = config.P0
            ch_stype = spectrum_type
            osa_f_ch = None
            osa_psd_ch = None
            if ch_stype == SpectrumType.OSA_SAMPLED:
                osa_f_ch = f_c + osa_offsets
                osa_psd_ch = osa_template_psd
        elif i in quantum_set:
            ch_type = "quantum"
            power = 0.0
            ch_stype = SpectrumType.SINGLE_FREQ
            osa_f_ch = None
            osa_psd_ch = None
        else:
            ch_type = "inactive"
            power = 0.0
            ch_stype = SpectrumType.SINGLE_FREQ
            osa_f_ch = None
            osa_psd_ch = None

        channels.append(
            WDMChannel(
                f_center=float(f_c),
                power=power,
                channel_type=ch_type,
                spectrum_type=ch_stype,
                B_s=config.B_s,
                beta_rolloff=config.beta_rolloff,
                osa_f=osa_f_ch,
                osa_psd=osa_psd_ch,
            )
        )

    return WDMGrid(channels=channels, f_grid=f_grid)


def build_frequency_grid(
    config: WDMConfig,
    resolution: float = 0.1e9,
    padding_factor: float = 1.5,
) -> np.ndarray:
    """构建连续模型使用的精细频率网格。

    网格范围覆盖所有信道 ± padding_factor × channel_spacing。

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
        均匀频率网格 [Hz]
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
        频率 [Hz]（升序排列）
    G_osa : ndarray
        功率谱密度 [W/Hz]
    """
    import csv

    frequencies = []
    powers_dBm = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frequencies.append(float(row["frequency_THz"]))
            powers_dBm.append(float(row["power_dBm"]))

    f_osa = np.array(frequencies, dtype=np.float64) * 1e12
    P_linear = power_dBm_to_W(np.array(powers_dBm, dtype=np.float64))
    G_osa = P_linear / rbw

    order = np.argsort(f_osa)
    return f_osa[order], G_osa[order]
