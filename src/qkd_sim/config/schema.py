"""配置数据类：FiberConfig, WDMConfig, SimulationConfig。

YAML文件使用常用单位（dB/km, ps/nm/km 等），
dataclass 的 __post_init__ 统一转换为 SI 基本单位。
变量名对应 docs/parameters.md 的"代码变量名"列。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qkd_sim.utils.units import (
    alpha_dB_km_to_per_m,
    gamma_per_W_km_to_per_W_m,
    D_ps_nm_km_to_s_m2,
    D_slope_ps_nm2_km_to_s_m3,
    L_km_to_m,
)


@dataclass
class FiberConfig:
    """光纤参数配置。

    YAML 输入使用常用单位，__post_init__ 转为 SI。

    YAML 输入字段 (常用单位):
        alpha_dB_per_km : float  衰减 [dB/km]
        gamma_per_W_km  : float  非线性系数 [1/(W·km)]
        D_ps_nm_km      : float  色散 [ps/(nm·km)]
        D_slope_ps_nm2_km : float  色散斜率 [ps/(nm²·km)]
        L_km            : float  光纤长度 [km]
        A_eff           : float  有效模场面积 [m²]
        rayleigh_coeff  : float  瑞利散射系数 S·α_R [1/m³]
        T_kelvin        : float  温度 [K]
        length_km_samples : list[float] | None  可选：Dash app 长度采样点

    SI 输出字段 (由 __post_init__ 计算):
        alpha    : float  衰减 [1/m]
        gamma    : float  非线性系数 [1/(W·m)]
        D_c      : float  色散 [s/m²]
        D_slope  : float  色散斜率 [s/m³]
        L        : float  光纤长度 [m]
    """

    # --- YAML 输入字段 (常用单位) ---
    alpha_dB_per_km: float
    gamma_per_W_km: float
    D_ps_nm_km: float
    D_slope_ps_nm2_km: float
    L_km: float
    A_eff: float              # m² (直接 SI)
    rayleigh_coeff: float     # 1/m³ (直接 SI)
    T_kelvin: float = 300.0   # K
    length_km_samples: list[float] | None = None  # 可选：Dash app 长度采样点

    # --- SI 输出字段 (由 __post_init__ 计算) ---
    alpha: float = field(init=False, repr=False)
    gamma: float = field(init=False, repr=False)
    D_c: float = field(init=False, repr=False)
    D_slope: float = field(init=False, repr=False)
    L: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """将常用单位转换为 SI 基本单位。转换公式见 parameters.md。"""
        self.alpha = alpha_dB_km_to_per_m(self.alpha_dB_per_km)
        self.gamma = gamma_per_W_km_to_per_W_m(self.gamma_per_W_km)
        self.D_c = D_ps_nm_km_to_s_m2(self.D_ps_nm_km)
        self.D_slope = D_slope_ps_nm2_km_to_s_m3(self.D_slope_ps_nm2_km)
        self.L = L_km_to_m(self.L_km)
        if self.length_km_samples is None:
            object.__setattr__(
                self,
                'length_km_samples',
                [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            )


@dataclass
class WDMConfig:
    """WDM 系统参数配置。所有字段直接使用 SI 单位。

    频率公式: f(n) = start_freq + (n - start_channel) * channel_spacing
    ITU-T G.694.1 标准 C-band: start_freq=190.1e12, start_channel=1, end_channel=61

    Attributes
    ----------
    start_freq : float
        波段起始频率 [Hz]，如 190.1e12 (C01 = 190.1 THz)
    channel_spacing : float
        信道间隔 [Hz]，如 100e9
    start_channel : float
        起始信道号（支持半信道，如 1.5）
    end_channel : float
        终止信道号
    num_channels : int | None
        信道总数（由 start_channel / end_channel 派生）。
        为兼容旧调用保留该字段；若显式传入，必须与派生值一致。
    B_s : float
        信号带宽 / 符号速率 [Hz]
    P0 : float
        单信道发射功率 [W]
    beta_rolloff : float
        升余弦滚降因子 [0, 1]，0 = 矩形谱
    quantum_channel_indices : list[int]
        量子信道在信道数组中的索引列表（0-based）
    channel_powers_W : dict[int, float] | None
        可选：zero-based 经典信道功率覆盖 [W]。未列出的经典信道使用 P0。
    """

    start_freq: float
    channel_spacing: float
    start_channel: float
    end_channel: float
    B_s: float
    P0: float
    beta_rolloff: float = 0.0
    quantum_channel_indices: list[int] = field(default_factory=list)
    channel_powers_W: dict[int, float] | None = None
    num_channels: int | None = None  # 可选：信道总数

    def __post_init__(self) -> None:
        if self.channel_spacing <= 0.0:
            raise ValueError("channel_spacing must be positive")
        if self.B_s <= 0.0:
            raise ValueError("B_s must be positive")
        if self.P0 < 0.0:
            raise ValueError("P0 must be non-negative")

        derived_channels = self.end_channel - self.start_channel + 1.0
        derived_rounded = int(round(derived_channels))
        if derived_rounded <= 0 or not math.isclose(
            derived_channels,
            float(derived_rounded),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "start_channel and end_channel must define a positive integer "
                f"number of channels; got {self.start_channel} -> {self.end_channel}"
            )

        if self.num_channels is not None and int(self.num_channels) != derived_rounded:
            raise ValueError(
                "num_channels conflicts with start_channel/end_channel: "
                f"derived {derived_rounded}, got {self.num_channels}"
            )
        object.__setattr__(self, "num_channels", derived_rounded)

        invalid_quantum = [
            idx for idx in self.quantum_channel_indices
            if idx < 0 or idx >= derived_rounded
        ]
        if invalid_quantum:
            raise ValueError(
                "quantum_channel_indices contains out-of-range values: "
                f"{invalid_quantum}"
            )

        if self.channel_powers_W is None:
            return

        normalized_channel_powers: dict[int, float] = {}
        for raw_idx, raw_power in self.channel_powers_W.items():
            idx = int(raw_idx)
            power = float(raw_power)
            if idx < 0 or idx >= derived_rounded:
                raise ValueError(
                    "channel_powers_W contains out-of-range channel index: "
                    f"{raw_idx}"
                )
            if power < 0.0:
                raise ValueError(
                    "channel_powers_W contains negative power: "
                    f"channel {raw_idx} -> {raw_power}"
                )
            normalized_channel_powers[idx] = power
        object.__setattr__(self, "channel_powers_W", normalized_channel_powers)


@dataclass
class SimulationConfig:
    """仿真总配置，聚合光纤和 WDM 参数。

    Attributes
    ----------
    fiber : FiberConfig
        光纤参数
    wdm : WDMConfig
        WDM 系统参数
    model_type : str
        信号/噪声模型类型: "discrete" | "continuous"
    spectrum_shape : str
        频谱形状: "rect" | "raised_cosine" | "osa"
    f_grid_resolution : float
        连续模型频率网格分辨率 [Hz]，默认 0.1 GHz
    """

    fiber: FiberConfig
    wdm: WDMConfig
    model_type: str = "discrete"
    spectrum_shape: str = "rect"
    f_grid_resolution: float = 0.1e9


def load_fiber_config(path: str | Path) -> FiberConfig:
    """从 YAML 文件加载光纤配置。

    Parameters
    ----------
    path : str or Path
        YAML 文件路径

    Returns
    -------
    FiberConfig
    """
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return FiberConfig(**raw)


def load_wdm_config(path: str | Path) -> WDMConfig:
    """从 YAML 文件加载 WDM 配置。

    Parameters
    ----------
    path : str or Path
        YAML 文件路径

    Returns
    -------
    WDMConfig
    """
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return WDMConfig(**raw)


def load_simulation_config(
    fiber_path: str | Path,
    wdm_path: str | Path,
    **overrides: Any,
) -> SimulationConfig:
    """从两个 YAML 文件加载仿真配置，支持参数覆盖。

    Parameters
    ----------
    fiber_path : str or Path
        光纤 YAML 路径
    wdm_path : str or Path
        WDM YAML 路径
    **overrides
        覆盖 SimulationConfig 的字段，如 model_type="continuous"

    Returns
    -------
    SimulationConfig
    """
    fiber = load_fiber_config(fiber_path)
    wdm = load_wdm_config(wdm_path)
    return SimulationConfig(fiber=fiber, wdm=wdm, **overrides)
