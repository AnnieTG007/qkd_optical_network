"""配置数据类：FiberConfig, WDMConfig, SimulationConfig。

YAML文件使用常用单位（dB/km, ps/nm/km 等），
dataclass 的 __post_init__ 统一转换为 SI 基本单位。
变量名对应 docs/parameters.md 的"代码变量名"列。
"""

from __future__ import annotations

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


@dataclass
class WDMConfig:
    """WDM 系统参数配置。所有字段直接使用 SI 单位。

    Attributes
    ----------
    f_center : float
        WDM 网格中心频率 [Hz]，如 193.5e12 (C波段)
    N_ch : int
        信道总数
    channel_spacing : float
        信道间隔 [Hz]，如 50e9
    B_s : float
        信号带宽 / 符号速率 [Hz]
    P0 : float
        单信道发射功率 [W]
    beta_rolloff : float
        升余弦滚降因子 [0, 1]，0 = 矩形谱
    quantum_channel_indices : list[int]
        量子信道在信道数组中的索引列表（0-based）
    """

    f_center: float
    N_ch: int
    channel_spacing: float
    B_s: float
    P0: float
    beta_rolloff: float = 0.0
    quantum_channel_indices: list[int] = field(default_factory=list)


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
