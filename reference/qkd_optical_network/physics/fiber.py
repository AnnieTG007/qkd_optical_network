"""
光纤类定义

定义 Fiber 类，包含光纤的所有物理参数和属性。
支持单芯光纤（SMF/HCF），预留多芯光纤接口。
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
import numpy as np
import warnings
from scipy.constants import c

from constants.fiber_parameters import (
    FiberType,
    FiberParameters,
    get_fiber_parameters,
    GNPY_RAMAN_COEFFICIENT,
)
from physics.signal import WDMChannel


@dataclass
class Fiber:
    """
    光纤类

    封装光纤的所有物理参数，支持单芯光纤（SSMF/HCF）。
    所有参数使用国际单位制（SI）。

    Attributes
    ----------
    fiber_type : FiberType
        光纤类型（SSMF 或 HCF）
    length : float
        光纤长度 [m]
    loss : float
        光纤衰减系数 [m⁻¹]
    nonlinear_coff : float
        非线性系数 γ [W⁻¹·m⁻¹]
    cd_coff : float
        色散系数 D [s/m²]
    cd_slope : float
        色散斜率 S [s/m³]
    effective_area : float
        有效模场面积 [m²]
    refractive_index : float
        折射率 n
    temperature : float
        工作温度 [K]

    Notes
    -----
    单位统一：
    - 所有长度使用 m（不使用 km）
    - 所有频率使用 Hz（不使用 THz）
    - 所有功率使用 W（不使用 dBm）
    - 衰减系数使用 m⁻¹（不使用 dB/km）

    Examples
    --------
    >>> fiber = Fiber(fiber_type=FiberType.SSMF, length=50e3)
    >>> fiber.get_loss_at_freq(193.4e12)  # 获取指定频率的衰减
    >>> fiber.get_dispersion(193.4e12)  # 获取色散参数
    """

    # 光纤类型
    fiber_type: FiberType = FiberType.SSMF

    # 基本参数（可使用默认值）
    length: float = 50e3  # 光纤长度 [m]
    loss: Optional[float] = None  # 衰减系数 [m⁻¹]（None 则使用默认值）
    nonlinear_coff: Optional[float] = None  # 非线性系数 [W⁻¹·m⁻¹]
    cd_coff: Optional[float] = None  # 色散系数 [s/m²]
    cd_slope: Optional[float] = None  # 色散斜率 [s/m³]
    effective_area: Optional[float] = None  # 有效面积 [m²]
    refractive_index: Optional[float] = None  # 折射率

    # 温度
    temperature: float = 300.0  # 工作温度 [K]

    # 拉曼数据
    raman_data: Optional[Dict[str, Any]] = None

    # 瑞利散射参数（HCF 专用）
    rayleigh_loss: Optional[float] = None  # 瑞利散射衰减 [m⁻¹]
    recapture_factor_rayleigh: Optional[float] = None  # 后向捕获因子

    # 波长相关的衰减（可选，用于更精确的建模）
    wavelength_dependent_loss: Optional[np.ndarray] = None  # shape: (n_freqs,)
    loss_freq_array: Optional[np.ndarray] = None  # shape: (n_freqs,)

    def __post_init__(self):
        """
        数据类初始化后的处理

        如果使用默认参数，则从预设的光纤参数中加载。
        """
        # 获取默认参数
        default_params = get_fiber_parameters(self.fiber_type)

        # 使用默认值填充 None 的参数
        if self.loss is None:
            self.loss = default_params.loss
        if self.nonlinear_coff is None:
            self.nonlinear_coff = default_params.nonlinear_coff
        if self.cd_coff is None:
            self.cd_coff = default_params.cd_coff
        if self.cd_slope is None:
            self.cd_slope = default_params.cd_slope
        if self.effective_area is None:
            self.effective_area = default_params.effective_area
        if self.refractive_index is None:
            self.refractive_index = default_params.refractive_index
        if self.raman_data is None:
            self.raman_data = default_params.raman_data
        if self.rayleigh_loss is None:
            self.rayleigh_loss = default_params.rayleigh_loss
        if self.recapture_factor_rayleigh is None:
            self.recapture_factor_rayleigh = default_params.recapture_factor_rayleigh

    @property
    def attenuation_dB_km(self) -> float:
        """衰减系数 [dB/km]（只读属性，用于与规格书对比）"""
        return self.loss * 4.343 * 1e3

    @property
    def beta2(self) -> float:
        """
        群速度色散参数 β₂ [s²/m]

        与色散系数 D 的关系：
        β₂ = -λ² × D / (2πc)

        使用中心波长 1550 nm 计算。
        """
        lambda0 = 1550e-9  # 1550 nm
        return -lambda0**2 * self.cd_coff / (2 * np.pi * c)

    @property
    def gamma(self) -> float:
        """
        非线性系数 γ [W⁻¹·m⁻¹]（别名，与非线性光纤光学符号一致）
        """
        return self.nonlinear_coff

    def get_loss_at_freq(self, freq: float) -> float:
        """
        获取指定频率处的衰减系数

        如果定义了波长相关衰减，则进行插值；否则返回常数衰减。

        Parameters
        ----------
        freq : float
            频率 [Hz]

        Returns
        -------
        loss : float
            衰减系数 [m⁻¹]
        """
        if self.wavelength_dependent_loss is not None:
            # 波长相关衰减：插值
            if self.loss_freq_array is None:
                raise ValueError("loss_freq_array is required when wavelength_dependent_loss is set")
            return np.interp(freq, self.loss_freq_array, self.wavelength_dependent_loss)
        else:
            # 常数衰减
            return self.loss

    def get_loss_array(self, freq_array: np.ndarray) -> np.ndarray:
        """
        获取多个频率处的衰减系数数组

        Parameters
        ----------
        freq_array : np.ndarray
            频率数组 [Hz]

        Returns
        -------
        loss_array : np.ndarray
            衰减系数数组 [m⁻¹]，与 freq_array 同形状
        """
        if self.wavelength_dependent_loss is not None:
            if self.loss_freq_array is None:
                raise ValueError("loss_freq_array is required when wavelength_dependent_loss is set")
            return np.interp(freq_array, self.loss_freq_array, self.wavelength_dependent_loss)
        else:
            return np.full_like(freq_array, self.loss, dtype=np.float64)

    def get_dispersion_at_freq(self, freq: float) -> float:
        """
        获取指定频率处的色散系数 D

        考虑色散斜率：
        D(λ) = D₀ + S₀ × (λ - λ₀)

        Parameters
        ----------
        freq : float
            频率 [Hz]

        Returns
        -------
        D : float
            色散系数 [s/m²]
        """
        lambda0 = 1550e-9  # 参考波长 1550 nm
        lam = c / freq  # 当前波长

        # 一阶近似：D(λ) ≈ D₀ + S₀ × (λ - λ₀)
        return self.cd_coff + self.cd_slope * (lam - lambda0)

    def get_phase_mismatch(
        self,
        freq_i: float,
        freq_j: float,
        freq_k: float
    ) -> float:
        """
        计算四波混频的相位失配因子 Δβ

        完整表达式（考虑色散斜率）：
        Δβ = 2π·λ²/c · |fi-fk|·|fj-fk| · [D + λ²/(2c)·(|fi-fk|+|fj-fk|)·S]

        Parameters
        ----------
        freq_i, freq_j, freq_k : float
            参与 FWM 的三个信道频率 [Hz]
            满足 f_fwm = fi + fj - fk

        Returns
        -------
        delta_beta : float
            相位失配因子 [rad/m]

        Notes
        -----
        公式来源：光纤光学标准推导
        假设：忽略高阶色散（β₄及以上）
        """
        # 四波混频产物频率
        freq_fwm = freq_i + freq_j - freq_k
        lam_fwm = c / freq_fwm  # FWM 产物波长

        # 频率差
        delta_f_ik = np.abs(freq_i - freq_k)
        delta_f_jk = np.abs(freq_j - freq_k)

        # 色散系数（在 FWM 频率处）
        D = self.get_dispersion_at_freq(freq_fwm)

        # 相位失配公式
        delta_beta = (
            2 * np.pi * lam_fwm**2 / c
            * delta_f_ik * delta_f_jk
            * (D + lam_fwm**2 / (2 * c) * (delta_f_ik + delta_f_jk) * self.cd_slope)
        )

        return delta_beta

    def get_raman_gain_coefficient(
        self,
        pump_freq: float,
        signal_freq: float
    ) -> float:
        """
        获取拉曼增益系数 gR(|signal_freq - pump_freq|)

        使用 GNpy 拉曼系数表进行线性插值。

        Parameters
        ----------
        pump_freq : float
            泵浦光频率 [Hz]
        signal_freq : float
            信号光频率 [Hz]

        Returns
        -------
        gR : float
            拉曼增益系数 [m/W]

        Notes
        -----
        公式来源：GNpy 开源项目
        参考频率：206.184634112792 THz（1454 nm）
        参考有效面积：75.74659443542413 μm²

        频率修正：
        gR(ν) = gR_ref × (ν / ν_ref) × (A_eff_ref / A_eff)
        """
        # 频率差的绝对值
        delta_f = np.abs(signal_freq - pump_freq)

        # 从拉曼数据表中插值
        raman_table = self.raman_data
        if raman_table is None:
            raise ValueError("Raman data table is not available")

        # 线性插值
        g0 = np.interp(
            delta_f,
            raman_table['frequency_offset'],
            raman_table['g0'],
            left=0.0,
            right=0.0
        )

        # 频率修正（相对于参考频率）
        freq_correction = pump_freq / raman_table['reference_frequency']

        # 有效面积修正
        area_correction = raman_table['reference_effective_area'] / self.effective_area

        return g0 * freq_correction * area_correction

    def get_effective_length(self, freq: float) -> float:
        """
        计算有效长度 L_eff

        L_eff = (1 - exp(-α·L)) / α

        Parameters
        ----------
        freq : float
            频率 [Hz]。此参数为必选。

        Returns
        -------
        L_eff : float
            有效长度 [m]

        Notes
        -----
        物理意义：考虑光纤衰减后，非线性效应的"等效作用长度"

        Raises
        ------
        UserWarning
            如果 freq 参数为 None（实际不会发生，因已移除默认值）
        """
        if freq is None:
            warnings.warn(
                "freq parameter is None, using default 193.4 THz. "
                "This is deprecated and will raise an error in future versions. "
                "Please explicitly pass the frequency.",
                UserWarning,
                stacklevel=2
            )
            freq = 193.4e12  # 默认 C 波段中心频率

        alpha = self.get_loss_at_freq(freq)

        # 避免除零
        if alpha < 1e-15:
            return self.length

        return (1 - np.exp(-alpha * self.length)) / alpha

    def get_transmission(self, freq: float) -> float:
        """
        计算光纤透过率

        T = exp(-α·L)

        Parameters
        ----------
        freq : float
            频率 [Hz]。此参数为必选。

        Returns
        -------
        T : float
            透过率（0~1 之间）

        Raises
        ------
        UserWarning
            如果 freq 参数为 None（实际不会发生，因已移除默认值）
        """
        if freq is None:
            warnings.warn(
                "freq parameter is None, using default 193.4 THz. "
                "This is deprecated and will raise an error in future versions. "
                "Please explicitly pass the frequency.",
                UserWarning,
                stacklevel=2
            )
            freq = 193.4e12

        alpha = self.get_loss_at_freq(freq)
        return np.exp(-alpha * self.length)

    def __repr__(self):
        return (f"Fiber(type={self.fiber_type.value}, L={self.length/1e3:.1f} km, "
                f"loss={self.attenuation_dB_km:.3f} dB/km)")


# ========== 多芯光纤预留接口 ==========
# 当前版本仅实现单芯光纤，以下类用于未来扩展

@dataclass
class MultiCoreFiber:
    """
    多芯光纤类（预留接口）

    未来扩展时实现：
    - 芯间 FWM（Inter-core FWM）
    - 芯间拉曼散射（Inter-core Raman scattering）
    - 芯间串扰（Inter-core crosstalk）

    Notes
    -----
    当前版本不使用此接口，仅用于代码结构预留。
    """

    # 单芯参数（与 Fiber 相同）
    single_core_params: FiberParameters

    # 多芯特有参数
    n_cores: int = 7  # 纤芯数量
    core_pitch: float = 40e-6  # 纤芯间距 [m]
    power_coupling_matrix: Optional[np.ndarray] = None  # 功率耦合系数矩阵 [m⁻¹]

    def __post_init__(self):
        """验证输入"""
        if self.power_coupling_matrix is not None:
            if self.power_coupling_matrix.shape != (self.n_cores, self.n_cores):
                raise ValueError(
                    f"power_coupling_matrix shape {self.power_coupling_matrix.shape} "
                    f"!= expected ({self.n_cores}, {self.n_cores})"
                )

    def get_inter_core_coupling(self, core_i: int, core_j: int) -> float:
        """
        获取芯间耦合系数（预留接口）

        Parameters
        ----------
        core_i, core_j : int
            纤芯编号

        Returns
        -------
        h_ij : float
            功率耦合系数 [m⁻¹]
        """
        raise NotImplementedError("Inter-core coupling not implemented yet")
