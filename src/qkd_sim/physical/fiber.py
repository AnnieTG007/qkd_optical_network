"""光纤物理参数模型。

提供波长/频率相关的衰减、色散、拉曼增益系数等。
公开接口统一使用频率 (Hz)，内部按需通过 λ = c/f 转换。
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.constants import c as c_light

from qkd_sim.config.schema import FiberConfig


def _array_cache_key(arr: np.ndarray) -> tuple:
    """Convert a NumPy array to a hashable cache key (preserves dtype + values)."""
    return (arr.flags["C_CONTIGUOUS"], arr.dtype.str, tuple(arr.ravel()))


class Fiber:
    """光纤参数容器，支持频率相关的物理参数查询。

    当前实现：C波段常数参数 + 色散斜率修正。

    C+L 波段扩展路线
    ----------------
    - get_loss_at_freq: 替换为多项式拟合或插值表（参考 ITU-T G.652 衰减曲线）
    - get_dispersion_at_freq: 已支持斜率修正，C+L 可切换为 Sellmeier 方程
    - 新增 get_gamma_at_freq: γ = 2πn₂/(λ·A_eff(λ))，A_eff 波长依赖
    - get_effective_length: 使用频率相关 α
    - f_ref (当前硬编码 193.5 THz): 提取为 FiberConfig 字段

    Parameters
    ----------
    config : FiberConfig
        光纤配置（SI 单位，由 __post_init__ 已转换）
    """

    def __init__(self, config: FiberConfig) -> None:
        self.config = config
        # 缓存常用参数
        self._alpha = config.alpha            # 1/m
        self._gamma = config.gamma            # 1/(W·m)
        self._D_c = config.D_c                # s/m²
        self._D_slope = config.D_slope        # s/m³
        self._L = config.L                    # m
        self._A_eff = config.A_eff            # m²
        self._rayleigh_coeff = config.rayleigh_coeff  # 1/m³
        self._T = config.T_kelvin             # K
        # Phase-mismatch memoization cache: key = (key(f2), key(f3), key(f4)), value = Δβ
        self._pm_cache: dict = {}

    # ---- 属性访问 ----

    @property
    def alpha(self) -> float:
        """光纤衰减系数 [1/m]。"""
        return self._alpha

    @property
    def gamma(self) -> float:
        """非线性系数 [1/(W·m)]。"""
        return self._gamma

    @property
    def L(self) -> float:
        """光纤长度 [m]。"""
        return self._L

    @property
    def A_eff(self) -> float:
        """有效模场面积 [m²]。"""
        return self._A_eff

    @property
    def rayleigh_coeff(self) -> float:
        """瑞利散射系数 S·α_R [1/m³]。"""
        return self._rayleigh_coeff

    @property
    def T_kelvin(self) -> float:
        """工作温度 [K]。"""
        return self._T

    # ---- 频率相关参数 ----

    def get_loss_at_freq(self, freq: float | np.ndarray) -> float | np.ndarray:
        """获取给定频率处的衰减系数。

        当前实现：返回常数 alpha（C波段近似）。
        C+L 扩展时可替换为波长相关模型。

        Parameters
        ----------
        freq : float or ndarray
            频率 [Hz]

        Returns
        -------
        float or ndarray
            衰减系数 [1/m]
        """
        if isinstance(freq, np.ndarray):
            return np.full_like(freq, self._alpha)
        return self._alpha

    def get_dispersion_at_freq(self, freq: float | np.ndarray) -> float | np.ndarray:
        """获取给定频率处的色散系数 D_c。

        使用色散斜率线性修正：D_c(λ) = D_c(λ₀) + D_slope × (λ - λ₀)
        其中 λ₀ 对应参考频率 (默认 193.5 THz)。

        Parameters
        ----------
        freq : float or ndarray
            频率 [Hz]

        Returns
        -------
        float or ndarray
            色散系数 [s/m²]
        """
        # 参考波长: C波段中心 193.5 THz
        f_ref = 193.5e12  # C波段中心，C+L扩展时应提取为FiberConfig字段
        lambda_ref = c_light / f_ref
        wavelength = c_light / np.asarray(freq)
        return self._D_c + self._D_slope * (wavelength - lambda_ref)

    def get_phase_mismatch(
        self,
        f2: float | np.ndarray,
        f3: float | np.ndarray,
        f4: float | np.ndarray,
    ) -> float | np.ndarray:
        """计算 FWM 相位失配 Δβ（二阶 Taylor 展开近似）。

        .. note:: 近似说明
           采用色散系数 β(ω) 在 f₂ 对应波长处的二阶 Taylor 展开，
           仅保留 D_c 和 dD_c/dλ 两项。当信道间距远小于载波频率时
           （典型 WDM 场景，Δf/f < 1%）精度良好。精确计算需要
           β(ω) 的完整色散关系（如 Sellmeier 方程）。

        公式 2.2.3 (formulas_fwm.md):
        Δβ = (2π λ² / c) × |f₃ - f₂| × |f₄ - f₂|
             × [D_c + (λ²/(2c)) × (|f₃ - f₂| + |f₄ - f₂|) × dD_c/dλ]

        其中 λ 取中心波长（f₂ 对应的波长）。

        Parameters
        ----------
        f2 : float or ndarray
            频率 f₂ [Hz] (由频率匹配确定: f₂ = f₃ + f₄ - f₁)
        f3 : float or ndarray
            频率 f₃ [Hz]
        f4 : float or ndarray
            频率 f₄ [Hz]

        Returns
        -------
        float or ndarray
            相位失配 Δβ [rad/m]
        """
        f2 = np.asarray(f2, dtype=np.float64)
        f3 = np.asarray(f3, dtype=np.float64)
        f4 = np.asarray(f4, dtype=np.float64)

        # Memoization helps repeated scalar/small-array calls, but large arrays
        # are usually one-off in continuous FWM sweeps. Building tuple keys for
        # those arrays can dominate runtime, so skip caching above this size.
        cache_key = None
        if max(f2.size, f3.size, f4.size) <= 4096:
            cache_key = (_array_cache_key(f2), _array_cache_key(f3), _array_cache_key(f4))
            if cache_key in self._pm_cache:
                return self._pm_cache[cache_key]

        # 使用 f₂ 对应波长作为展开中心
        lambda_c = c_light / f2

        df32 = np.abs(f3 - f2)
        df42 = np.abs(f4 - f2)

        # D_c 取 f₂ 处的色散
        D_c = self.get_dispersion_at_freq(f2)

        delta_beta = (
            (2.0 * np.pi * lambda_c**2 / c_light)
            * df32 * df42
            * (D_c + (lambda_c**2 / (2.0 * c_light)) * (df32 + df42) * self._D_slope)
        )
        if cache_key is not None:
            self._pm_cache[cache_key] = delta_beta
        return delta_beta

    def get_effective_length(self, freq: float | np.ndarray | None = None) -> float:
        """计算有效长度 L_eff = (1 - exp(-α·L)) / α。

        Parameters
        ----------
        freq : float or ndarray or None
            频率 [Hz]，用于获取频率相关的衰减。None 使用默认常数
            alpha，此时会发出警告。

        Returns
        -------
        float
            有效长度 [m]。当前实现中即使 freq 为 ndarray，由于 C 波段
            衰减为常数，返回值仍为标量 float。
        """
        if freq is not None:
            alpha = np.asarray(self.get_loss_at_freq(freq))
        else:
            warnings.warn(
                "get_effective_length: freq=None, 使用常数 alpha。"
                " 传入 freq 以启用频率相关衰减。",
                stacklevel=2,
            )
            alpha = self._alpha
        return (1.0 - np.exp(-alpha * self._L)) / alpha
