"""单位转换工具函数。

所有转换公式来源于 docs/parameters.md。
项目内部统一使用SI基本单位，此模块提供常用单位与SI之间的转换。
"""

import numpy as np


def alpha_dB_km_to_per_m(alpha_dB: float) -> float:
    """光纤衰减系数: dB/km → 1/m。

    公式: alpha = alpha_dB * 1e-3 / (10 * log10(e))
    量纲验证: 0.2 dB/km → 0.2 * 1e-3 / 4.343 ≈ 4.606e-5 1/m (即 46.06 1/km)

    Parameters
    ----------
    alpha_dB : float
        衰减系数 [dB/km]

    Returns
    -------
    float
        衰减系数 [1/m]
    """
    return alpha_dB * 1e-3 / (10.0 * np.log10(np.e))


def alpha_per_m_to_dB_km(alpha: float) -> float:
    """光纤衰减系数: 1/m → dB/km。

    Parameters
    ----------
    alpha : float
        衰减系数 [1/m]

    Returns
    -------
    float
        衰减系数 [dB/km]
    """
    return alpha * 10.0 * np.log10(np.e) * 1e3


def gamma_per_W_km_to_per_W_m(gamma_km: float) -> float:
    """非线性系数: 1/(W·km) → 1/(W·m)。

    公式: gamma_SI = gamma * 1e-3

    Parameters
    ----------
    gamma_km : float
        非线性系数 [1/(W·km)]

    Returns
    -------
    float
        非线性系数 [1/(W·m)]
    """
    return gamma_km * 1e-3


def gamma_per_W_m_to_per_W_km(gamma: float) -> float:
    """非线性系数: 1/(W·m) → 1/(W·km)。"""
    return gamma * 1e3


def D_ps_nm_km_to_s_m2(D: float) -> float:
    """色散系数: ps/(nm·km) → s/m²。

    公式: D_c = D * 1e-6

    Parameters
    ----------
    D : float
        色散系数 [ps/(nm·km)]

    Returns
    -------
    float
        色散系数 [s/m²]
    """
    return D * 1e-6


def D_s_m2_to_ps_nm_km(D_c: float) -> float:
    """色散系数: s/m² → ps/(nm·km)。"""
    return D_c * 1e6


def D_slope_ps_nm2_km_to_s_m3(S: float) -> float:
    """色散斜率: ps/(nm²·km) → s/m³。

    公式: D_slope = S * 1e3

    量纲验证:
    1 ps/(nm²·km) = 1e-12 s / ((1e-9 m)² × 1e3 m) = 1e-12 / 1e-15 = 1e3 s/m³

    Parameters
    ----------
    S : float
        色散斜率 [ps/(nm²·km)]

    Returns
    -------
    float
        色散斜率 [s/m³]
    """
    return S * 1e3


def D_slope_s_m3_to_ps_nm2_km(D_slope: float) -> float:
    """色散斜率: s/m³ → ps/(nm²·km)。"""
    return D_slope * 1e-3


def L_km_to_m(L_km: float) -> float:
    """光纤长度: km → m。"""
    return L_km * 1e3


def L_m_to_km(L: float) -> float:
    """光纤长度: m → km。"""
    return L * 1e-3


def power_W_to_dBm(P: float | np.ndarray) -> float | np.ndarray:
    """功率: W → dBm。

    Parameters
    ----------
    P : float or ndarray
        功率 [W]，必须 > 0

    Returns
    -------
    float or ndarray
        功率 [dBm]
    """
    return 10.0 * np.log10(np.asarray(P) * 1e3)


def power_dBm_to_W(P_dBm: float | np.ndarray) -> float | np.ndarray:
    """功率: dBm → W。

    Parameters
    ----------
    P_dBm : float or ndarray
        功率 [dBm]

    Returns
    -------
    float or ndarray
        功率 [W]
    """
    return 10.0 ** (np.asarray(P_dBm) / 10.0) * 1e-3


def freq_Hz_to_wavelength_m(freq: float | np.ndarray) -> float | np.ndarray:
    """频率 → 波长: λ = c / f。

    Parameters
    ----------
    freq : float or ndarray
        频率 [Hz]

    Returns
    -------
    float or ndarray
        波长 [m]
    """
    from scipy.constants import c
    return c / np.asarray(freq)


def wavelength_m_to_freq_Hz(wavelength: float | np.ndarray) -> float | np.ndarray:
    """波长 → 频率: f = c / λ。

    Parameters
    ----------
    wavelength : float or ndarray
        波长 [m]

    Returns
    -------
    float or ndarray
        频率 [Hz]
    """
    from scipy.constants import c
    return c / np.asarray(wavelength)
