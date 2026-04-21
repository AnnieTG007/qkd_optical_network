"""离散 FWM（四波混频）噪声求解器。

公式来源：docs/formulas_fwm.md 第 2-3 节（离散模型）
推导参考：Gao et al., JLT 2025, doi:10.1109/JLT.2025.3610854

算法（方案 B：等间隔索引算术）
---------------------------------
对等间隔 DWDM 网格，信道频率 f_n = f_center + n·g，
则频率匹配条件 f₃ + f₄ - f₂ = f₁ 等价于 n₃ + n₄ - n₂ = n₁。
有效性判断用整数范围检查，无需 searchsorted。

向量化策略：
1. 对量子信道 f₁，在经典信道集合中用 meshgrid 枚举所有 (n₃, n₄) 对
2. 计算 n₂ = n₃ + n₄ - n₁，整数检查 n₂ 是否为有效经典信道索引
3. 排除 n₂ == n₃ 或 n₂ == n₄（SPM/XPM 项）
4. 批量计算 Δβ、Δα、η、D² 并乘以 P₂P₃P₄，对所有有效组合累加

GPU 加速
--------
当 GPU 可用时（检测到 CUDA），compute_fwm_spectrum_conti 自动将批量
计算路径切换至 GPU（_gpu_compute_fwm_spectrum_conti），消除逐频率
Python 循环，实现 ~10-100x 加速。
"""

from __future__ import annotations

import numpy as np

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import SpectrumType, WDMGrid
from qkd_sim.physical.noise.base import NoiseSolver

# GPU acceleration: lazily-imported so CPU-only environments are unaffected.
_gpu_xp: "type | None" = None


def _get_gpu_module():
    """Return (cupy, True) if GPU available, else (numpy, False)."""
    global _gpu_xp
    if _gpu_xp is None:
        try:
            from qkd_sim.utils.gpu_utils import get_array_module, has_cupy

            if has_cupy():
                _gpu_xp = get_array_module()
                return _gpu_xp, True
        except Exception:
            pass
        _gpu_xp = np
    return _gpu_xp, _gpu_xp is not np


def _fwm_efficiency(
    delta_alpha: np.ndarray,
    delta_beta: np.ndarray,
    L: float,
) -> np.ndarray:
    """FWM 效率因子 η。

    公式 2.1.1 (formulas_fwm.md):
        η = [exp(-Δα·L) - 2·exp(-Δα·L/2)·cos(Δβ·L) + 1]
            / [(Δα)²/4 + (Δβ)²]

    Parameters
    ----------
    delta_alpha : ndarray
        衰减差值 Δα = α₃+α₄+α₂-α₁ [1/m]
    delta_beta : ndarray
        相位失配 Δβ [rad/m]
    L : float
        光纤长度 [m]

    Returns
    -------
    ndarray
        FWM 效率因子 η [m²]，形状与输入相同
    """
    numerator = (
        np.exp(-delta_alpha * L)
        - 2.0 * np.exp(-delta_alpha * L / 2.0) * np.cos(delta_beta * L)
        + 1.0
    )
    denominator = (delta_alpha ** 2) / 4.0 + delta_beta ** 2
    return numerator / denominator


def _F_antiderivative(
    l: np.ndarray,
    z_obs: float,
    alpha1: float,
    delta_alpha: np.ndarray,
    delta_beta: np.ndarray,
    L: float,
) -> np.ndarray:
    """后向 FWM 积分原函数 F(l)。

    公式 2.2.6 (formulas_fwm.md):
        F(l) = exp(α₁·z_obs) / denom
               × [ -exp(-A·l)/A
                   - exp(-B·l)/(B²+Δβ²)·(-B·cos(Δβ·l)+Δβ·sin(Δβ·l))
                   - exp(-C·l)/C ]

    辅助变量（函数内部计算）：
        A = Δα + 2·α₁
        B = Δα/2 + 2·α₁
        C = 2·α₁
        denom = (Δα)²/4 + (Δβ)²

    计算 P_{b,1}(0) 时 z_obs = 0，exp(α₁·0) = 1 自动消去。

    Parameters
    ----------
    l : ndarray
        积分变量（光纤位置）[m]
    z_obs : float
        观测位置 [m]（评估后向噪声的位置，通常为 0）
    alpha1 : float
        量子信道衰减 α₁ [1/m]
    delta_alpha : ndarray
        衰减差值 Δα = α_k + α_i + α_j - α₁ [1/m]
    delta_beta : ndarray
        相位失配 Δβ [rad/m]
    L : float
        光纤长度 [m]

    Returns
    -------
    ndarray
        F(l) 的值
    """
    # 辅助变量（内部计算）
    A = delta_alpha + 2.0 * alpha1
    B = delta_alpha / 2.0 + 2.0 * alpha1
    C = 2.0 * alpha1 * np.ones_like(delta_alpha)
    denom = (delta_alpha ** 2) / 4.0 + delta_beta ** 2

    exp_z = np.exp(alpha1 * z_obs)
    term_A = -np.exp(-A * l) / A
    # B 项：分母 B²+Δβ²，乘以三角修正因子 (-B·cos+Δβ·sin)
    term_B = (
        -np.exp(-B * l)
        / (B ** 2 + delta_beta ** 2)
        * (-B * np.cos(delta_beta * l) + delta_beta * np.sin(delta_beta * l))
    )
    term_C = -np.exp(-C * l) / C
    return exp_z / denom * (term_A + term_B + term_C)


def _fwm_efficiency_vec(
    delta_alpha: np.ndarray,
    delta_beta: np.ndarray,
    L_arr: np.ndarray,
) -> np.ndarray:
    """Vectorized FWM efficiency over L dimension.

    delta_alpha: (1, N_a, N_a, 1) — per-frequency pump-pair delta_alpha
    delta_beta:  (1, N_a, N_a, 1)
    L_arr: (N_L,) — multiple fiber lengths [m]

    Returns: (1, N_a, N_a, N_L) FWM efficiency η broadcast over L.
    """
    N_L = L_arr.shape[0]
    da = delta_alpha.astype(np.float64)
    db = delta_beta.astype(np.float64)
    L = L_arr.reshape((1, 1, 1, N_L))

    exp_da_L = np.exp(-da * L)
    exp_da_L_2 = np.exp(-da * L / 2.0)
    cos_db_L = np.cos(db * L)
    numerator = exp_da_L - 2.0 * exp_da_L_2 * cos_db_L + 1.0
    denom = (da ** 2) / 4.0 + db ** 2
    return numerator / denom


def _F_antiderivative_vec(
    l_scalar: float,
    z_obs: float,
    alpha1: float,
    delta_alpha: np.ndarray,
    delta_beta: np.ndarray,
    L_arr: np.ndarray,
) -> np.ndarray:
    """Vectorized F antiderivative over L dimension.

    l_scalar: 0.0 or L — scalar l base (all elements equal to this value)
    alpha1: scalar alpha1 for this f1 point
    delta_alpha: (1, N_a, N_a, 1)
    delta_beta:  (1, N_a, N_a, 1)
    L_arr: (N_L,) — multiple fiber lengths [m]

    Returns: (1, N_a, N_a, N_L)
    """
    N_L = L_arr.shape[0]
    da = delta_alpha.astype(np.float64)
    db = delta_beta.astype(np.float64)
    L = L_arr.reshape((1, 1, 1, N_L))

    a1 = alpha1  # scalar
    A = da + 2.0 * a1
    B = da / 2.0 + 2.0 * a1
    C = 2.0 * a1 * np.ones_like(da)
    denom = (da ** 2) / 4.0 + db ** 2

    exp_z = np.exp(a1 * z_obs)
    # l=0 for lower bound, l=L_arr for upper bound.
    l_4d = np.zeros_like(da) if abs(l_scalar) < 1e-30 else np.ones_like(da) * L
    term_A = -np.exp(-A * l_4d) / A
    term_B = (
        -np.exp(-B * l_4d)
        / (B ** 2 + db ** 2)
        * (-B * np.cos(db * l_4d) + db * np.sin(db * l_4d))
    )
    term_C = -np.exp(-C * l_4d) / C
    return exp_z / denom * (term_A + term_B + term_C)


# =============================================================================
# GPU-accelerated helpers (CuPy equivalents of the CPU vectorized functions)
# =============================================================================
def _gpu_fwm_efficiency(
    delta_alpha, delta_beta, L
) -> "ndarray":
    """GPU FWM efficiency η — element-wise on GPU arrays.

    delta_alpha, delta_beta: CuPy arrays, shape broadcastable to each other.
    L: scalar float (fiber length [m]).

    Returns CuPy array.
    """
    xp, _ = _get_gpu_module()
    exp_da_L = xp.exp(-delta_alpha * L)
    exp_da_L_2 = xp.exp(-delta_alpha * L / 2.0)
    cos_db_L = xp.cos(delta_beta * L)
    numerator = exp_da_L - 2.0 * exp_da_L_2 * cos_db_L + 1.0
    denominator = (delta_alpha ** 2) / 4.0 + delta_beta ** 2
    return numerator / denominator


def _gpu_fwm_efficiency_vec(
    delta_alpha, delta_beta, L_arr
) -> "ndarray":
    """GPU vectorized FWM efficiency over L dimension.

    delta_alpha, delta_beta: CuPy arrays, shape (1, N_a, N_a, 1) or broadcastable.
    L_arr: CuPy array, shape (N_L,).

    Returns CuPy array shape (..., N_L).
    """
    xp, _ = _get_gpu_module()
    N_L = L_arr.shape[0]
    da = delta_alpha.astype(xp.float64)
    db = delta_beta.astype(xp.float64)
    L = L_arr.reshape((1, 1, 1, N_L))
    exp_da_L = xp.exp(-da * L)
    exp_da_L_2 = xp.exp(-da * L / 2.0)
    cos_db_L = xp.cos(db * L)
    numerator = exp_da_L - 2.0 * exp_da_L_2 * cos_db_L + 1.0
    denom = (da ** 2) / 4.0 + db ** 2
    return numerator / denom


def _gpu_F_antiderivative(
    l, z_obs, alpha1, delta_alpha, delta_beta, L
) -> "ndarray":
    """GPU F antiderivative F(l) — element-wise.

    All inputs are CuPy arrays; l may be an array (same shape as delta_alpha),
    or a scalar that broadcasts.
    Returns CuPy array.
    """
    xp, _ = _get_gpu_module()
    a1 = xp.asarray(alpha1, dtype=xp.float64)
    if a1.ndim == 1 and delta_alpha.ndim > 1:
        a1 = a1.reshape((a1.shape[0],) + (1,) * (delta_alpha.ndim - 1))
    A = delta_alpha + 2.0 * a1
    B = delta_alpha / 2.0 + 2.0 * a1
    C = xp.ones_like(delta_alpha) * (2.0 * a1)
    denom = (delta_alpha ** 2) / 4.0 + delta_beta ** 2
    exp_z = xp.exp(a1 * z_obs)
    term_A = -xp.exp(-A * l) / A
    term_B = (
        -xp.exp(-B * l)
        / (B ** 2 + delta_beta ** 2)
        * (-B * xp.cos(delta_beta * l) + delta_beta * xp.sin(delta_beta * l))
    )
    term_C = -xp.exp(-C * l) / C
    return exp_z / denom * (term_A + term_B + term_C)


def _gpu_F_antiderivative_vec(
    l_scalar, z_obs, alpha1, delta_alpha, delta_beta, L_arr
) -> "ndarray":
    """GPU vectorized F antiderivative over L dimension.

    Returns CuPy array shape (1, N_a, N_a, N_L).
    """
    xp, _ = _get_gpu_module()
    N_L = L_arr.shape[0]
    da = delta_alpha.astype(xp.float64)
    db = delta_beta.astype(xp.float64)
    L = L_arr.reshape((1, 1, 1, N_L))
    a1 = xp.asarray(alpha1, dtype=xp.float64)
    if a1.ndim == 1 and da.ndim > 1:
        a1 = a1.reshape((a1.shape[0],) + (1,) * (da.ndim - 1))
    A = da + 2.0 * a1
    B = da / 2.0 + 2.0 * a1
    C = xp.ones_like(da) * (2.0 * a1)
    denom = (da ** 2) / 4.0 + db ** 2
    exp_z = xp.exp(a1 * z_obs)
    l_4d = xp.zeros_like(da) if abs(l_scalar) < 1e-30 else xp.ones_like(da) * L
    term_A = -xp.exp(-A * l_4d) / A
    term_B = (
        -xp.exp(-B * l_4d)
        / (B ** 2 + db ** 2)
        * (-B * xp.cos(db * l_4d) + db * xp.sin(db * l_4d))
    )
    term_C = -xp.exp(-C * l_4d) / C
    return exp_z / denom * (term_A + term_B + term_C)


def _gpu_phase_mismatch(
    f2, f3, f4, D_c, D_slope
) -> "ndarray":
    """GPU batch phase mismatch Δβ = f(f2, f3, f4) on CuPy arrays.

    All inputs: CuPy arrays. D_c and D_slope are Python floats.

    Formula 2.2.3 (formulas_fwm.md):
        Δβ = (2π λ² / c) × |f₃-f₂| × |f₄-f₂|
             × [D_c + (λ²/(2c)) × (|f₃-f₂|+|f₄-f₂|) × D_slope]
        λ = c / f2 (centered at f2)
    """
    xp, _ = _get_gpu_module()
    # Use same constant value as scipy.constants but inline for GPU compatibility
    c_light = 299792458.0  # m/s
    lambda_c = c_light / f2  # (...,) CuPy array
    df32 = xp.abs(f3 - f2)
    df42 = xp.abs(f4 - f2)
    delta_beta = (
        (2.0 * xp.pi * lambda_c ** 2 / c_light)
        * df32
        * df42
        * (D_c + (lambda_c ** 2 / (2.0 * c_light)) * (df32 + df42) * D_slope)
    )
    return delta_beta


class DiscreteFWMSolver(NoiseSolver):
    """离散 FWM 噪声求解器。

    计算量子信道受经典信道四波混频产生的前向/后向噪声功率。

    公式来源：docs/formulas_fwm.md 第 2-3 节
    算法：方案 B（等间隔索引算术），复杂度 O(N_q × N_c²)。

    Parameters
    ----------
    channel_spacing : float or None
        信道间隔 [Hz]。None 时从 wdm_grid 自动推断（要求等间隔）。

    Examples
    --------
    >>> solver = DiscreteFWMSolver()
    >>> P_fwd = solver.compute_forward(fiber, wdm_grid)  # shape (N_q,)
    >>> P_bwd = solver.compute_backward(fiber, wdm_grid)  # shape (N_q,)
    """

    def __init__(self, channel_spacing: float | None = None) -> None:
        self.channel_spacing = channel_spacing
        # 推断后缓存（首次调用时计算，后续直接复用）
        self._inferred_spacing: float | None = None
        # 拓扑级缓存：避免重复计算 FWM 有效三元组
        self._cache_key: tuple[int, ...] | None = None
        self._classical_set_cache: np.ndarray | None = None  # is_classical[bool] array
        self._valid_combo_cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] | None = None  # n1 → (n2, n3, n4)

    def _get_valid_combinations(
        self,
        n1: int,
        idx_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """枚举对量子信道 n₁ 产生 FWM 噪声的有效经典信道三元组 (n₂, n₃, n₄)。

        频率匹配：n₃ + n₄ - n₂ = n₁，即 n₂ = n₃ + n₄ - n₁。
        约束：n₂ ∈ idx_c，n₂ ≠ n₃，n₂ ≠ n₄。

        使用拓扑级缓存：相同 idx_c 的多次调用复用 meshgrid +
        O(1) bool 数组成员判断，而非重复 O(N_c²) 的 np.isin。

        Parameters
        ----------
        n1 : int
            量子信道在全局整数网格中的索引
        idx_c : ndarray, shape (N_c,)
            所有经典信道的整数索引

        Returns
        -------
        n2_valid, n3_valid, n4_valid : ndarray, shape (N_valid,)
            有效组合的信道索引数组
        """
        cache_key = tuple(sorted(idx_c.tolist()))

        # 检查缓存是否命中
        if (
            self._cache_key == cache_key
            and self._valid_combo_cache is not None
            and n1 in self._valid_combo_cache
        ):
            return self._valid_combo_cache[n1]

        # 缓存未命中或失效：重建
        self._cache_key = cache_key

        # 构建 (n₃, n₄) 全网格
        n3_grid, n4_grid = np.meshgrid(idx_c, idx_c, indexing="ij")
        # n2_grid 随 n1 变化（但 meshgrid 可复用）
        n2_grid = n3_grid + n4_grid - n1  # shape (N_c, N_c)

        # O(N_c) sorted-array membership test: np.searchsorted(idx_c, n2_grid.ravel())
        # returns insertion positions; check they land on actual idx_c values
        sorted_idx_c = np.sort(idx_c)
        positions = np.searchsorted(sorted_idx_c, n2_grid.ravel())
        positions = np.clip(positions, 0, len(sorted_idx_c) - 1)
        in_classical_flat = sorted_idx_c.ravel()[positions] == n2_grid.ravel()
        in_classical = in_classical_flat.reshape(n2_grid.shape)
        not_spm_xpm = (n2_grid != n3_grid) & (n2_grid != n4_grid)
        valid_mask = in_classical & not_spm_xpm

        result = (n2_grid[valid_mask], n3_grid[valid_mask], n4_grid[valid_mask])

        # 惰性初始化组合缓存字典
        if self._valid_combo_cache is None:
            self._valid_combo_cache = {}
        self._valid_combo_cache[n1] = result

        return result

    def _compute_noise_for_channel(
        self,
        f1: float,
        n1: int,
        idx_c: np.ndarray,
        f_all: np.ndarray,
        P_all: np.ndarray,
        fiber: Fiber,
        direction: str,
    ) -> float:
        """计算单个量子信道的 FWM 噪声功率。

        Parameters
        ----------
        f1 : float
            量子信道频率 [Hz]
        n1 : int
            量子信道在全局整数网格中的索引
        idx_c : ndarray, shape (N_c,)
            经典信道整数索引
        f_all : ndarray
            全局频率网格（f_all[n] = f_center + n·g）[Hz]
        P_all : ndarray
            全局功率数组（经典信道处为 P_ch，其余为 0）[W]
        fiber : Fiber
        direction : str
            "forward" | "backward"

        Returns
        -------
        float
            量子信道 f₁ 处的 FWM 噪声功率 [W]
        """
        n2_arr, n3_arr, n4_arr = self._get_valid_combinations(n1, idx_c)
        if n2_arr.size == 0:
            return 0.0

        f2 = f_all[n2_arr]
        f3 = f_all[n3_arr]
        f4 = f_all[n4_arr]

        P2 = P_all[n2_arr]
        P3 = P_all[n3_arr]
        P4 = P_all[n4_arr]

        # 简并因子 D（公式 2.2.1）
        D = np.where(n3_arr == n4_arr, 3.0, 6.0)

        # 衰减差值 Δα = α(f₃)+α(f₄)+α(f₂)-α(f₁)（精确公式）
        alpha1 = fiber.get_loss_at_freq(f1)  # 标量
        alpha2 = fiber.get_loss_at_freq(f2)  # shape (N_valid,)
        alpha3 = fiber.get_loss_at_freq(f3)  # shape (N_valid,)
        alpha4 = fiber.get_loss_at_freq(f4)  # shape (N_valid,)
        delta_alpha = alpha4 + alpha3 + alpha2 - alpha1

        # 相位失配 Δβ（调用 Fiber 接口，见 docs/formulas_fwm.md 效率因子节）
        delta_beta = fiber.get_phase_mismatch(f2=f2, f3=f3, f4=f4)

        L = fiber.L
        gamma = fiber.gamma

        if direction == "forward":
            # 公式 2.2.1
            eta = _fwm_efficiency(delta_alpha, delta_beta, L)
            contributions = (
                np.exp(-alpha1 * L) * (gamma ** 2 / 9.0)
                * D ** 2 * eta * P2 * P3 * P4
            )
            return float(contributions.sum())

        else:  # backward
            # 公式 2.2.4-2.2.7：瑞利散射积分解析式
            # z_obs = 0（光纤发射端），exp(α₁·0) = 1
            # 辅助变量 A、B、C、denom 由 _F_antiderivative 内部计算
            F_L = _F_antiderivative(
                l=np.full_like(delta_alpha, L), z_obs=0.0,
                alpha1=alpha1, delta_alpha=delta_alpha, delta_beta=delta_beta, L=L,
            )
            F_0 = _F_antiderivative(
                l=np.zeros_like(delta_alpha), z_obs=0.0,
                alpha1=alpha1, delta_alpha=delta_alpha, delta_beta=delta_beta, L=L,
            )

            contributions = (
                fiber.rayleigh_coeff * (gamma ** 2 / 9.0)
                * D ** 2 * P2 * P3 * P4
                * (F_L - F_0)
            )
            return float(contributions.sum())

    def _prepare_grid(
        self, fiber: Fiber, wdm_grid: WDMGrid
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """从 WDMGrid 构建整数索引网格。

        Returns
        -------
        f_q : ndarray, shape (N_q,)  量子信道频率 [Hz]
        n_q : ndarray, shape (N_q,)  量子信道整数索引
        idx_c : ndarray, shape (N_c,)  经典信道整数索引
        f_all : ndarray  全局频率数组（f_center + n·g）[Hz]
        P_all : ndarray  全局功率数组 [W]（经典信道处为功率，其余 0）
        g : float  信道间隔 [Hz]
        """
        all_freqs = wdm_grid.get_channel_frequencies()
        all_powers = wdm_grid.get_channel_powers()

        # 推断信道间隔（结果缓存，避免扫描循环中重复排序）
        if self.channel_spacing is not None:
            g = self.channel_spacing
        elif self._inferred_spacing is not None:
            g = self._inferred_spacing
        else:
            diffs = np.diff(np.sort(all_freqs))
            assert diffs.size > 0, "至少需要 2 个信道才能推断间隔"
            g = float(np.round(diffs.min()))
            assert np.allclose(diffs, g, rtol=1e-6), (
                f"信道间隔不等：{diffs}，请手动指定 channel_spacing"
            )
            self._inferred_spacing = g

        # 将所有信道频率映射到整数索引（相对于最小频率）
        f_min = all_freqs.min()
        all_indices = np.round((all_freqs - f_min) / g).astype(int)

        # 全局数组（索引 0 对应 f_min）
        n_max = all_indices.max()
        f_all = f_min + np.arange(n_max + 1) * g
        P_all = np.zeros(n_max + 1)
        for idx, p in zip(all_indices, all_powers):
            P_all[idx] = p

        # 量子信道整数索引
        q_chs = wdm_grid.get_quantum_channels()
        c_chs = wdm_grid.get_classical_channels()
        f_q = np.array([ch.f_center for ch in q_chs])
        n_q = np.round((f_q - f_min) / g).astype(int)

        # 经典信道整数索引
        f_c = np.array([ch.f_center for ch in c_chs])
        idx_c = np.round((f_c - f_min) / g).astype(int)

        # 清零量子信道处的功率（量子信道功率不参与 FWM 泵浦）
        for nq in n_q:
            P_all[nq] = 0.0

        return f_q, n_q, idx_c, f_all, P_all, g

    def _compute_noise_for_channel_vec(
        self,
        n1_arr: np.ndarray,
        idx_c: np.ndarray,
        f_all: np.ndarray,
        P_all: np.ndarray,
        alpha1_arr: np.ndarray,
        fiber: Fiber,
        direction: str,
    ) -> np.ndarray:
        """Vectorized FWM noise for multiple quantum channels sharing the same alpha1_arr.

        Uses topology cache: all n1 in n1_arr share the same idx_c, so meshgrid
        and searchsorted are done once, then indexed per-n1.

        Parameters
        ----------
        n1_arr : ndarray, shape (N_q,)
            Quantum channel integer indices
        idx_c : ndarray, shape (N_c,)
            Classical channel integer indices
        f_all : ndarray
            Global frequency array
        P_all : ndarray
            Global power array
        alpha1_arr : ndarray, shape (N_q,)
            Precomputed fiber loss at each quantum channel frequency
        fiber : Fiber
        direction : str
            "forward" | "backward"

        Returns
        -------
        ndarray, shape (N_q,)
            FWM noise power for each quantum channel
        """
        n_q = len(n1_arr)
        P_result = np.zeros(n_q, dtype=np.float64)

        # Build the full (n3, n4) meshgrid once (shared across all n1)
        n3_grid, n4_grid = np.meshgrid(idx_c, idx_c, indexing="ij")
        n2_grid_base = n3_grid + n4_grid  # shape (N_c, N_c), n2 = base - n1

        # O(N_c) sorted-array membership test setup (shared across all n1)
        sorted_idx_c = np.sort(idx_c)

        L = fiber.L
        gamma = fiber.gamma

        for i in range(n_q):
            n1 = int(n1_arr[i])
            alpha1 = alpha1_arr[i]

            # n2 specific to this n1
            n2_grid = n2_grid_base - n1

            positions = np.searchsorted(sorted_idx_c, n2_grid.ravel())
            positions = np.clip(positions, 0, len(sorted_idx_c) - 1)
            in_classical_flat = sorted_idx_c.ravel()[positions] == n2_grid.ravel()
            in_classical = in_classical_flat.reshape(n2_grid.shape)
            not_spm_xpm = (n2_grid != n3_grid) & (n2_grid != n4_grid)
            valid_mask = in_classical & not_spm_xpm

            if not np.any(valid_mask):
                continue

            n2_valid = n2_grid[valid_mask]
            n3_valid = n3_grid[valid_mask]
            n4_valid = n4_grid[valid_mask]

            f2 = f_all[n2_valid]
            f3 = f_all[n3_valid]
            f4 = f_all[n4_valid]

            P2 = P_all[n2_valid]
            P3 = P_all[n3_valid]
            P4 = P_all[n4_valid]

            D = np.where(n3_valid == n4_valid, 3.0, 6.0)

            alpha2 = fiber.get_loss_at_freq(f2)
            alpha3 = fiber.get_loss_at_freq(f3)
            alpha4 = fiber.get_loss_at_freq(f4)
            delta_alpha = alpha4 + alpha3 + alpha2 - alpha1

            delta_beta = fiber.get_phase_mismatch(f2=f2, f3=f3, f4=f4)

            if direction == "forward":
                eta = _fwm_efficiency(delta_alpha, delta_beta, L)
                contributions = (
                    np.exp(-alpha1 * L) * (gamma ** 2 / 9.0)
                    * D ** 2 * eta * P2 * P3 * P4
                )
                P_result[i] = float(contributions.sum())
            else:
                F_L = _F_antiderivative(
                    l=np.full_like(delta_alpha, L), z_obs=0.0,
                    alpha1=alpha1, delta_alpha=delta_alpha, delta_beta=delta_beta, L=L,
                )
                F_0 = _F_antiderivative(
                    l=np.zeros_like(delta_alpha), z_obs=0.0,
                    alpha1=alpha1, delta_alpha=delta_alpha, delta_beta=delta_beta, L=L,
                )
                contributions = (
                    fiber.rayleigh_coeff * (gamma ** 2 / 9.0)
                    * D ** 2 * P2 * P3 * P4
                    * (F_L - F_0)
                )
                P_result[i] = float(contributions.sum())

        return P_result

    def compute_forward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算各量子信道在光纤接收端（z=L）的前向 FWM 噪声功率。

        公式 2.2.1 (formulas_fwm.md)

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid

        Returns
        -------
        ndarray, shape (N_q,)
            前向 FWM 噪声功率 [W]
        """
        f_q, n_q, idx_c, f_all, P_all, _ = self._prepare_grid(fiber, wdm_grid)
        alpha1_arr = fiber.get_loss_at_freq(f_q)
        return self._compute_noise_for_channel_vec(
            n_q, idx_c, f_all, P_all, alpha1_arr, fiber, direction="forward",
        )

    def compute_backward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算各量子信道在光纤发射端（z=0）的后向 FWM 噪声功率。

        公式 2.2.4-2.2.7 (formulas_fwm.md)

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid

        Returns
        -------
        ndarray, shape (N_q,)
            后向 FWM 噪声功率 [W]
        """
        f_q, n_q, idx_c, f_all, P_all, _ = self._prepare_grid(fiber, wdm_grid)
        alpha1_arr = fiber.get_loss_at_freq(f_q)
        return self._compute_noise_for_channel_vec(
            n_q, idx_c, f_all, P_all, alpha1_arr, fiber, direction="backward",
        )

    # --- 连续模型方法 -------------------------------------------------------

    @staticmethod
    def _validate_frequency_grid(f_grid: np.ndarray) -> float:
        """验证 1D 积分网格并返回频率步长。

        公式 2.3.1-2.3.6 (formulas_fwm.md) 使用均匀网格黎曼和计算。
        """
        f_grid = np.asarray(f_grid, dtype=np.float64)
        assert f_grid.ndim == 1 and f_grid.size >= 2, (
            f"f_grid must be 1D with at least 2 points, got shape {f_grid.shape}"
        )
        diffs = np.diff(f_grid)
        assert np.all(diffs > 0.0), "f_grid must be strictly increasing"
        df = float(np.mean(diffs))
        assert np.allclose(diffs, df, rtol=1e-6, atol=0.0), (
            "f_grid must be approximately uniform for continuous integration"
        )
        return df

    @staticmethod
    def _build_total_classical_psd(
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        df: float,
    ) -> np.ndarray:
        """在 f_grid 上构建经典信道总发射 PSD。

        SINGLE_FREQ 信道：使用 G=P/df 放在最近格点，
        使得 sum(G*df) ≈ P，用于离散/连续交叉验证极限。
        """
        classical_channels = wdm_grid.get_classical_channels()
        assert len(classical_channels) > 0, "WDMGrid 中无经典信道（泵浦）"

        total_psd = np.zeros(f_grid.size, dtype=np.float64)
        for ch in classical_channels:
            if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
                idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
                total_psd[idx] += ch.power / df
            else:
                total_psd += ch.get_psd(f_grid)
        return total_psd

    @staticmethod
    def _target_noise_bandwidth(ch, df: float) -> float:
        """返回公式 2.3.6 的目标积分带宽。

        SINGLE_FREQ 量子信道使用一个频率 bins df。
        """
        if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
            return df
        return ch.B_s

    def compute_forward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续前向 FWM 噪声功率 [公式 2.3.1 + 2.3.6]。

        在量子信道中心频率处计算 G_{f,1}(L) PSD，
        再乘以带宽得 in-band 噪声功率。
        当所有经典信道均为 SINGLE_FREQ 时，退化为离散模型（交叉验证极限）。
        """
        c_chs = wdm_grid.get_classical_channels()
        if len(c_chs) > 0 and all(
            ch.spectrum_type == SpectrumType.SINGLE_FREQ for ch in c_chs
        ):
            return self.compute_forward(fiber, wdm_grid)

        df = self._validate_frequency_grid(f_grid)
        q_chs = wdm_grid.get_quantum_channels()
        assert len(q_chs) > 0, "WDMGrid 中无量子信道"

        f_grid = np.asarray(f_grid, dtype=np.float64)
        G_tx = self._build_total_classical_psd(wdm_grid, f_grid, df)
        alpha_grid = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)

        G_min = G_tx.max() * 1e-5  # -50 dB relative threshold
        active = G_tx > G_min
        if not np.any(active):
            return np.zeros(len(q_chs), dtype=np.float64)

        f_active = f_grid[active]
        G_active = G_tx[active]
        alpha_active = alpha_grid[active]

        Fi, Fj = np.meshgrid(f_active, f_active, indexing="ij")
        Gi, Gj = np.meshgrid(G_active, G_active, indexing="ij")
        alpha_i, alpha_j = np.meshgrid(alpha_active, alpha_active, indexing="ij")

        D = np.where(np.abs(Fi - Fj) <= 0.5 * df, 3.0, 6.0)

        L = fiber.L
        gamma = fiber.gamma

        P_fwd = np.zeros(len(q_chs), dtype=np.float64)

        for iq, q_ch in enumerate(q_chs):
            f1 = float(q_ch.f_center)
            alpha1 = float(np.asarray(fiber.get_loss_at_freq(f1), dtype=np.float64))

            Fk = Fi + Fj - f1
            Gk = np.interp(Fk, f_grid, G_tx, left=0.0, right=0.0)
            if not np.any(Gk > 0.0):
                continue

            alpha_k = np.interp(Fk, f_grid, alpha_grid, left=alpha_grid[0], right=alpha_grid[-1])
            delta_alpha = alpha_k + alpha_i + alpha_j - alpha1
            delta_beta = fiber.get_phase_mismatch(f2=Fk, f3=Fi, f4=Fj)

            eta = _fwm_efficiency(delta_alpha=delta_alpha, delta_beta=delta_beta, L=L)
            integrand = (D ** 2) * eta * Gi * Gj * Gk
            integral_ij = np.sum(integrand) * df * df

            G_fwd_psd = 4.0 * (gamma ** 2) * np.exp(-alpha1 * L) * integral_ij / 9.0
            bw_q = self._target_noise_bandwidth(q_ch, df)
            P_fwd[iq] = G_fwd_psd * bw_q

        assert P_fwd.shape == (len(q_chs),)
        return P_fwd

    def compute_backward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续后向 FWM 噪声功率 [公式 2.3.2-2.3.6]。

        通过瑞利重分配的前向 FWM 计算 G_{b,1}(0) PSD，
        再乘以带宽得 in-band 噪声功率。
        当所有经典信道均为 SINGLE_FREQ 时，退化为离散模型（交叉验证极限）。
        """
        c_chs = wdm_grid.get_classical_channels()
        if len(c_chs) > 0 and all(
            ch.spectrum_type == SpectrumType.SINGLE_FREQ for ch in c_chs
        ):
            return self.compute_backward(fiber, wdm_grid)

        df = self._validate_frequency_grid(f_grid)
        q_chs = wdm_grid.get_quantum_channels()
        assert len(q_chs) > 0, "WDMGrid 中无量子信道"

        f_grid = np.asarray(f_grid, dtype=np.float64)
        G_tx = self._build_total_classical_psd(wdm_grid, f_grid, df)
        alpha_grid = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)

        G_min = G_tx.max() * 1e-5  # -50 dB relative threshold
        active = G_tx > G_min
        if not np.any(active):
            return np.zeros(len(q_chs), dtype=np.float64)

        f_active = f_grid[active]
        G_active = G_tx[active]
        alpha_active = alpha_grid[active]

        Fi, Fj = np.meshgrid(f_active, f_active, indexing="ij")
        Gi, Gj = np.meshgrid(G_active, G_active, indexing="ij")
        alpha_i, alpha_j = np.meshgrid(alpha_active, alpha_active, indexing="ij")

        D = np.where(np.abs(Fi - Fj) <= 0.5 * df, 3.0, 6.0)

        L = fiber.L
        gamma = fiber.gamma

        P_bwd = np.zeros(len(q_chs), dtype=np.float64)

        for iq, q_ch in enumerate(q_chs):
            f1 = float(q_ch.f_center)
            alpha1 = float(np.asarray(fiber.get_loss_at_freq(f1), dtype=np.float64))

            Fk = Fi + Fj - f1
            Gk = np.interp(Fk, f_grid, G_tx, left=0.0, right=0.0)
            if not np.any(Gk > 0.0):
                continue

            alpha_k = np.interp(Fk, f_grid, alpha_grid, left=alpha_grid[0], right=alpha_grid[-1])
            delta_alpha = alpha_k + alpha_i + alpha_j - alpha1
            delta_beta = fiber.get_phase_mismatch(f2=Fk, f3=Fi, f4=Fj)

            # 辅助变量 A、B、C、denom 由 _F_antiderivative 内部计算
            F_L = _F_antiderivative(
                l=np.full_like(delta_alpha, L),
                z_obs=0.0,
                alpha1=alpha1,
                delta_alpha=delta_alpha,
                delta_beta=delta_beta,
                L=L,
            )
            F_0 = _F_antiderivative(
                l=np.zeros_like(delta_alpha),
                z_obs=0.0,
                alpha1=alpha1,
                delta_alpha=delta_alpha,
                delta_beta=delta_beta,
                L=L,
            )

            integrand = (D ** 2) * Gi * Gj * Gk * (F_L - F_0)
            integral_ij = np.sum(integrand) * df * df

            G_bwd_psd = fiber.rayleigh_coeff * (gamma ** 2) * integral_ij / 9.0
            bw_q = self._target_noise_bandwidth(q_ch, df)
            P_bwd[iq] = G_bwd_psd * bw_q

        assert P_bwd.shape == (len(q_chs),)
        return P_bwd

    # --- 噪声 PSD 谱计算（向量优化版）-----------

    def compute_fwm_spectrum_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str = "forward",
        L_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """计算 FWM 噪声 PSD G_fwm(f) [W/Hz]，在 f_grid 每个频率点评估。

        返回 shape (N_f,) 的噪声功率谱密度数组；当 L_arr 给出时，
        通过逐个标量长度重复原始计算路径，返回 shape (N_f, N_L)。
        用于绘制连续噪声功率谱曲线（而非信道积分噪声标量）。

        GPU 加速：当 GPU 可用时，自动切换至 compute_fwm_spectrum_conti_gpu，
        将逐频率 Python 循环替换为 GPU 批量计算（约 10-100x 加速）。

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid
        f_grid : ndarray
            输出频率网格 [Hz]
        direction : {"forward", "backward"}
        L_arr : ndarray | None
            多个光纤长度 [m]，shape (N_L,)。None 时使用 fiber.L。

        Returns
        -------
        ndarray, shape (N_f,) or (N_f, N_L)
            G_fwm(f) [W/Hz] at each f_grid point
        """
        # GPU path: offload the full batch computation to GPU.
        _, is_gpu = _get_gpu_module()
        if is_gpu:
            return self.compute_fwm_spectrum_conti_gpu(
                fiber, wdm_grid, f_grid, direction=direction, L_arr=L_arr
            )

        f_grid = np.asarray(f_grid, dtype=np.float64)
        self._validate_frequency_grid(f_grid)

        df = float(np.mean(np.diff(f_grid)))
        G_tx = self._build_total_classical_psd(wdm_grid, f_grid, df)
        alpha_grid = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)

        G_min = G_tx.max() * 1e-5  # -50 dB relative threshold
        active = G_tx > G_min
        if not np.any(active):
            if L_arr is None:
                return np.zeros_like(f_grid, dtype=np.float64)
            L_values = np.asarray(L_arr, dtype=np.float64).reshape(-1)
            return np.zeros((f_grid.size, L_values.size), dtype=np.float64)

        f_active = f_grid[active]
        G_active = G_tx[active]
        alpha_active = alpha_grid[active]

        Fi, Fj = np.meshgrid(f_active, f_active, indexing="ij")
        Gi, Gj = np.meshgrid(G_active, G_active, indexing="ij")
        alpha_i, alpha_j = np.meshgrid(alpha_active, alpha_active, indexing="ij")
        D = np.where(np.abs(Fi - Fj) <= 0.5 * df, 3.0, 6.0)

        # Precompute 2D pump-pair frequency sum (independent of output f1)
        Fk_ij = Fi + Fj  # shape (N_active, N_active)

        # Keep CPU evaluation in float64 so the batched L_arr path matches the
        # per-length loop within test tolerance.
        Fi_f32 = Fi.astype(np.float64)
        Fj_f32 = Fj.astype(np.float64)
        Gi_f32 = Gi.astype(np.float64)
        Gj_f32 = Gj.astype(np.float64)
        alpha_i_f32 = alpha_i.astype(np.float64)
        alpha_j_f32 = alpha_j.astype(np.float64)
        D_f32 = D.astype(np.float64)
        Fk_ij_f32 = (Fi_f32 + Fj_f32).astype(np.float64)
        Fi_f64 = Fi_f32.astype(np.float64)
        Fj_f64 = Fj_f32.astype(np.float64)

        pump_grid_f32 = f_grid.astype(np.float64)
        pump_G_f32 = G_tx.astype(np.float64)
        alpha_grid_f32 = alpha_grid.astype(np.float64)

        n_f = f_grid.size

        # For large N_active, fall back to per-point evaluation to avoid
        # O(N_f × N_active²) batched memory pressure. Use chunk_size=10
        # to keep memory bounded while batching searchsorted calls.
        N_a = Fi_f32.shape[0]
        N_pairs = N_a * N_a
        chunk_size = 10 if N_pairs * N_a > 50_000_000 else 50

        def compute_for_length(L: float) -> np.ndarray:
            """Compute FWM spectrum for a single fiber length L [m]."""
            out = np.zeros(n_f, dtype=np.float64)
            # Precompute alpha1 for ALL f_grid points once (eliminates per-iteration fiber call)
            alpha1_full = fiber.get_loss_at_freq(f_grid)  # (N_f,)

            for start in range(0, n_f, chunk_size):
                end = min(start + chunk_size, n_f)
                f1_chunk = f_grid[start:end]  # shape (chunk,)
                n_chunk = end - start
                alpha1_chunk = alpha1_full[start:end]  # (chunk,)

                # Fk for all pump pairs × all f1 in this chunk
                # Fk_ij: (N_a, N_a), f1_chunk: (chunk,) → (N_a, N_a, chunk)
                Fk = Fk_ij_f32[:, :, np.newaxis] - f1_chunk[np.newaxis, np.newaxis, :]

                # ---- batch interpolation for Gk: pump power at Fk ----
                Fk_T = Fk.transpose(2, 0, 1).reshape(n_chunk, -1)  # (chunk, N_pairs)
                sorted_idx = np.searchsorted(pump_grid_f32, Fk_T)
                sorted_idx = np.clip(sorted_idx, 1, len(pump_grid_f32) - 1)

                x0 = pump_grid_f32[sorted_idx - 1]
                x1 = pump_grid_f32[sorted_idx]
                y0 = pump_G_f32[sorted_idx - 1]
                y1 = pump_G_f32[sorted_idx]
                t = np.clip((Fk_T - x0) / (x1 - x0), 0.0, 1.0)
                Gk_T = (y0 + (y1 - y0) * t).astype(np.float64)  # (chunk, N_pairs)

                # ---- batch interpolation for alpha_k: fiber loss at Fk ----
                a0 = alpha_grid_f32[sorted_idx - 1]
                a1_interp = alpha_grid_f32[sorted_idx]
                alpha_k_T = (a0 + (a1_interp - a0) * t).astype(np.float64)  # (chunk, N_pairs)

                for j in range(n_chunk):
                    alpha1 = alpha1_chunk[j]
                    Gk_j = Gk_T[j, :]  # (N_pairs,)
                    alpha_k_j = alpha_k_T[j, :]
                    valid_mask_2d = (Gk_j > 0.0).reshape(N_a, N_a)

                    if not np.any(valid_mask_2d):
                        out[start + j] = 0.0
                        continue

                    Gk_2d = Gk_j.reshape(N_a, N_a)
                    alpha_k_2d = alpha_k_j.reshape(N_a, N_a)

                    delta_alpha = alpha_k_2d + alpha_i_f32 + alpha_j_f32 - float(alpha1)
                    delta_beta = fiber.get_phase_mismatch(
                        f2=(Fk_ij_f32 - float(f1_chunk[j])).astype(np.float64),
                        f3=Fi_f64,
                        f4=Fj_f64,
                    )

                    if direction == "forward":
                        eta = _fwm_efficiency(
                            delta_alpha=delta_alpha.astype(np.float64),
                            delta_beta=delta_beta,
                            L=L,
                        )
                        integrand = (D_f32 ** 2) * eta * Gi_f32 * Gj_f32 * Gk_2d
                        integral_ij = np.sum(integrand[valid_mask_2d]) * df * df
                        out[start + j] = (
                            4.0 * (fiber.gamma ** 2)
                            * np.exp(-float(alpha1) * L)
                            * integral_ij
                            / 9.0
                        )
                    else:
                        alpha1_f64 = float(alpha1)
                        F_L = _F_antiderivative(
                            l=np.full_like(delta_alpha, L),
                            z_obs=0.0,
                            alpha1=alpha1_f64,
                            delta_alpha=delta_alpha,
                            delta_beta=delta_beta,
                            L=L,
                        )
                        F_0 = _F_antiderivative(
                            l=np.zeros_like(delta_alpha),
                            z_obs=0.0,
                            alpha1=alpha1_f64,
                            delta_alpha=delta_alpha,
                            delta_beta=delta_beta,
                            L=L,
                        )
                        integrand = (D_f32 ** 2) * Gi_f32 * Gj_f32 * Gk_2d * (
                            F_L - F_0
                        ).astype(np.float64)
                        integral_ij = np.sum(integrand[valid_mask_2d]) * df * df
                        out[start + j] = (
                            fiber.rayleigh_coeff * (fiber.gamma ** 2) * integral_ij / 9.0
                        )

            return out

        # ---- Vectorized L path: hoist Fk/Gk/alpha_k, broadcast over L ----
        # CPU memory guard: N_active² × N_L floats can exceed available RAM when
        # N_active is large (~1260泵浦). Process L in batches of MAX_L_BATCH to
        # keep peak memory bounded (~50 MiB/array instead of >1 GB).
        _MAX_L_BATCH: int = 4

        def compute_vectorized_L(L_values_arr: np.ndarray) -> np.ndarray:
            """Compute FWM spectrum for all L in batches to avoid OOM on CPU."""
            N_L = L_values_arr.shape[0]
            out = np.zeros((n_f, N_L), dtype=np.float64)
            alpha1_arr = fiber.get_loss_at_freq(f_grid)  # (N_f,)

            # Broadcast pump-pair scalars to (1, N_a, N_a, 1)
            Gi_4d = Gi_f32[np.newaxis, :, :, np.newaxis]
            Gj_4d = Gj_f32[np.newaxis, :, :, np.newaxis]
            D_4d = D_f32[np.newaxis, :, :, np.newaxis]
            alpha_i_4d = alpha_i_f32[np.newaxis, :, :, np.newaxis]
            alpha_j_4d = alpha_j_f32[np.newaxis, :, :, np.newaxis]
            Fk_ij_4d = Fk_ij_f32[np.newaxis, :, :, np.newaxis]

            # Process L in batches to keep memory bounded
            for L_start in range(0, N_L, _MAX_L_BATCH):
                L_end = min(L_start + _MAX_L_BATCH, N_L)
                L_batch = L_values_arr[L_start:L_end]  # (n_L_batch,)
                n_L_batch = L_batch.shape[0]

                for start in range(0, n_f, chunk_size):
                    end = min(start + chunk_size, n_f)
                    f1_chunk = f_grid[start:end]  # (n_chunk,)
                    n_chunk = end - start
                    alpha1_chunk = np.asarray(alpha1_arr[start:end], dtype=np.float64)  # (n_chunk,)

                    # Fk: computed ONCE, shared by all L values
                    Fk = Fk_ij_4d - f1_chunk.reshape((1, 1, 1, n_chunk))  # (1, N_a, N_a, n_chunk)
                    Fk_T = Fk.transpose(3, 1, 2, 0).reshape(n_chunk, -1)  # (n_chunk, N_pairs)
                    sorted_idx = np.clip(np.searchsorted(pump_grid_f32, Fk_T), 1, len(pump_grid_f32) - 1)

                    x0 = pump_grid_f32[sorted_idx - 1]
                    x1 = pump_grid_f32[sorted_idx]
                    y0 = pump_G_f32[sorted_idx - 1]
                    y1 = pump_G_f32[sorted_idx]
                    t = np.clip((Fk_T - x0) / (x1 - x0), 0.0, 1.0)
                    Gk_T = (y0 + (y1 - y0) * t).astype(np.float64)  # (n_chunk, N_pairs)
                    a0 = alpha_grid_f32[sorted_idx - 1]
                    a1_arr = alpha_grid_f32[sorted_idx]
                    alpha_k_T = (a0 + (a1_arr - a0) * t).astype(np.float64)  # (n_chunk, N_pairs)

                    for j in range(n_chunk):
                        Gk_j = Gk_T[j, :]  # (N_pairs,)
                        alpha_k_j = alpha_k_T[j, :]
                        valid_mask_2d = (Gk_j > 0.0).reshape(N_a, N_a)
                        if not np.any(valid_mask_2d):
                            continue

                        Gk_2d = Gk_j.reshape(N_a, N_a)  # (N_a, N_a)
                        alpha_k_2d = alpha_k_j.reshape(N_a, N_a)  # (N_a, N_a)
                        alpha1_j = float(alpha1_chunk[j])
                        f1_j = float(f1_chunk[j])

                        delta_beta_j = fiber.get_phase_mismatch(
                            f2=(Fk_ij_f32 - f1_j).astype(np.float64),
                            f3=Fi_f64,
                            f4=Fj_f64,
                        )  # (N_a, N_a)

                        da_4d = (alpha_k_2d + alpha_i_f32 + alpha_j_f32 - alpha1_j)[
                            np.newaxis, :, :, np.newaxis
                        ]  # (1, N_a, N_a, 1)
                        db_4d = delta_beta_j[np.newaxis, :, :, np.newaxis]  # (1, N_a, N_a, 1)
                        Gk_4d = Gk_2d[np.newaxis, :, :, np.newaxis]  # (1, N_a, N_a, 1)

                        if direction == "forward":
                            eta_4d = _fwm_efficiency_vec(da_4d, db_4d, L_batch)
                            integrand = (
                                (D_4d ** 2) * eta_4d * Gi_4d * Gj_4d * Gk_4d
                            )
                            integral = np.sum(
                                integrand[0, valid_mask_2d, :], axis=0
                            ) * df * df
                            exp_factor = np.exp(-alpha1_j * L_batch)
                            out[start + j, L_start:L_end] = (
                                4.0 * (fiber.gamma ** 2) * exp_factor * integral / 9.0
                            )
                        else:
                            F_L_4d = _F_antiderivative_vec(
                                1.0, 0.0, alpha1_j, da_4d, db_4d, L_batch
                            )
                            F_0_4d = _F_antiderivative_vec(
                                0.0, 0.0, alpha1_j, da_4d, db_4d, L_batch
                            )
                            diff = F_L_4d - F_0_4d  # (1, N_a, N_a, n_L_batch)
                            integrand = (
                                (D_4d ** 2) * Gi_4d * Gj_4d * Gk_4d * diff
                            )
                            integral = np.sum(
                                integrand[0, valid_mask_2d, :], axis=0
                            ) * df * df
                            out[start + j, L_start:L_end] = (
                                fiber.rayleigh_coeff * (fiber.gamma ** 2) * integral / 9.0
                            )

            return out

        if L_arr is None:
            return compute_for_length(fiber.L)

        L_values = np.asarray(L_arr, dtype=np.float64).reshape(-1)
        return compute_vectorized_L(L_values)

    # -------------------------------------------------------------------------
    # GPU-accelerated entry point (called by the CPU path above when GPU available)
    # -------------------------------------------------------------------------

    def compute_fwm_spectrum_conti_gpu(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str = "forward",
        L_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """GPU-accelerated FWM spectrum computation.

        Batches all output-frequency evaluations onto the GPU in a single
        kernel-friendly pass, eliminating the per-frequency Python loop.

        Returns the same shapes as the CPU version.
        """
        xp, is_gpu = _get_gpu_module()
        if not is_gpu:
            # Fallback to CPU if GPU init failed.
            return self.compute_fwm_spectrum_conti(
                fiber, wdm_grid, f_grid, direction=direction, L_arr=L_arr
            )

        from qkd_sim.utils.gpu_utils import to_device, to_host

        f_grid_cpu = np.asarray(f_grid, dtype=np.float64)
        self._validate_frequency_grid(f_grid_cpu)

        df = float(np.mean(np.diff(f_grid_cpu)))
        G_tx_cpu = self._build_total_classical_psd(wdm_grid, f_grid_cpu, df)
        alpha_grid_cpu = np.asarray(fiber.get_loss_at_freq(f_grid_cpu), dtype=np.float64)

        G_min_cpu = G_tx_cpu.max() * 1e-5  # -50 dB relative threshold
        active_cpu = G_tx_cpu > G_min_cpu
        n_f = f_grid_cpu.size
        if not np.any(active_cpu):
            if L_arr is None:
                return np.zeros_like(f_grid_cpu, dtype=np.float64)
            L_values = np.asarray(L_arr, dtype=np.float64).reshape(-1)
            return np.zeros((n_f, L_values.size), dtype=np.float64)

        f_grid = xp.asarray(f_grid_cpu, dtype=xp.float64)
        G_tx = xp.asarray(G_tx_cpu, dtype=xp.float64)
        alpha_grid = xp.asarray(alpha_grid_cpu, dtype=xp.float64)
        active = xp.asarray(active_cpu)
        f_active = f_grid[active]
        G_active = G_tx[active]
        alpha_active = alpha_grid[active]

        Fi, Fj = xp.meshgrid(f_active, f_active, indexing="ij")
        Gi, Gj = xp.meshgrid(G_active, G_active, indexing="ij")
        alpha_i, alpha_j = xp.meshgrid(alpha_active, alpha_active, indexing="ij")
        D = xp.where(xp.abs(Fi - Fj) <= 0.5 * df, 3.0, 6.0)

        Fk_ij = Fi + Fj  # (N_a, N_a)
        Fi_f32 = Fi.astype(xp.float64)
        Fj_f32 = Fj.astype(xp.float64)
        Gi_f32 = Gi.astype(xp.float64)
        Gj_f32 = Gj.astype(xp.float64)
        alpha_i_f32 = alpha_i.astype(xp.float64)
        alpha_j_f32 = alpha_j.astype(xp.float64)
        D_f32 = D.astype(xp.float64)
        Fk_ij_f32 = (Fi_f32 + Fj_f32).astype(xp.float64)

        pump_grid_f32 = f_grid.astype(xp.float64)
        pump_G_f32 = G_tx.astype(xp.float64)
        alpha_grid_f32 = alpha_grid.astype(xp.float64)

        N_a = Fi_f32.shape[0]
        N_pairs = N_a * N_a
        chunk_size = 10 if N_pairs * N_a > 50_000_000 else 50

        # Extract fiber scalars for GPU phase-mismatch (avoids calling Fiber methods on GPU).
        D_c = float(fiber._D_c)
        D_slope = float(fiber._D_slope)
        gamma = float(fiber.gamma)
        rayleigh_coeff = float(fiber.rayleigh_coeff)
        L_fiber = float(fiber.L)
        L_values_arr = xp.asarray(L_arr, dtype=xp.float64).reshape(-1) if L_arr is not None else None

        def _gpu_batch_fwm(chunk_start: int, chunk_end: int) -> xp.ndarray:
            """Compute FWM for output frequencies [chunk_start, chunk_end) entirely on GPU."""
            f1_chunk = f_grid[chunk_start:chunk_end]  # (n_chunk,) on GPU
            n_chunk = chunk_end - chunk_start
            alpha1_chunk = alpha_grid_f32[chunk_start:chunk_end]  # (n_chunk,) on GPU

            # ---- Batch interpolation: Gk and alpha_k for all f1 in one GPU pass ----
            # Fk: (n_chunk, N_a, N_a) = f2 values for each output freq × each pump pair
            Fk = Fk_ij_f32[:, :, xp.newaxis] - f1_chunk[xp.newaxis, xp.newaxis, :]  # noqa: E501
            Fk_T = Fk.transpose(2, 0, 1).reshape(n_chunk, -1)  # (n_chunk, N_pairs)

            sorted_idx = xp.clip(
                xp.searchsorted(pump_grid_f32, Fk_T), 1, len(pump_grid_f32) - 1
            )
            x0 = pump_grid_f32[sorted_idx - 1]
            x1 = pump_grid_f32[sorted_idx]
            y0 = pump_G_f32[sorted_idx - 1]
            y1 = pump_G_f32[sorted_idx]
            t = xp.clip((Fk_T - x0) / (x1 - x0), 0.0, 1.0)
            Gk_T = (y0 + (y1 - y0) * t).astype(xp.float64)  # (n_chunk, N_pairs)
            a0 = alpha_grid_f32[sorted_idx - 1]
            a1_arr = alpha_grid_f32[sorted_idx]
            alpha_k_T = (a0 + (a1_arr - a0) * t).astype(xp.float64)  # (n_chunk, N_pairs)

            # Reshape to (n_chunk, N_a, N_a) for batch FWM
            Gk = Gk_T.reshape(n_chunk, N_a, N_a)  # (n_chunk, N_a, N_a)
            alpha_k = alpha_k_T.reshape(n_chunk, N_a, N_a)  # (n_chunk, N_a, N_a)

            # Compute delta_beta batch on GPU: f2 = Fk_ij - f1_chunk[j]
            # Fk_ij_f32: (N_a, N_a), f1_chunk: (n_chunk,)
            # delta_beta: (n_chunk, N_a, N_a) = batch_phase_mismatch(f2, f3, f4)
            f2_3d = (Fk_ij_f32[:, :, xp.newaxis] - f1_chunk[xp.newaxis, xp.newaxis, :]).transpose(
                2, 0, 1
            )  # (n_chunk, N_a, N_a)
            # f3_3d and f4_3d: same Fi/Fj broadcast over all f1_chunk values
            f3_3d = xp.broadcast_to(Fi_f32[xp.newaxis, :, :], (n_chunk, N_a, N_a))
            f4_3d = xp.broadcast_to(Fj_f32[xp.newaxis, :, :], (n_chunk, N_a, N_a))
            delta_beta_3d = _gpu_phase_mismatch(f2_3d, f3_3d, f4_3d, D_c, D_slope)

            # Mask: Gk > 0 (valid pump pair)
            valid_mask = Gk > 0.0  # (n_chunk, N_a, N_a)

            # delta_alpha: (n_chunk, N_a, N_a)
            da_3d = (
                alpha_k
                + xp.broadcast_to(alpha_i_f32, (n_chunk, N_a, N_a))
                + xp.broadcast_to(alpha_j_f32, (n_chunk, N_a, N_a))
                - alpha1_chunk[:, xp.newaxis, xp.newaxis]
            )

            if L_values_arr is None:
                # ---- Scalar L path: vectorized over all f1 in the chunk ----
                # Broadcast Gi, Gj to (n_chunk, N_a, N_a) to match pump-pair indexing
                Gi_3d = xp.broadcast_to(Gi_f32[xp.newaxis, :, :], (n_chunk, N_a, N_a))
                Gj_3d = xp.broadcast_to(Gj_f32[xp.newaxis, :, :], (n_chunk, N_a, N_a))
                D2_3d = xp.broadcast_to((D_f32 ** 2)[xp.newaxis, :, :], (n_chunk, N_a, N_a))

                eta = _gpu_fwm_efficiency(da_3d, delta_beta_3d, L_fiber)
                eta = xp.where(valid_mask, eta, 0.0)

                if direction == "forward":
                    exp_alpha1 = xp.exp(-alpha1_chunk * L_fiber)  # (n_chunk,)
                    integrand = D2_3d * eta * Gi_3d * Gj_3d * Gk
                    # Sum over pump-pair axes (1, 2), result: (n_chunk,)
                    integral = xp.sum(integrand, axis=(1, 2)) * df * df
                    out = (gamma ** 2 / 9.0) * 4.0 * exp_alpha1 * integral
                else:
                    # Backward: alpha1_chunk stays 1D so float(alpha1) works
                    F_L = _gpu_F_antiderivative(
                        xp.full_like(da_3d, L_fiber), 0.0, alpha1_chunk, da_3d, delta_beta_3d, L_fiber
                    )
                    F_0 = _gpu_F_antiderivative(
                        xp.zeros_like(da_3d), 0.0, alpha1_chunk, da_3d, delta_beta_3d, L_fiber
                    )
                    diff = xp.where(valid_mask, F_L - F_0, 0.0)
                    integrand = D2_3d * Gi_3d * Gj_3d * Gk * diff
                    integral = xp.sum(integrand, axis=(1, 2)) * df * df
                    out = rayleigh_coeff * (gamma ** 2 / 9.0) * integral
                return out

            else:
                # ---- Vectorized L path: 4D broadcast over L ----
                N_L = L_values_arr.shape[0]
                out = xp.zeros((n_chunk, N_L), dtype=xp.float64)

                # Broadcast pump-pair scalars to (n_chunk, N_a, N_a, N_L)
                Gi_4d = xp.broadcast_to(Gi_f32[xp.newaxis, :, :, xp.newaxis], (n_chunk, N_a, N_a, N_L))
                Gj_4d = xp.broadcast_to(Gj_f32[xp.newaxis, :, :, xp.newaxis], (n_chunk, N_a, N_a, N_L))
                D_4d = xp.broadcast_to(D_f32[xp.newaxis, :, :, xp.newaxis], (n_chunk, N_a, N_a, N_L))
                da_4d = da_3d[:, :, :, xp.newaxis]  # (n_chunk, N_a, N_a, 1)
                db_4d = delta_beta_3d[:, :, :, xp.newaxis]
                Gk_4d = Gk[:, :, :, xp.newaxis]
                alpha1_4d = alpha1_chunk[:, xp.newaxis, xp.newaxis, xp.newaxis]

                if direction == "forward":
                    eta_4d = _gpu_fwm_efficiency_vec(da_4d, db_4d, L_values_arr)
                    eta_4d = xp.where(valid_mask[:, :, :, xp.newaxis], eta_4d, 0.0)
                    integrand = (D_4d ** 2) * eta_4d * Gi_4d * Gj_4d * Gk_4d
                    # valid_mask: (n_chunk, N_a, N_a) → (n_chunk, N_a, N_a, 1) broadcast to N_L
                    integral_all = xp.sum(
                        integrand * valid_mask[:, :, :, xp.newaxis], axis=(1, 2)
                    ) * df * df  # (n_chunk, N_L)
                    exp_factor = xp.exp(-alpha1_chunk[:, xp.newaxis] * L_values_arr[xp.newaxis, :])  # (n_chunk, N_L)
                    out[:] = (gamma ** 2 / 9.0) * 4.0 * exp_factor * integral_all
                else:
                    # alpha1_4d broadcasts to (n_chunk, N_a, N_a, N_L) — correct per-frequency alpha1
                    alpha1_broadcast = xp.broadcast_to(alpha1_4d, (n_chunk, N_a, N_a, N_L))
                    F_L_4d = _gpu_F_antiderivative_vec(
                        L_fiber, 0.0, alpha1_broadcast, da_4d, db_4d, L_values_arr
                    )
                    F_0_4d = _gpu_F_antiderivative_vec(
                        0.0, 0.0, alpha1_broadcast, da_4d, db_4d, L_values_arr
                    )
                    diff = F_L_4d - F_0_4d
                    diff = xp.where(valid_mask[:, :, :, xp.newaxis], diff, 0.0)
                    integrand = (D_4d ** 2) * Gi_4d * Gj_4d * Gk_4d * diff
                    integral_all = xp.sum(
                        integrand * valid_mask[:, :, :, xp.newaxis], axis=(1, 2)
                    ) * df * df
                    out[:] = rayleigh_coeff * (gamma ** 2 / 9.0) * integral_all
                return out

        # ---- Assemble GPU results in chunks, copy back to CPU ----
        if L_arr is None:
            out_cpu = np.zeros(n_f, dtype=np.float64)
            for start in range(0, n_f, chunk_size):
                end = min(start + chunk_size, n_f)
                chunk_result = _gpu_batch_fwm(start, end)
                out_cpu[start:end] = to_host(chunk_result)
            return out_cpu

        N_L = L_values_arr.shape[0]
        out_cpu = np.zeros((n_f, N_L), dtype=np.float64)
        for start in range(0, n_f, chunk_size):
            end = min(start + chunk_size, n_f)
            chunk_result = _gpu_batch_fwm(start, end)
            out_cpu[start:end, :] = to_host(chunk_result)
        return out_cpu
