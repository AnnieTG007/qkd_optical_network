"""离散 SpRS（自发拉曼散射）噪声求解器。

公式来源：docs/formulas_sprs.md 第4节（离散模型）

计算量子信道受经典信道泵浦产生的前向/后向 SpRS 噪声功率。
全程向量化：量子信道 f_q shape (N_q, 1)，经典信道 f_c shape (1, N_c)，
广播得 (N_q, N_c) 矩阵，无显式 for 循环。
"""

from __future__ import annotations

import numpy as np
from scipy.constants import h, k

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import SpectrumType, WDMGrid
from qkd_sim.physical.noise.base import NoiseSolver
from qkd_sim.physical.noise.raman_data import get_raman_gain

# α相等判定阈值 [1/m]：|α₂ - α₁| < _ALPHA_EPS 时使用洛必达极限公式
_ALPHA_EPS: float = 1e-10


def _phonon_occupation(delta_f: np.ndarray, T: float) -> np.ndarray:
    """声子占据因子 n_th。

    公式 3.1.2 (formulas_sprs.md):
        n_th(Δf) = 1 / [exp(h·Δf / (k·T)) - 1]

    Δf = 0 处定义 n_th = 0（避免除零，该点的 σ 系数也为 0）。

    Parameters
    ----------
    delta_f : ndarray
        频移绝对值 |f₁ - f₂| [Hz]，形状任意
    T : float
        温度 [K]

Returns
    -------
    ndarray
        声子占据因子（无量纲），形状与 delta_f 相同
    """
    n = np.zeros_like(delta_f, dtype=np.float64)
    nonzero = delta_f > 0.0
    exponent = h * delta_f[nonzero] / (k * T)
    n[nonzero] = 1.0 / (np.exp(exponent) - 1.0)
    return n


def _raman_cross_section(
    f_q: np.ndarray,
    f_c: np.ndarray,
    g_R: np.ndarray,
    n_th: np.ndarray,
    delta_f: np.ndarray,
    noise_bandwidth: float | np.ndarray,
) -> np.ndarray:
    """拉曼横截面积 σ_{1,2}，区分 Stokes / anti-Stokes。

    这里采用的实现口径与 Raman amplifier 文献中的 spontaneous-noise 项一致：
    ``g_R`` 和 ``n_th`` 由泵浦-信号频移 ``delta_f = |f_q - f_c|`` 决定，
    而乘法中的频率项对应接收端噪声带宽 ``noise_bandwidth``，不是 Raman 频移本身。

    公式 3.1.4 (formulas_sprs.md, 修订后):
        Stokes (f_c > f_q，泵浦频率高于量子信道，散射光为低频):
            σ = 2·h·f_q · g_R(f_c, f_q) · (1 + n_th) · B_noise

        anti-Stokes (f_c < f_q，泵浦频率低于量子信道，散射光为高频):
            σ = 2·h·f_q · g_R(f_q, f_c) · n_th · (f_q/f_c) · B_noise

    ``delta_f`` 参数仅用于插值 ``g_R`` 和计算声子占据因子 ``n_th``。

    Parameters
    ----------
    f_q : ndarray, shape (N_q, 1)
        量子信道中心频率 [Hz]
    f_c : ndarray, shape (1, N_c)
        经典信道中心频率 [Hz]
    g_R : ndarray, shape (N_q, N_c)
        拉曼增益系数 [1/(W·m)]（已插值修正）
    n_th : ndarray, shape (N_q, N_c)
        声子占据因子
    delta_f : ndarray, shape (N_q, N_c)
        频移绝对值 |f_q - f_c| [Hz]（用于插值和 n_th）
    noise_bandwidth : float or ndarray
        噪声收集带宽 [Hz]。离散模型通常取量子信道带宽，连续谱绘制时
        取输出频率 bin 宽度 ``df``。

    Returns
    -------
    ndarray, shape (N_q, N_c)
        拉曼横截面积 σ_{1,2} [1/m]
    """
    # Stokes 掩码：泵浦频率 f_c > 量子信道频率 f_q
    stokes_mask = f_c > f_q  # shape (N_q, N_c)

    sigma = np.zeros_like(g_R)

    # 广播 f_q, f_c 至 (N_q, N_c) 便于掩码索引
    f_q_bc = np.broadcast_to(f_q, g_R.shape)
    f_c_bc = np.broadcast_to(f_c, g_R.shape)

    noise_bandwidth = np.broadcast_to(
        np.asarray(noise_bandwidth, dtype=np.float64),
        g_R.shape,
    )

    # Stokes 分量：频率项取接收端噪声带宽 B_noise
    sigma[stokes_mask] = (
        2.0 * h * f_q_bc[stokes_mask]
        * g_R[stokes_mask]
        * (1.0 + n_th[stokes_mask])
        * noise_bandwidth[stokes_mask]
    )

    # anti-Stokes 分量
    anti_mask = ~stokes_mask & (delta_f > 0.0)  # 排除 f_c == f_q
    sigma[anti_mask] = (
        2.0 * h * f_q_bc[anti_mask]
        * g_R[anti_mask]
        * n_th[anti_mask]
        * (f_q_bc[anti_mask] / f_c_bc[anti_mask])
        * noise_bandwidth[anti_mask]
    )

    return sigma


def _forward_propagation(
    sigma: np.ndarray,
    P_pump: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    L: float | np.ndarray,
) -> np.ndarray:
    """前向 SpRS 噪声功率（单泵浦→单量子信道贡献矩阵）。

    公式 3.1.3 (formulas_sprs.md):
        当 |α₂ - α₁| < ε：
            P_{f,2→1}(L) = P_pump · σ · exp(-α₁·L) · L

        当 |α₂ - α₁| ≥ ε：
            P_{f,2→1}(L) = P_pump · σ · exp(-α₁·L)
                           · [1 - exp(-(α₂-α₁)·L)] / (α₂ - α₁)

    Parameters
    ----------
    sigma : ndarray, shape (N_q, N_c)
        拉曼横截面积 [1/m]
    P_pump : ndarray, shape (1, N_c)
        经典信道总泵浦功率 P_{f,2} + P_{b,2} [W]
    alpha1 : ndarray, shape (N_q, 1)
        量子信道衰减系数 [1/m]
    alpha2 : ndarray, shape (1, N_c)
        经典信道衰减系数 [1/m]
    L : float or ndarray
        光纤长度 [m]。标量时返回 (N_q, N_c)；数组 (N_L,) 时返回 (N_q, N_c, N_L)

    Returns
    -------
    ndarray, shape (N_q, N_c) or (N_q, N_c, N_L)
        各泵浦信道对各量子信道的前向噪声贡献 [W]
    """
    d_alpha = alpha2 - alpha1  # shape (N_q, N_c)，广播
    L_arr = np.asarray(L, dtype=np.float64)
    scalar_L = L_arr.ndim == 0

    equal_mask = np.abs(d_alpha) < _ALPHA_EPS  # (N_q, N_c)
    if scalar_L:
        exp_m_alpha1_L = np.exp(-alpha1 * L_arr)  # (N_q, 1)
        integral = np.where(
            equal_mask,
            L_arr,  # scalar
            (1.0 - np.exp(-d_alpha * L_arr)) / np.where(equal_mask, 1.0, d_alpha),
        )
        return P_pump * sigma * exp_m_alpha1_L * integral  # (N_q, N_c)

    # L_arr path: broadcast sigma and P_pump to (N_q, N_c, 1) for (N_q, N_c, N_L) output
    L_arr = L_arr.reshape(1, 1, -1)  # (1, 1, N_L)
    exp_m_alpha1_L = np.exp(-alpha1[..., np.newaxis] * L_arr)  # (N_q, 1, N_L)
    integral = np.where(
        equal_mask[..., np.newaxis],
        L_arr,  # (1, 1, N_L) → broadcasts
        (1.0 - np.exp(-d_alpha[..., np.newaxis] * L_arr)) / np.where(equal_mask[..., np.newaxis], 1.0, d_alpha[..., np.newaxis]),
    )
    return (P_pump[..., np.newaxis] * sigma[..., np.newaxis]
            * exp_m_alpha1_L * integral)  # (N_q, N_c, N_L)


def _backward_propagation(
    sigma: np.ndarray,
    P_pump: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    L: float | np.ndarray,
) -> np.ndarray:
    """后向 SpRS 噪声功率（单泵浦→单量子信道贡献矩阵）。

    公式 3.1.7 (formulas_sprs.md):
        P_{b,2→1}(L) = P_pump · σ · [1 - exp(-(α₁+α₂)·L)] / (α₁ + α₂)

    Parameters
    ----------
    sigma : ndarray, shape (N_q, N_c)
        拉曼横截面积 [1/m]
    P_pump : ndarray, shape (1, N_c)
        经典信道总泵浦功率 [W]
    alpha1 : ndarray, shape (N_q, 1)
        量子信道衰减系数 [1/m]
    alpha2 : ndarray, shape (1, N_c)
        经典信道衰减系数 [1/m]
    L : float or ndarray
        光纤长度 [m]。标量时返回 (N_q, N_c)；数组 (N_L,) 时返回 (N_q, N_c, N_L)

    Returns
    -------
    ndarray, shape (N_q, N_c) or (N_q, N_c, N_L)
        各泵浦信道对各量子信道的后向噪声贡献 [W]
    """
    sum_alpha = alpha1 + alpha2  # shape (N_q, N_c)
    L_arr = np.asarray(L, dtype=np.float64)
    if L_arr.ndim == 0:
        return P_pump * sigma * (1.0 - np.exp(-sum_alpha * L_arr)) / sum_alpha
    # L_arr path: broadcast over new trailing dimension
    L_arr = L_arr.reshape(1, 1, -1)  # (1, 1, N_L)
    return (P_pump[..., np.newaxis] * sigma[..., np.newaxis]
            * (1.0 - np.exp(-sum_alpha[..., np.newaxis] * L_arr))
            / sum_alpha[..., np.newaxis])  # (N_q, N_c, N_L)


class DiscreteSPRSSolver(NoiseSolver):
    """离散 SpRS 噪声求解器。

    计算每个量子信道受所有经典信道泵浦产生的前向/后向
    自发拉曼散射噪声功率。

    公式来源：docs/formulas_sprs.md 第 4 节（离散模型）
    推导参考：formulas_sprs.md 第 3.1.2 ~ 3.1.8 节

    Examples
    --------
    >>> from qkd_sim.physical.noise.sprs_solver import DiscreteSPRSSolver
    >>> solver = DiscreteSPRSSolver()
    >>> P_fwd = solver.compute_forward(fiber, wdm_grid)  # shape (N_q,)
    >>> P_bwd = solver.compute_backward(fiber, wdm_grid)  # shape (N_q,)
    """

    def __init__(self, noise_bandwidth_hz: float | None = 20e9) -> None:
        """离散 SpRS 求解器。

        Parameters
        ----------
        noise_bandwidth_hz : float | None
            噪声收集带宽 [Hz]。用于拉曼截面公式 σ ∝ B_noise。
            默认 20 GHz。不为 None 时覆盖量子信道的 B_s。
            传入 None 则回退到每个量子信道的 B_s。
        """
        self._noise_bandwidth_hz = noise_bandwidth_hz
        # Continuous SpRS caches sigma only for active pump frequency columns;
        # this avoids N_f^2 memory at fine frequency resolution.
        self._sigma_cache: "dict[tuple, np.ndarray]" = {}
        self._sigma_cache_max: int = 2

    def _prepare(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """提取并整形信道参数，计算公共中间量。

        Returns
        -------
        f_q : ndarray, shape (N_q, 1)
        f_c : ndarray, shape (1, N_c)
        P_pump : ndarray, shape (1, N_c)  — P_{f,2} + P_{b,2}
        alpha1 : ndarray, shape (N_q, 1)
        alpha2 : ndarray, shape (1, N_c)
        """
        q_chs = wdm_grid.get_quantum_channels()
        c_chs = wdm_grid.get_classical_channels()

        assert len(q_chs) > 0, "WDMGrid 中无量子信道"
        assert len(c_chs) > 0, "WDMGrid 中无经典信道（泵浦）"

        f_q = np.array([ch.f_center for ch in q_chs], dtype=np.float64).reshape(-1, 1)
        f_c = np.array([ch.f_center for ch in c_chs], dtype=np.float64).reshape(1, -1)
        P_ch = np.array([ch.power for ch in c_chs], dtype=np.float64).reshape(1, -1)

        # 总泵浦功率 = 前向 + 后向（离散模型均视为发射端功率）
        P_pump = P_ch  # P_{f,2} + P_{b,2} = P_ch（前/后向之和）

        # C 波段近似：衰减与频率无关
        alpha1 = np.full((len(q_chs), 1), fiber.alpha, dtype=np.float64)
        alpha2 = np.full((1, len(c_chs)), fiber.alpha, dtype=np.float64)

        return f_q, f_c, P_pump, alpha1, alpha2

    def _compute_sigma(
        self,
        f_q: np.ndarray,
        f_c: np.ndarray,
        fiber: Fiber,
        wdm_grid: WDMGrid,
    ) -> np.ndarray:
        """计算拉曼横截面积矩阵 σ_{1,2}，shape (N_q, N_c)。"""
        delta_f = np.abs(f_q - f_c)  # shape (N_q, N_c)

        # 拉曼增益系数：g_R(|Δf|, f_pump=f_c)
        # f_c 广播至 (N_q, N_c) 用于修正
        # f_c shape (1, N_c) 由 get_raman_gain 内部 asarray 广播至 (N_q, N_c)，无需 copy
        g_R = get_raman_gain(
            delta_f=delta_f,
            f_pump=f_c,
            A_eff=fiber.A_eff,
        )  # shape (N_q, N_c)

        n_th = _phonon_occupation(delta_f, fiber.T_kelvin)  # shape (N_q, N_c)

        # B_noise：显式配置优先；否则取量子信道带宽 B_s。
        if self._noise_bandwidth_hz is not None:
            q_chs = wdm_grid.get_quantum_channels()
            noise_bandwidth = np.full(
                (len(q_chs), 1),
                self._noise_bandwidth_hz,
                dtype=np.float64,
            )
        else:
            q_chs = wdm_grid.get_quantum_channels()
            noise_bandwidth = np.array(
                [ch.B_s for ch in q_chs],
                dtype=np.float64,
            ).reshape(-1, 1)

        sigma = _raman_cross_section(
            f_q,
            f_c,
            g_R,
            n_th,
            delta_f,
            noise_bandwidth,
        )
        return sigma

    def compute_forward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算各量子信道在光纤接收端（z=L）的前向 SpRS 噪声功率。

        公式 3.1.3 + 3.1.6 (formulas_sprs.md)

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid

        Returns
        -------
        ndarray, shape (N_q,)
            前向 SpRS 噪声功率 [W]
        """
        f_q, f_c, P_pump, alpha1, alpha2 = self._prepare(fiber, wdm_grid)
        sigma = self._compute_sigma(f_q, f_c, fiber, wdm_grid)

        # shape (N_q, N_c)
        P_fwd_matrix = _forward_propagation(sigma, P_pump, alpha1, alpha2, fiber.L)

        # 对所有泵浦信道求和（公式 3.1.6）
        P_fwd = P_fwd_matrix.sum(axis=1)  # shape (N_q,)
        assert P_fwd.shape == (f_q.shape[0],)
        return P_fwd

    def compute_backward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算各量子信道在光纤发射端（z=0）的后向 SpRS 噪声功率。

        公式 3.1.7 + 3.1.8 (formulas_sprs.md)

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid

        Returns
        -------
        ndarray, shape (N_q,)
            后向 SpRS 噪声功率 [W]
        """
        f_q, f_c, P_pump, alpha1, alpha2 = self._prepare(fiber, wdm_grid)
        sigma = self._compute_sigma(f_q, f_c, fiber, wdm_grid)

        # shape (N_q, N_c)
        P_bwd_matrix = _backward_propagation(sigma, P_pump, alpha1, alpha2, fiber.L)

        # 对所有泵浦信道求和（公式 3.1.8）
        P_bwd = P_bwd_matrix.sum(axis=1)  # shape (N_q,)
        assert P_bwd.shape == (f_q.shape[0],)
        return P_bwd

    def compute_forward_l_array(
        self, fiber: Fiber, wdm_grid: WDMGrid, L_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute forward SpRS noise for multiple fiber lengths.

        Computes sigma (L-independent) once, then propagates for all lengths
        in a single vectorized call.

        Returns
        -------
        ndarray, shape (N_q, N_L)
        """
        f_q, f_c, P_pump, alpha1, alpha2 = self._prepare(fiber, wdm_grid)
        sigma = self._compute_sigma(f_q, f_c, fiber, wdm_grid)
        P_fwd_matrix = _forward_propagation(sigma, P_pump, alpha1, alpha2, L_arr)
        return P_fwd_matrix.sum(axis=1)  # (N_q, N_L)

    def compute_backward_l_array(
        self, fiber: Fiber, wdm_grid: WDMGrid, L_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute backward SpRS noise for multiple fiber lengths.

        Computes sigma (L-independent) once, then propagates for all lengths
        in a single vectorized call.

        Returns
        -------
        ndarray, shape (N_q, N_L)
        """
        f_q, f_c, P_pump, alpha1, alpha2 = self._prepare(fiber, wdm_grid)
        sigma = self._compute_sigma(f_q, f_c, fiber, wdm_grid)
        P_bwd_matrix = _backward_propagation(sigma, P_pump, alpha1, alpha2, L_arr)
        return P_bwd_matrix.sum(axis=1)  # (N_q, N_L)

    # --- 连续模型方法 -------------------------------------------------------

    @staticmethod
    def _validate_frequency_grid(f_grid: np.ndarray) -> float:
        """验证 1D 积分网格并返回频率步长。

        公式 3.2.2 和 3.2.4 (formulas_sprs.md) 使用均匀网格黎曼和计算。
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
    def _build_classical_psd_matrix(
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        df: float,
    ) -> np.ndarray:
        """在 f_grid 上构建经典信道 PSD 矩阵。

        SINGLE_FREQ 信道：使用 delta 近似 G=P/df 放在最近格点，
        使得 sum(G*df) ≈ P，用于离散/连续交叉验证极限。
        """
        classical_channels = wdm_grid.get_classical_channels()
        assert len(classical_channels) > 0, "WDMGrid 中无经典信道（泵浦）"

        psd = np.zeros((len(classical_channels), f_grid.size), dtype=np.float64)
        for i, ch in enumerate(classical_channels):
            if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
                idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
                psd[i, idx] = ch.power / df
            else:
                psd[i, :] = ch.get_psd(f_grid)
        return psd

    @staticmethod
    def _integrate_psd_per_channel(
        psd: np.ndarray,
        f_grid: np.ndarray,
        df: float,
        q_chs: list,
    ) -> np.ndarray:
        """对 PSD 在量子信道带宽内积分得到 per-channel 功率。

        使用前缀和（prefix sum）实现 O(N_f + N_q) 复杂度。
        f_grid 必须等间距，索引由公式直接计算无需二分查找。

        Parameters
        ----------
        psd : ndarray, shape (N_f,) or (N_f, N_L)
            PSD G_sprs(f) [W/Hz]
        f_grid : ndarray, shape (N_f,)
            评估频率网格 [Hz]，等间距
        df : float
            频率步长 [Hz]
        q_chs : list[WDMChannel]
            量子信道列表

        Returns
        -------
        ndarray, shape (N_q,) or (N_q, N_L)
            每个量子信道的积分噪声功率 [W]
        """
        n_q = len(q_chs)
        f0 = float(f_grid[0])
        n_f = f_grid.size
        psd_2d = psd.ndim == 2

        prefix = np.cumsum(psd, axis=0) * df
        P = np.zeros((n_q, psd.shape[1]) if psd_2d else n_q, dtype=np.float64)

        for i, ch in enumerate(q_chs):
            f_lo = ch.f_center - ch.B_s / 2.0
            f_hi = ch.f_center + ch.B_s / 2.0
            idx_lo = max(0, min(n_f - 1, int(np.round((f_lo - f0) / df))))
            idx_hi = max(0, min(n_f - 1, int(np.round((f_hi - f0) / df))))
            if idx_hi > idx_lo:
                P[i] = prefix[idx_hi] - prefix[idx_lo]
            else:
                idx = max(0, min(n_f - 1, int(np.round((ch.f_center - f0) / df))))
                P[i] = psd[idx] * df if not psd_2d else psd[idx] * df

        return P

    def compute_forward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续前向 SpRS 噪声功率 [公式 3.2.1 + 3.2.2]。

        复用 ``compute_sprs_spectrum_conti`` 的 PSD 结果，
        在每个量子信道带宽内积分得到 per-channel 功率。
        当所有经典信道均为 SINGLE_FREQ 时，退化为离散模型（交叉验证极限）。
        """
        c_chs = wdm_grid.get_classical_channels()
        if len(c_chs) > 0 and all(
            ch.spectrum_type == SpectrumType.SINGLE_FREQ for ch in c_chs
        ):
            return self.compute_forward(fiber, wdm_grid)

        df = self._validate_frequency_grid(f_grid)
        f_grid = np.asarray(f_grid, dtype=np.float64)
        q_chs = wdm_grid.get_quantum_channels()
        assert len(q_chs) > 0, "WDMGrid 中无量子信道"

        psd = self.compute_sprs_spectrum_conti(
            fiber, wdm_grid, f_grid, direction="forward"
        )
        return self._integrate_psd_per_channel(psd, f_grid, df, q_chs)

    def compute_backward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续后向 SpRS 噪声功率 [公式 3.2.3 + 3.2.4]。

        复用 ``compute_sprs_spectrum_conti`` 的 PSD 结果，
        在每个量子信道带宽内积分得到 per-channel 功率。
        当所有经典信道均为 SINGLE_FREQ 时，退化为离散模型（交叉验证极限）。
        """
        c_chs = wdm_grid.get_classical_channels()
        if len(c_chs) > 0 and all(
            ch.spectrum_type == SpectrumType.SINGLE_FREQ for ch in c_chs
        ):
            return self.compute_backward(fiber, wdm_grid)

        df = self._validate_frequency_grid(f_grid)
        f_grid = np.asarray(f_grid, dtype=np.float64)
        q_chs = wdm_grid.get_quantum_channels()
        assert len(q_chs) > 0, "WDMGrid 中无量子信道"

        psd = self.compute_sprs_spectrum_conti(
            fiber, wdm_grid, f_grid, direction="backward"
        )
        return self._integrate_psd_per_channel(psd, f_grid, df, q_chs)

    # --- 噪声 PSD 谱计算 — 向量化版本（全频率一次计算）-----------

    def _get_sigma_2d(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        pump_freqs: np.ndarray,
        df: float,
    ) -> np.ndarray:
        """返回 (N_f, N_f) 拉曼横截面矩阵 σ_2d，跨调用复用。

        σ_2d 仅依赖 (f_grid, T_kelvin, A_eff, df)，与 G_pump、L、direction、
        model_key 无关。同一次 Dash 启动里多个 model_key 都会复用同一份。

        Cache key 用 f_grid 的二进制 sha1 + 标量参数；LRU 由
        ``self._sigma_cache_max`` 控制。
        """
        # Actual shape is (N_f, N_active_pump); pump_freqs is part of cache key.
        import hashlib

        f_arr = np.ascontiguousarray(f_grid, dtype=np.float64)
        pump_arr = np.ascontiguousarray(pump_freqs, dtype=np.float64)
        key = (
            hashlib.sha1(f_arr.tobytes()).hexdigest(),
            hashlib.sha1(pump_arr.tobytes()).hexdigest(),
            float(fiber.T_kelvin),
            float(fiber.A_eff),
            float(df),
        )
        cached = self._sigma_cache.get(key)
        if cached is not None:
            return cached

        f1_2d = f_arr.reshape(-1, 1)
        f2_2d = pump_arr.reshape(1, -1)
        delta_f_2d = np.abs(f1_2d - f2_2d)

        g_R_2d = get_raman_gain(
            delta_f=delta_f_2d,
            f_pump=f2_2d,
            A_eff=fiber.A_eff,
        )
        n_th_2d = _phonon_occupation(delta_f_2d, fiber.T_kelvin)
        sigma_2d = _raman_cross_section(
            f_q=f1_2d,
            f_c=f2_2d,
            g_R=g_R_2d,
            n_th=n_th_2d,
            delta_f=delta_f_2d,
            noise_bandwidth=df,
        )

        if len(self._sigma_cache) >= self._sigma_cache_max:
            # LRU 退化版：弹出最早 entry（dict 在 Python 3.7+ 保插入顺序）
            self._sigma_cache.pop(next(iter(self._sigma_cache)))
        self._sigma_cache[key] = sigma_2d
        return sigma_2d

    # ---------------------------------------------------------------------
    # 连续谱 PSD 计算 — 统一实现（CPU / GPU，单 L / 多 L，单 / 双向）
    # ---------------------------------------------------------------------

    def _compute_sprs_psd_batch_cpu(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        G_pump: np.ndarray,
        df: float,
        L_values: np.ndarray,
        do_fwd: bool,
        do_bwd: bool,
    ):
        """CPU vectorized SpRS PSD batch — unified over L and direction.

        Parameters
        ----------
        fiber : Fiber
        f_grid : ndarray, shape (N_f,)
        G_pump : ndarray, shape (N_f,)
        df : float
        L_values : ndarray, shape (N_L,)
        do_fwd : bool
        do_bwd : bool

        Returns
        -------
        If do_fwd and do_bwd: (fwd, bwd) each (N_f, N_L)
        Else: ndarray, shape (N_f, N_L)
        """
        active_idx = np.flatnonzero(np.asarray(G_pump, dtype=np.float64) > 0.0)
        pump_freqs = f_grid[active_idx]
        pump_psd = np.asarray(G_pump, dtype=np.float64)[active_idx]
        sigma_2d = self._get_sigma_2d(fiber, f_grid, pump_freqs, df)  # (N_f, N_p)

        alpha_grid = fiber.get_loss_at_freq(f_grid)
        alpha1_2d = alpha_grid.reshape(-1, 1)             # (N_f, 1)
        alpha2_2d = alpha_grid[active_idx].reshape(1, -1)  # (1, N_p)
        base_2d = pump_psd.reshape(1, -1) * sigma_2d       # (N_f, N_p)

        n_l = L_values.size
        n_f = f_grid.size
        n_p = active_idx.size

        # Memory-budgeted batch size: limit intermediate (N_f, N_p, batch) usage.
        _MAX_SPRS_TMP_MB = 1024
        _bytes_per_slice = n_f * n_p * 8
        max_l_batch = max(1, int((_MAX_SPRS_TMP_MB * 1024 * 1024) / max(_bytes_per_slice, 1)))

        def _forward_batch(start: int, stop: int) -> np.ndarray:
            L_b = L_values[start:stop].reshape(1, 1, -1)
            d_alpha = alpha2_2d - alpha1_2d
            equal_mask = np.abs(d_alpha) < _ALPHA_EPS
            d_alpha_3d = d_alpha[:, :, np.newaxis]
            equal_3d = equal_mask[:, :, np.newaxis]
            exp_m_alpha1_L = np.exp(-alpha1_2d[:, :, np.newaxis] * L_b)
            integral = np.where(
                equal_3d,
                L_b,
                (1.0 - np.exp(-d_alpha_3d * L_b)) / np.where(equal_3d, 1.0, d_alpha_3d),
            )
            batch_t = base_2d[:, :, np.newaxis] * exp_m_alpha1_L * integral
            out_bin_power = np.sum(batch_t, axis=1) * df
            return out_bin_power / df  # (N_f, batch) [W/Hz]

        def _backward_batch(start: int, stop: int) -> np.ndarray:
            L_b = L_values[start:stop].reshape(1, 1, -1)
            sum_alpha = alpha1_2d + alpha2_2d
            sum_alpha_3d = sum_alpha[:, :, np.newaxis]
            batch_t = (
                base_2d[:, :, np.newaxis]
                * (1.0 - np.exp(-sum_alpha_3d * L_b)) / sum_alpha_3d
            )
            out_bin_power = np.sum(batch_t, axis=1) * df
            return out_bin_power / df

        fwd = np.zeros((n_f, n_l), dtype=np.float64) if do_fwd else None
        bwd = np.zeros((n_f, n_l), dtype=np.float64) if do_bwd else None

        for start in range(0, n_l, max_l_batch):
            stop = min(start + max_l_batch, n_l)
            if do_fwd:
                fwd[:, start:stop] = _forward_batch(start, stop)
            if do_bwd:
                bwd[:, start:stop] = _backward_batch(start, stop)

        if do_fwd and do_bwd:
            return fwd, bwd
        return fwd if do_fwd else bwd

    def _compute_sprs_psd_batch_gpu(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        G_pump: np.ndarray,
        df: float,
        L_values: np.ndarray,
        do_fwd: bool,
        do_bwd: bool,
    ):
        """GPU-accelerated SpRS PSD batch — unified over L and direction.

        Raman gain / phonon / cross-section computed on CPU (scipy.interpolate
        does not support CuPy), then sigma_2d is transferred to GPU for the
        propagation integral.
        """
        from qkd_sim.utils.gpu_utils import get_gpu_module as _get_gpu_module
        xp, _ = _get_gpu_module()

        active_idx = np.flatnonzero(np.asarray(G_pump, dtype=np.float64) > 0.0)
        pump_freqs = f_grid[active_idx]
        pump_psd = np.asarray(G_pump, dtype=np.float64)[active_idx]
        sigma_2d = self._get_sigma_2d(fiber, f_grid, pump_freqs, df)  # np (N_f, N_p)
        n_f = f_grid.size
        n_p = active_idx.size
        n_l = L_values.size

        sigma_gpu = xp.asarray(sigma_2d)
        alpha_np = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)
        alpha1_gpu = xp.asarray(alpha_np).reshape(-1, 1, 1)   # (N_f, 1, 1)
        alpha2_gpu = xp.asarray(alpha_np[active_idx]).reshape(1, -1, 1)  # (1, N_p, 1)
        base_gpu = xp.asarray(pump_psd, dtype=xp.float64).reshape(1, -1) * sigma_gpu  # (N_f, N_p)
        L_gpu = xp.asarray(L_values, dtype=xp.float64).reshape(1, 1, -1)            # (1, 1, N_L)

        # Memory-aware batching
        try:
            free_bytes, _total = xp.cuda.runtime.memGetInfo()
            budget = int(free_bytes * 0.85)
        except Exception:
            budget = 1024 * 1024 * 1024  # 1 GB fallback
        _fixed_slices = 4   # sigma + base + d_alpha + safe_d_alpha
        _per_batch_slices = 3  # exp + integral + batch_t (forward worst case)
        budget_slices = budget // max(n_f * n_p * 8, 1)
        max_l_batch = max(1, min(
            n_l,
            (budget_slices - _fixed_slices) // max(_per_batch_slices, 1),
        ))

        def _forward_gpu_chunk(start: int, stop: int):
            L_b = L_gpu[:, :, start:stop]
            d_alpha = alpha2_gpu[:, :, 0:1] - alpha1_gpu[:, :, 0:1]
            equal_mask = xp.abs(d_alpha) < _ALPHA_EPS
            safe_d_alpha = xp.where(equal_mask, xp.ones_like(d_alpha), d_alpha)
            exp_m_alpha1_L = xp.exp(-alpha1_gpu[:, :, 0:1] * L_b)
            integral = xp.where(
                equal_mask,
                L_b,
                (1.0 - xp.exp(-d_alpha * L_b)) / safe_d_alpha,
            )
            batch_t = base_gpu[:, :, xp.newaxis] * exp_m_alpha1_L * integral
            out_bin_power = xp.sum(batch_t, axis=1) * df
            return out_bin_power / df  # (N_f, batch) [W/Hz]

        def _backward_gpu_chunk(start: int, stop: int):
            L_b = L_gpu[:, :, start:stop]
            sum_alpha = alpha1_gpu[:, :, 0:1] + alpha2_gpu[:, :, 0:1]
            batch_t = (
                base_gpu[:, :, xp.newaxis]
                * (1.0 - xp.exp(-sum_alpha * L_b)) / sum_alpha
            )
            out_bin_power = xp.sum(batch_t, axis=1) * df
            return out_bin_power / df

        fwd = xp.zeros((n_f, n_l), dtype=xp.float64) if do_fwd else None
        bwd = xp.zeros((n_f, n_l), dtype=xp.float64) if do_bwd else None

        for start in range(0, n_l, max_l_batch):
            stop = min(start + max_l_batch, n_l)
            if do_fwd:
                fwd[:, start:stop] = _forward_gpu_chunk(start, stop)
            if do_bwd:
                bwd[:, start:stop] = _backward_gpu_chunk(start, stop)

        if do_fwd and do_bwd:
            return np.asarray(fwd.get()), np.asarray(bwd.get())
        result = fwd if do_fwd else bwd
        return np.asarray(result.get())

    def _compute_sprs_spectrum_conti_impl(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str,
        L_arr: np.ndarray | None,
        use_gpu: bool,
    ):
        """Unified SpRS PSD computation (single-L, multi-L, single/both directions).

        direction: "forward" | "backward" | "both"
        """
        do_fwd = direction in ("forward", "both")
        do_bwd = direction in ("backward", "both")

        f_grid = np.asarray(f_grid, dtype=np.float64)
        df = self._validate_frequency_grid(f_grid)

        G_classical = self._build_classical_psd_matrix(wdm_grid, f_grid, df)
        G_pump = G_classical.sum(axis=0)
        L_values = np.array([fiber.L]) if L_arr is None else np.asarray(L_arr, dtype=np.float64).reshape(-1)

        n_f = f_grid.size
        n_l = L_values.size

        def _zeros(shape):
            return np.zeros(shape, dtype=np.float64)

        if not np.any(G_pump > 0.0):
            if do_fwd and do_bwd:
                shape = (n_f, n_l) if n_l > 1 else (n_f,)
                z = _zeros(shape)
                return z, z.copy() if L_arr is not None else (z.copy(), z.copy())
            shape = (n_f, n_l)
            return _zeros(shape).squeeze() if L_arr is None else _zeros(shape)

        if use_gpu:
            result = self._compute_sprs_psd_batch_gpu(
                fiber, f_grid, G_pump, df, L_values, do_fwd, do_bwd,
            )
        else:
            result = self._compute_sprs_psd_batch_cpu(
                fiber, f_grid, G_pump, df, L_values, do_fwd, do_bwd,
            )

        # Squeeze single-L dimension for backward-compatible output shape
        if L_arr is None:
            if do_fwd and do_bwd:
                return result[0][:, 0], result[1][:, 0]  # (N_f,), (N_f,)
            return result[:, 0]  # (N_f,)
        return result

    def compute_sprs_spectrum_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str = "forward",
        L_arr: np.ndarray | None = None,
    ):
        """计算 SpRS 噪声 PSD G_sprs(f) [W/Hz]。

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid
        f_grid : ndarray
            输出频率网格 [Hz]
        direction : {"forward", "backward", "both"}
            ``"both"`` 返回 ``(fwd, bwd)`` tuple。
        L_arr : ndarray or None
            光纤长度数组 [m]。None 时使用 ``fiber.L``。

        Returns
        -------
        ndarray, shape (N_f,) or (N_f, N_L)
            direction="both" 时返回 ``(fwd, bwd)`` 各 shape (N_f,) or (N_f, N_L)。
        """
        from qkd_sim.utils.gpu_utils import get_gpu_module as _get_gpu_module
        _, is_gpu = _get_gpu_module()
        return self._compute_sprs_spectrum_conti_impl(
            fiber, wdm_grid, f_grid,
            direction=direction, L_arr=L_arr, use_gpu=is_gpu,
        )

    def compute_sprs_spectrum_conti_l_array(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        L_arr: np.ndarray,
        direction: str = "forward",
    ) -> np.ndarray:
        """Compute continuous SpRS spectrum for multiple fiber lengths.

        Thin wrapper around ``compute_sprs_spectrum_conti(..., L_arr=L_arr)``.
        """
        return self.compute_sprs_spectrum_conti(
            fiber, wdm_grid, f_grid, direction=direction, L_arr=L_arr,
        )
