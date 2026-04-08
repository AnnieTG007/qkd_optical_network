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
) -> np.ndarray:
    """拉曼横截面积 σ_{1,2}，区分 Stokes / anti-Stokes。

    公式 3.1.4 (formulas_sprs.md):
        Stokes (f_c > f_q，泵浦频率高于量子信道，散射光为低频):
            σ = 2·h·f_q · g_R(f_c, f_q) · (1 + n_th) · Δf

        anti-Stokes (f_c < f_q，泵浦频率低于量子信道，散射光为高频):
            σ = 2·h·f_q · g_R(f_q, f_c) · n_th · (f_q/f_c) · Δf

    注意：σ 中的 Δf 是频率差绝对值，不是积分微元。

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
        频移绝对值 |f_q - f_c| [Hz]

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

    # Stokes 分量
    sigma[stokes_mask] = (
        2.0 * h * f_q_bc[stokes_mask]
        * g_R[stokes_mask]
        * (1.0 + n_th[stokes_mask])
        * delta_f[stokes_mask]
    )

    # anti-Stokes 分量
    anti_mask = ~stokes_mask & (delta_f > 0.0)  # 排除 f_c == f_q
    sigma[anti_mask] = (
        2.0 * h * f_q_bc[anti_mask]
        * g_R[anti_mask]
        * n_th[anti_mask]
        * (f_q_bc[anti_mask] / f_c_bc[anti_mask])
        * delta_f[anti_mask]
    )

    return sigma


def _forward_propagation(
    sigma: np.ndarray,
    P_pump: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    L: float,
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
    L : float
        光纤长度 [m]

    Returns
    -------
    ndarray, shape (N_q, N_c)
        各泵浦信道对各量子信道的前向噪声贡献 [W]
    """
    d_alpha = alpha2 - alpha1  # shape (N_q, N_c)，广播
    exp_m_alpha1_L = np.exp(-alpha1 * L)  # shape (N_q, 1)

    # 分支选择
    equal_mask = np.abs(d_alpha) < _ALPHA_EPS
    integral = np.where(
        equal_mask,
        L,  # 洛必达极限：[1 - exp(-(α₂-α₁)L)] / (α₂-α₁) → L，当 α₂→α₁
        (1.0 - np.exp(-d_alpha * L)) / np.where(equal_mask, 1.0, d_alpha),
    )
    return P_pump * sigma * exp_m_alpha1_L * integral


def _backward_propagation(
    sigma: np.ndarray,
    P_pump: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    L: float,
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
    L : float
        光纤长度 [m]

    Returns
    -------
    ndarray, shape (N_q, N_c)
        各泵浦信道对各量子信道的后向噪声贡献 [W]
    """
    sum_alpha = alpha1 + alpha2  # shape (N_q, N_c)
    return P_pump * sigma * (1.0 - np.exp(-sum_alpha * L)) / sum_alpha


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

    def __init__(self) -> None:
        pass

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

        sigma = _raman_cross_section(f_q, f_c, g_R, n_th, delta_f)
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
        sigma = self._compute_sigma(f_q, f_c, fiber)

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
        sigma = self._compute_sigma(f_q, f_c, fiber)

        # shape (N_q, N_c)
        P_bwd_matrix = _backward_propagation(sigma, P_pump, alpha1, alpha2, fiber.L)

        # 对所有泵浦信道求和（公式 3.1.8）
        P_bwd = P_bwd_matrix.sum(axis=1)  # shape (N_q,)
        assert P_bwd.shape == (f_q.shape[0],)
        return P_bwd

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

    def compute_forward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续前向 SpRS 噪声功率 [公式 3.2.1 + 3.2.2]。

        在泵浦频率 f_2 上对 G_{f,2→1} 积分。
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

        f_q = np.array([ch.f_center for ch in q_chs], dtype=np.float64).reshape(-1, 1)
        f_2 = np.asarray(f_grid, dtype=np.float64).reshape(1, -1)

        G_classical = self._build_classical_psd_matrix(wdm_grid, f_grid, df)
        G_pump = G_classical.sum(axis=0, keepdims=True)  # shape (1, N_f)

        alpha1 = np.asarray(fiber.get_loss_at_freq(f_q), dtype=np.float64).reshape(-1, 1)
        alpha2 = np.asarray(fiber.get_loss_at_freq(f_2), dtype=np.float64).reshape(1, -1)

        delta_f = np.abs(f_q - f_2)
        g_R = get_raman_gain(delta_f=delta_f, f_pump=f_2, A_eff=fiber.A_eff)
        n_th = _phonon_occupation(delta_f, fiber.T_kelvin)
        sigma = _raman_cross_section(
            f_q=f_q, f_c=f_2, g_R=g_R, n_th=n_th, delta_f=delta_f
        )

        G_fwd = _forward_propagation(
            sigma=sigma,
            P_pump=G_pump,
            alpha1=alpha1,
            alpha2=alpha2,
            L=fiber.L,
        )  # shape (N_q, N_f)

        P_fwd = np.sum(G_fwd, axis=1) * df  # formula 3.2.2
        assert P_fwd.shape == (len(q_chs),)
        return P_fwd

    def compute_backward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续后向 SpRS 噪声功率 [公式 3.2.3 + 3.2.4]。

        在泵浦频率 f_2 上对 G_{b,2→1} 积分。
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

        f_q = np.array([ch.f_center for ch in q_chs], dtype=np.float64).reshape(-1, 1)
        f_2 = np.asarray(f_grid, dtype=np.float64).reshape(1, -1)

        G_classical = self._build_classical_psd_matrix(wdm_grid, f_grid, df)
        G_pump = G_classical.sum(axis=0, keepdims=True)  # shape (1, N_f)

        alpha1 = np.asarray(fiber.get_loss_at_freq(f_q), dtype=np.float64).reshape(-1, 1)
        alpha2 = np.asarray(fiber.get_loss_at_freq(f_2), dtype=np.float64).reshape(1, -1)

        delta_f = np.abs(f_q - f_2)
        g_R = get_raman_gain(delta_f=delta_f, f_pump=f_2, A_eff=fiber.A_eff)
        n_th = _phonon_occupation(delta_f, fiber.T_kelvin)
        sigma = _raman_cross_section(
            f_q=f_q, f_c=f_2, g_R=g_R, n_th=n_th, delta_f=delta_f
        )

        G_bwd = _backward_propagation(
            sigma=sigma,
            P_pump=G_pump,
            alpha1=alpha1,
            alpha2=alpha2,
            L=fiber.L,
        )  # shape (N_q, N_f)

        P_bwd = np.sum(G_bwd, axis=1) * df  # formula 3.2.4
        assert P_bwd.shape == (len(q_chs),)
        return P_bwd

    # --- 噪声 PSD 谱计算（在 f_grid 每点计算 G_sprs(f)）-----------

    def _compute_sprs_psd_at_f1(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        G_pump: np.ndarray,
        alpha2: np.ndarray,
        f1: float,
        direction: str,
    ) -> float:
        """在单个输出频率 f1 处计算 SpRS 噪声 PSD G_sprs(f1) [W/Hz]。

        用于 compute_sprs_spectrum_conti，在 f_grid 每个频率点调用。
        """
        f_grid_arr = np.asarray(f_grid, dtype=np.float64).reshape(1, -1)
        G_pump_arr = np.asarray(G_pump, dtype=np.float64).reshape(1, -1)
        alpha2_arr = np.asarray(alpha2, dtype=np.float64).reshape(1, -1)

        f1_arr = np.full_like(f_grid_arr, f1, dtype=np.float64)
        alpha1_val = float(np.asarray(fiber.get_loss_at_freq(f1), dtype=np.float64))
        alpha1_arr = np.full_like(f_grid_arr, alpha1_val)

        delta_f = np.abs(f1_arr - f_grid_arr)
        g_R = get_raman_gain(delta_f=delta_f, f_pump=f_grid_arr, A_eff=fiber.A_eff)
        n_th = _phonon_occupation(delta_f, fiber.T_kelvin)
        sigma = _raman_cross_section(
            f_q=f1_arr, f_c=f_grid_arr, g_R=g_R, n_th=n_th, delta_f=delta_f
        )

        if direction == "forward":
            integrand = _forward_propagation(
                sigma=sigma, P_pump=G_pump_arr,
                alpha1=alpha1_arr, alpha2=alpha2_arr, L=fiber.L,
            )
        elif direction == "backward":
            integrand = _backward_propagation(
                sigma=sigma, P_pump=G_pump_arr,
                alpha1=alpha1_arr, alpha2=alpha2_arr, L=fiber.L,
            )
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        df = float(np.mean(np.diff(f_grid)))
        return float(np.sum(integrand) * df)

    def compute_sprs_spectrum_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str = "forward",
    ) -> np.ndarray:
        """计算 SpRS 噪声 PSD G_sprs(f) [W/Hz]，在 f_grid 每个频率点评估。

        返回 shape (N_f,) 的噪声功率谱密度数组。
        用于绘制连续噪声功率谱曲线（而非信道积分噪声标量）。

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid
        f_grid : ndarray
            输出频率网格 [Hz]
        direction : {"forward", "backward"}

        Returns
        -------
        ndarray, shape (N_f,)
            G_sprs(f) [W/Hz] at each f_grid point
        """
        f_grid = np.asarray(f_grid, dtype=np.float64)
        df = self._validate_frequency_grid(f_grid)

        G_classical = self._build_classical_psd_matrix(wdm_grid, f_grid, df)
        G_pump = G_classical.sum(axis=0)
        if not np.any(G_pump > 0.0):
            return np.zeros_like(f_grid, dtype=np.float64)

        alpha2 = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)

        out = np.zeros_like(f_grid, dtype=np.float64)
        for idx, f1 in enumerate(f_grid):
            out[idx] = self._compute_sprs_psd_at_f1(
                fiber=fiber, f_grid=f_grid,
                G_pump=G_pump, alpha2=alpha2,
                f1=float(f1), direction=direction,
            )
        return out
