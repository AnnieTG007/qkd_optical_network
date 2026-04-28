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
        # σ_2d cache: 同一 (f_grid, T, A_eff, df) 在一次 Dash 启动里跨 model_key
        # 复用，避免重复做 ~6300² 个点的 Raman / phonon 插值。LRU=2 已足够覆盖
        # forward+backward 调用对，且单份 σ_2d 在 N_f=6301 时约 302 MB。
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
        noise_bandwidth = np.array(
            [ch.B_s for ch in q_chs],
            dtype=np.float64,
        ).reshape(-1, 1)
        sigma = _raman_cross_section(
            f_q=f_q,
            f_c=f_2,
            g_R=g_R,
            n_th=n_th,
            delta_f=delta_f,
            noise_bandwidth=noise_bandwidth,
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
        noise_bandwidth = np.array(
            [ch.B_s for ch in q_chs],
            dtype=np.float64,
        ).reshape(-1, 1)
        sigma = _raman_cross_section(
            f_q=f_q,
            f_c=f_2,
            g_R=g_R,
            n_th=n_th,
            delta_f=delta_f,
            noise_bandwidth=noise_bandwidth,
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

    # --- 噪声 PSD 谱计算 — 向量化版本（全频率一次计算）-----------

    def _get_sigma_2d(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        df: float,
    ) -> np.ndarray:
        """返回 (N_f, N_f) 拉曼横截面矩阵 σ_2d，跨调用复用。

        σ_2d 仅依赖 (f_grid, T_kelvin, A_eff, df)，与 G_pump、L、direction、
        model_key 无关。同一次 Dash 启动里多个 model_key 都会复用同一份。

        Cache key 用 f_grid 的二进制 sha1 + 标量参数；LRU 由
        ``self._sigma_cache_max`` 控制。
        """
        import hashlib

        f_arr = np.ascontiguousarray(f_grid, dtype=np.float64)
        key = (
            hashlib.sha1(f_arr.tobytes()).hexdigest(),
            float(fiber.T_kelvin),
            float(fiber.A_eff),
            float(df),
        )
        cached = self._sigma_cache.get(key)
        if cached is not None:
            return cached

        f1_2d = f_arr.reshape(-1, 1)
        f2_2d = f_arr.reshape(1, -1)
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

    def _compute_sprs_psd_batch(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        G_pump: np.ndarray,
        direction: str,
        df: float,
    ) -> np.ndarray:
        """向量化计算 SpRS 噪声 PSD G_sprs(f) [W/Hz]，一次处理全部 f_grid 点。

        将原来的 N_f 次循环（每次 O(N_f²)）合并为一次 O(N_f²) 操作，
        通过 2D 广播完全消除 Python for 循环。

        Parameters
        ----------
        fiber : Fiber
        f_grid : ndarray, shape (N_f,)
        G_pump : ndarray, shape (N_f,)  — 总泵浦 PSD
        direction : {"forward", "backward"}
        df : float  — 频率网格步长

        Returns
        -------
        ndarray, shape (N_f,)  — G_sprs(f) at each f_grid point
        """
        sigma_2d = self._get_sigma_2d(fiber, f_grid, df)  # (N_f, N_f)

        # ---- 衰减系数（2D 广播）----
        alpha1_2d = fiber.get_loss_at_freq(f_grid).reshape(-1, 1)  # (N_f, 1)
        alpha2_2d = fiber.get_loss_at_freq(f_grid).reshape(1, -1)  # (1, N_f)

        # ---- 泵浦功率（2D 广播）----
        G_pump_2d = np.asarray(G_pump, dtype=np.float64).reshape(1, -1)  # (1, N_f)

        # ---- 方向传播积分 ----
        if direction == "forward":
            integrand_2d = _forward_propagation(
                sigma=sigma_2d,
                P_pump=G_pump_2d,
                alpha1=alpha1_2d,
                alpha2=alpha2_2d,
                L=fiber.L,
            )  # (N_f, N_f)
        elif direction == "backward":
            integrand_2d = _backward_propagation(
                sigma=sigma_2d,
                P_pump=G_pump_2d,
                alpha1=alpha1_2d,
                alpha2=alpha2_2d,
                L=fiber.L,
            )  # (N_f, N_f)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        # ---- 积分（沿泵浦轴求和）----
        # 对每个信号频率 f_grid[j]，对所有泵浦频率积分
        # integrand_2d[j, k]: f_grid[j] 作为信号、f_grid[k] 作为泵浦的贡献
        out_bin_power = np.sum(integrand_2d, axis=1) * df  # (N_f,) [W]
        return out_bin_power / df  # [W/Hz]

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

        # GPU dispatch: offload O(N_f²) propagation matrix to GPU when available.
        # NOTE: _get_gpu_module is defined in fwm_solver — gpu_utils only exposes
        # has_cupy / get_array_module. The earlier import path was silently
        # failing into the CPU branch.
        try:
            from qkd_sim.physical.noise.fwm_solver import _get_gpu_module
            _, is_gpu = _get_gpu_module()
            if is_gpu:
                return self._compute_sprs_psd_batch_gpu(
                    fiber=fiber, f_grid=f_grid, G_pump=G_pump,
                    direction=direction, df=df,
                )
        except Exception:
            pass

        # CPU vectorized path: one O(N_f²) operation, no Python for loops
        return self._compute_sprs_psd_batch(
            fiber=fiber,
            f_grid=f_grid,
            G_pump=G_pump,
            direction=direction,
            df=df,
        )

    def _compute_sprs_psd_batch_gpu(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        G_pump: np.ndarray,
        direction: str,
        df: float,
    ) -> np.ndarray:
        """GPU-accelerated SpRS PSD batch computation.

        Mirrors _compute_sprs_psd_batch but moves the (N_f, N_f) propagation
        integral to GPU. Raman gain / phonon / cross-section computed on CPU
        (scipy.interpolate doesn't support CuPy), then sigma_2d is transferred
        to GPU for the matrix multiply and propagation integral.
        """
        from qkd_sim.physical.noise.fwm_solver import _get_gpu_module
        xp, _ = _get_gpu_module()

        # CPU: σ_2d (scipy-based interpolation must stay on CPU). Cached across
        # repeated calls with the same (f_grid, T, A_eff, df).
        sigma_2d = self._get_sigma_2d(fiber, f_grid, df)  # (N_f, N_f) numpy

        # GPU: transfer and run propagation integral
        sigma_gpu = xp.asarray(sigma_2d)
        G_pump_gpu = xp.asarray(G_pump, dtype=xp.float64).reshape(1, -1)
        alpha_np = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)
        alpha1_gpu = xp.asarray(alpha_np).reshape(-1, 1)
        alpha2_gpu = xp.asarray(alpha_np).reshape(1, -1)

        if direction == "forward":
            d_alpha = alpha2_gpu - alpha1_gpu
            exp_m_alpha1_L = xp.exp(-alpha1_gpu * fiber.L)
            equal_mask = xp.abs(d_alpha) < _ALPHA_EPS
            safe_d_alpha = xp.where(equal_mask, xp.ones_like(d_alpha), d_alpha)
            integral = xp.where(
                equal_mask,
                float(fiber.L),
                (1.0 - xp.exp(-d_alpha * fiber.L)) / safe_d_alpha,
            )
            integrand = G_pump_gpu * sigma_gpu * exp_m_alpha1_L * integral
        elif direction == "backward":
            sum_alpha = alpha1_gpu + alpha2_gpu
            integrand = G_pump_gpu * sigma_gpu * (1.0 - xp.exp(-sum_alpha * fiber.L)) / sum_alpha
        else:
            raise ValueError(f"Unsupported direction: {direction!r}")

        out_bin_power = xp.sum(integrand, axis=1) * df  # (N_f,) [W]
        result = out_bin_power / df                      # (N_f,) [W/Hz]
        return np.asarray(result.get())

    def compute_sprs_spectrum_conti_l_array(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        L_arr: np.ndarray,
        direction: str = "forward",
    ) -> np.ndarray:
        """Compute continuous SpRS spectrum for multiple fiber lengths.

        This is the same formula path as ``compute_sprs_spectrum_conti``, but
        it hoists the Raman cross-section, pump PSD, and loss matrices outside
        the length loop. The returned array has shape ``(N_f, N_L)`` and each
        column matches a separate call using ``fiber.L == L_arr[i]``.
        """
        f_grid = np.asarray(f_grid, dtype=np.float64)
        L_values = np.asarray(L_arr, dtype=np.float64).reshape(-1)
        df = self._validate_frequency_grid(f_grid)

        G_classical = self._build_classical_psd_matrix(wdm_grid, f_grid, df)
        G_pump = G_classical.sum(axis=0)
        if not np.any(G_pump > 0.0):
            return np.zeros((f_grid.size, L_values.size), dtype=np.float64)

        # GPU dispatch: the (N_f, N_f, N_L) propagation tensor benefits hugely
        # from offload — at N_f=6301 the CPU path is forced to batch=1.
        try:
            from qkd_sim.physical.noise.fwm_solver import _get_gpu_module
            _, is_gpu = _get_gpu_module()
            if is_gpu:
                return self._compute_sprs_psd_batch_gpu_l_array(
                    fiber=fiber, f_grid=f_grid, G_pump=G_pump,
                    L_values=L_values, direction=direction, df=df,
                )
        except Exception:
            pass

        sigma_2d = self._get_sigma_2d(fiber, f_grid, df)

        alpha_grid = fiber.get_loss_at_freq(f_grid)
        alpha1_2d = alpha_grid.reshape(-1, 1)
        alpha2_2d = alpha_grid.reshape(1, -1)
        base_2d = np.asarray(G_pump, dtype=np.float64).reshape(1, -1) * sigma_2d

        out = np.zeros((f_grid.size, L_values.size), dtype=np.float64)
        # Limit intermediate (N_f, N_f, batch) array memory usage.
        # Each such array uses N_f² × batch × 8 bytes.  Raised from 256 MB to
        # 1024 MB so N_f=6301 (≈302 MB per slice) keeps batch ≥ 3 instead of 1.
        _MAX_SPRS_TMP_MB = 1024
        _bytes_per_batch = f_grid.size * f_grid.size * 8
        max_l_batch = max(1, int((_MAX_SPRS_TMP_MB * 1024 * 1024) / max(_bytes_per_batch, 1)))
        for start in range(0, L_values.size, max_l_batch):
            stop = min(start + max_l_batch, L_values.size)
            L_batch = L_values[start:stop].reshape(1, 1, -1)

            if direction == "forward":
                d_alpha = alpha2_2d - alpha1_2d
                equal_mask = np.abs(d_alpha) < _ALPHA_EPS
                d_alpha_3d = d_alpha[:, :, np.newaxis]
                equal_3d = equal_mask[:, :, np.newaxis]
                exp_m_alpha1_L = np.exp(-alpha1_2d[:, :, np.newaxis] * L_batch)
                integral = np.where(
                    equal_3d,
                    L_batch,
                    (1.0 - np.exp(-d_alpha_3d * L_batch))
                    / np.where(equal_3d, 1.0, d_alpha_3d),
                )
                batch = base_2d[:, :, np.newaxis] * exp_m_alpha1_L * integral
            elif direction == "backward":
                sum_alpha = alpha1_2d + alpha2_2d
                sum_alpha_3d = sum_alpha[:, :, np.newaxis]
                batch = (
                    base_2d[:, :, np.newaxis]
                    * (1.0 - np.exp(-sum_alpha_3d * L_batch))
                    / sum_alpha_3d
                )
            else:
                raise ValueError(f"Unsupported direction: {direction}")

            out_bin_power = np.sum(batch, axis=1) * df
            out[:, start:stop] = out_bin_power / df

        return out

    def _compute_sprs_psd_batch_gpu_l_array(
        self,
        fiber: Fiber,
        f_grid: np.ndarray,
        G_pump: np.ndarray,
        L_values: np.ndarray,
        direction: str,
        df: float,
    ) -> np.ndarray:
        """GPU-accelerated multi-length SpRS PSD batch.

        Parallels ``_compute_sprs_psd_batch_gpu`` but adds a batched L axis.
        σ_2d / α are computed on CPU (scipy interpolation), then transferred
        to GPU; the (N_f, N_f, batch) propagation tensor is materialised
        on-device and reduced along the pump axis. Batch size is sized to
        roughly 40% of free GPU memory.
        """
        from qkd_sim.physical.noise.fwm_solver import _get_gpu_module
        xp, _ = _get_gpu_module()

        # Reuse CPU σ_2d cache (computation must stay on CPU due to scipy)
        sigma_2d = self._get_sigma_2d(fiber, f_grid, df)
        n_f = f_grid.size

        sigma_gpu = xp.asarray(sigma_2d)
        alpha_np = np.asarray(fiber.get_loss_at_freq(f_grid), dtype=np.float64)
        alpha1_gpu = xp.asarray(alpha_np).reshape(-1, 1, 1)  # (N_f, 1, 1)
        alpha2_gpu = xp.asarray(alpha_np).reshape(1, -1, 1)  # (1, N_f, 1)
        base_gpu = xp.asarray(G_pump, dtype=xp.float64).reshape(1, -1) * sigma_gpu  # (N_f, N_f)
        L_gpu = xp.asarray(L_values, dtype=xp.float64).reshape(1, 1, -1)            # (1, 1, N_L)

        # Memory-aware batching on GPU. Each (N_f, N_f, batch) float64 tensor
        # costs ``N_f² × batch × 8 bytes``. The peak GPU allocation is:
        #   pre-loop (fixed): sigma_gpu + base_gpu + d_alpha + safe_d_alpha
        #                     = 4 × N_f² × 8 bytes  (≈ 1.24 GB at N_f=6301)
        #   forward loop:     exp + integral + batch_t = 3 × N_f² × 8 × batch
        #   backward loop:    batch_t = 1 × N_f² × 8 × batch  (sum_alpha is pre-loop)
        # So the per-batch marginal cost is 3 slices (forward, the worst case).
        # We reserve ~85% of free VRAM for this budget (leaving headroom for
        # CuPy's memory pool and any FWM allocations that may still be live).
        try:
            free_bytes, _total = xp.cuda.runtime.memGetInfo()
            budget = int(free_bytes * 0.85)
        except Exception:
            budget = 1024 * 1024 * 1024  # 1 GB fallback
        _fixed_slices = 4  # sigma + base + d_alpha + safe_d_alpha
        _per_batch_slices = 3  # exp + integral + batch_t (forward worst case)
        budget_slices = budget // max(n_f * n_f * 8, 1)
        max_l_batch = max(1, min(
            int(L_values.size),
            (budget_slices - _fixed_slices) // max(_per_batch_slices, 1),
        ))

        out_gpu = xp.zeros((n_f, L_values.size), dtype=xp.float64)
        if direction == "forward":
            d_alpha = alpha2_gpu[:, :, 0:1] - alpha1_gpu[:, :, 0:1]  # (N_f, N_f, 1)
            equal_mask = xp.abs(d_alpha) < _ALPHA_EPS
            safe_d_alpha = xp.where(equal_mask, xp.ones_like(d_alpha), d_alpha)
            for start in range(0, L_values.size, max_l_batch):
                stop = min(start + max_l_batch, L_values.size)
                L_b = L_gpu[:, :, start:stop]
                exp_m_alpha1_L = xp.exp(-alpha1_gpu[:, :, 0:1] * L_b)
                integral = xp.where(
                    equal_mask,
                    L_b,
                    (1.0 - xp.exp(-d_alpha * L_b)) / safe_d_alpha,
                )
                batch_t = base_gpu[:, :, xp.newaxis] * exp_m_alpha1_L * integral
                out_bin_power = xp.sum(batch_t, axis=1) * df  # (N_f, batch) [W]
                out_gpu[:, start:stop] = out_bin_power / df  # [W/Hz]
        elif direction == "backward":
            sum_alpha = alpha1_gpu[:, :, 0:1] + alpha2_gpu[:, :, 0:1]
            for start in range(0, L_values.size, max_l_batch):
                stop = min(start + max_l_batch, L_values.size)
                L_b = L_gpu[:, :, start:stop]
                batch_t = (
                    base_gpu[:, :, xp.newaxis]
                    * (1.0 - xp.exp(-sum_alpha * L_b)) / sum_alpha
                )
                out_bin_power = xp.sum(batch_t, axis=1) * df
                out_gpu[:, start:stop] = out_bin_power / df
        else:
            raise ValueError(f"Unsupported direction: {direction!r}")

        return np.asarray(out_gpu.get())
