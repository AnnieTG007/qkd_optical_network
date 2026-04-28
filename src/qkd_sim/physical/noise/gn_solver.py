"""GN-model 经典信道非线性干扰（NLI）噪声求解器。

公式来源：Poggiolini et al., "The GN Model of Fiber Nonlinear Propagation",
arXiv:1209.0394, Eq.1 + Eq.120（通用积分 + 单信道闭式）

实现说明
--------
本模块计算经典信道自身受到的非线性干扰噪声（Co-channel NLI），
与量子信道 FWM/SpRS 噪声（fwm_solver / sprs_solver）的物理机制相同，
但目标信道为经典信道（自干涉），且使用 Poggiolini 通用双重积分形式。

与 NoiseSolver ABC 的区别：
- NoiseSolver: 计算 shape (N_q,) 的量子信道噪声
- GNModelSolver: 计算 shape (N_c,) 或 shape (N_f,) 的经典信道 NLI

数学公式
--------
前向 NLI PSD（单偏振，单跨段）：
    G_NLI(f) = (16/27) × γ²
               × ∬ G_TX(f₁) × G_TX(f₂) × G_TX(f₁+f₂-f)
               × |μ(f, f₁, f₂)|² df₁ df₂

FWM 系数（等损耗近似 Δα = 2α）：
    |μ|² = [exp(-2αL) - 2·exp(-αL)·cos(Δβ·L) + 1] / (α² + Δβ²)

相位失配（β₂ 近似）：
    Δβ = 4π² × β₂ × (f₁ - f) × (f₂ - f)
    β₂ = -D_c(λ₀) × λ₀² / (2πc)  [s²/m]
    其中 λ₀ 为 C 波段参考波长（193.5 THz 对应 ~1549.3 nm）

参考文献
--------
- Poggiolini et al., arXiv:1209.0394, Eq.1, Eq.120-123
- Carena et al., "GN Model of Fiber-Optic Nonlinearities", J. Lightw. Technol., 2012
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as c_light

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import SpectrumType, WDMGrid, validate_uniform_frequency_grid


# ---- 内部辅助函数 -------------------------------------------------------

def _fwm_coefficient_single_alpha(
    alpha: float,
    delta_beta: np.ndarray,
    L: float,
) -> np.ndarray:
    """FWM 系数 |μ|²（等损耗近似，Δα = 2α）。

    等损耗近似下 Δα = αₖ+αᵢ+αⱼ-α₁ = 3α-α = 2α，
    代入公式 2.1.1 简化分母：(2α)²/4 + Δβ² = α² + Δβ²。

    Parameters
    ----------
    alpha : float
        光纤衰减系数 [1/m]（C 波段常数）
    delta_beta : ndarray
        相位失配 Δβ [rad/m]
    L : float
        光纤长度 [m]

    Returns
    -------
    ndarray
        |μ|² 系数 [m²]，形状与 delta_beta 相同
    """
    exp_m_alpha_L = np.exp(-alpha * L)
    numerator = (
        np.exp(-2.0 * alpha * L)
        - 2.0 * exp_m_alpha_L * np.cos(delta_beta * L)
        + 1.0
    )
    denominator = alpha ** 2 + delta_beta ** 2
    return numerator / denominator


def _compute_beta2(fiber: Fiber, f_ref: float) -> float:
    """计算 β₂ [s²/m]（二阶色散系数）。

    由色散系数 D_c 反推：
        D_c = -(2π/c) × d²β/dω²  →  β₂ = d²β/dω² = -D_c × c / (2π)

    或者从波长表达式：
        β₂ = -D_c(λ₀) × λ₀² / (2πc)

    Parameters
    ----------
    fiber : Fiber
    f_ref : float
        参考频率 [Hz]（用于取 D_c，通常用 C 波段中心 193.5 THz）

    Returns
    -------
    float
        β₂ [s²/m]
    """
    lambda_ref = c_light / f_ref
    D_c = fiber.get_dispersion_at_freq(f_ref)  # s/m²
    beta2 = -D_c * lambda_ref**2 / (2.0 * np.pi * c_light)
    return beta2


# ---- 主求解器类 ---------------------------------------------------------

class GNModelSolver:
    """GN-model 经典信道 NLI 噪声求解器。

    计算经典信道受 Kerr 非线性干扰产生的噪声功率谱密度（NLI）。
    使用 Poggiolini 通用数值双重积分（arXiv:1209.0394 Eq.1），
    支持任意 G_TX(f) 谱形（矩形、升余弦、OSA）。

    本质物理
    --------
    NLI 是同一信道内（和相邻信道间）四波混频（FWM）累积的结果。
    各频率三元组 (f₁, f₂, f₁+f₂-f) 产生的噪声功率在信道带宽内积分。
    当所有信道为 SINGLE_FREQ 时，退化为纯离散 FWM 泵浦结果（与 DiscreteFWMSolver 一致）。

    Attributes
    ----------
    f_ref : float
        参考频率 [Hz]，用于计算 β₂。默认为 193.5 THz（C 波段中心）。

    Examples
    --------
    >>> solver = GNModelSolver()
    >>> f_grid = np.linspace(191e12, 195e12, 5000)
    >>> G_nli = solver.compute_nli_psd(fiber, wdm_grid, f_grid)  # shape (N_f,)
    >>> result = solver.compute_nli_per_channel(fiber, wdm_grid, f_grid)
    >>> result["nli_fwd"]  # shape (N_c,)
    """

    def __init__(self, f_ref: float = 193.5e12) -> None:
        """Initialize GNModelSolver.

        Parameters
        ----------
        f_ref : float
            参考频率 [Hz]，用于从 FiberConfig 的 D_c 反推 β₂。
            默认 193.5 THz（C 波段中心）。
        """
        self.f_ref = f_ref

    @staticmethod
    def _validate_frequency_grid(f_grid: np.ndarray) -> float:
        """验证 1D 均匀频率网格，返回步长 df [Hz]。

        Raises AssertionError if grid is not valid.
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
    def _build_total_psd(
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        df: float,
    ) -> np.ndarray:
        """在 f_grid 上构建所有经典信道的总发射 PSD。

        SINGLE_FREQ 信道：delta 近似 G=P/df 置于最近格点。
        连续谱信道：调用 channel.get_psd(f_grid)。

        Parameters
        ----------
        wdm_grid : WDMGrid
        f_grid : ndarray, shape (N_f,)
        df : float
            频率网格步长 [Hz]

        Returns
        -------
        ndarray, shape (N_f,)
            总发射 PSD G_TX(f) [W/Hz]
        """
        classical_channels = wdm_grid.get_classical_channels()
        if len(classical_channels) == 0:
            return np.zeros(f_grid.size, dtype=np.float64)

        total_psd = np.zeros(f_grid.size, dtype=np.float64)
        for ch in classical_channels:
            if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
                idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
                total_psd[idx] += ch.power / df
            else:
                total_psd += ch.get_psd(f_grid)
        return total_psd

    def compute_nli_psd(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """计算前向 NLI 噪声 PSD G_NLI(f) [W/Hz]。

        Poggiolini Eq.1（单偏振，单跨段）：
            G_NLI(f) = (16/27) × γ²
                       × ∬ G_TX(f₁) G_TX(f₂) G_TX(f₁+f₂-f)
                       × |μ(f,f₁,f₂)|² df₁ df₂

        使用均匀网格黎曼和近似双重积分。

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid
        f_grid : ndarray, shape (N_f,)
            输出频率网格 [Hz]（须均匀）

        Returns
        -------
        ndarray, shape (N_f,)
            G_NLI(f) [W/Hz]
        """
        df = self._validate_frequency_grid(f_grid)
        f_grid = np.asarray(f_grid, dtype=np.float64)

        # ---- 构建总发射 PSD ----
        G_tx = self._build_total_psd(wdm_grid, f_grid, df)
        alpha = float(fiber.alpha)
        gamma = float(fiber.gamma)
        L = float(fiber.L)

        active = G_tx > 0.0
        if not np.any(active):
            return np.zeros_like(f_grid, dtype=np.float64)

        f_active = f_grid[active]
        G_active = G_tx[active]
        n_active = f_active.size

        # ---- 构建泵浦频率对网格 (N_active, N_active) ----
        Fi, Fj = np.meshgrid(f_active, f_active, indexing="ij")
        Gi, Gj = np.meshgrid(G_active, G_active, indexing="ij")

        # ---- 预计算泵浦对和 Fk_ij = Fi + Fj（与输出 f 无关）----
        Fk_ij = Fi + Fj  # shape (N_a, N_a)

        # ---- β₂（恒定近似，在 f_ref 处取值）----
        beta2 = _compute_beta2(fiber, self.f_ref)

        # ---- 分块处理输出频率以控制内存 ----
        n_f = f_grid.size
        out = np.zeros(n_f, dtype=np.float64)

        N_pairs = n_active * n_active
        # 大泵浦对时减小分块以防止 OOM
        chunk_size = 10 if N_pairs * n_active > 50_000_000 else 50

        # 预计算泵浦网格（float32 节省内存）
        Fi_f32 = Fi.astype(np.float32)
        Fj_f32 = Fj.astype(np.float32)
        Gi_f32 = Gi.astype(np.float32)
        Gj_f32 = Gj.astype(np.float32)
        Fk_ij_f32 = Fk_ij.astype(np.float32)

        pump_f32 = f_grid.astype(np.float32)
        G_tx_f32 = G_tx.astype(np.float32)

        for start in range(0, n_f, chunk_size):
            end = min(start + chunk_size, n_f)
            n_chunk = end - start
            f_out_chunk = f_grid[start:end]  # shape (chunk,)

            # Fk for all pump pairs × all output freq in this chunk
            # Fk = Fk_ij - f_out (broadcasting: (N_a,N_a) - (chunk,) → (N_a,N_a,chunk))
            Fk = Fk_ij_f32[:, :, np.newaxis] - f_out_chunk[np.newaxis, np.newaxis, :]

            # ---- 插值得到 G_TX(Fk) ----
            # Shape: (chunk, N_pairs) after transpose+reshape
            Fk_T = Fk.transpose(2, 0, 1).reshape(n_chunk, -1)  # (chunk, N_a²)
            sorted_idx = np.searchsorted(pump_f32, Fk_T)
            sorted_idx = np.clip(sorted_idx, 1, len(pump_f32) - 1)

            x0 = pump_f32[sorted_idx - 1]
            x1 = pump_f32[sorted_idx]
            y0 = G_tx_f32[sorted_idx - 1]
            y1 = G_tx_f32[sorted_idx]
            t = np.clip((Fk_T - x0) / (x1 - x0), 0.0, 1.0)
            Gk_T = (y0 + (y1 - y0) * t).astype(np.float32)  # (chunk, N_pairs)

            for j in range(n_chunk):
                f_out = float(f_out_chunk[j])
                # Gk at Fk = Fi + Fj - f_out
                Gk_2d = Gk_T[j, :].reshape(n_active, n_active)  # (N_a, N_a)

                # ---- 相位失配 Δβ = 4π²β₂(f₁-f)(f₂-f) ----
                # Fi_f32, Fj_f32: (N_a, N_a)
                dbeta = (
                    4.0
                    * np.pi**2
                    * beta2
                    * (Fi_f32 - f_out)
                    * (Fj_f32 - f_out)
                )

                # ---- FWM 系数 |μ|² ----
                eta = _fwm_coefficient_single_alpha(alpha, dbeta.astype(np.float64), L)
                eta_f32 = eta.astype(np.float32)

                # ---- 非线性干涉积分 ----
                valid_mask = (Gk_2d > 0.0) & (Gi_f32 > 0.0) & (Gj_f32 > 0.0)
                if not np.any(valid_mask):
                    out[start + j] = 0.0
                    continue

                integrand = eta_f32 * Gi_f32 * Gj_f32 * Gk_2d
                integral_ij = np.sum(integrand[valid_mask]) * df * df

                # Poggiolini Eq.1: (16/27)γ² × integral
                out[start + j] = (16.0 / 27.0) * (gamma**2) * integral_ij

        return out

    def compute_nli_psd_backward(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """计算后向 NLI 噪声 PSD G_NLI_bwd(f) [W/Hz]（弱瑞利近似）。

        弱瑞利近似：G_NLI_bwd(f) ≈ S·α_R × L_eff × G_NLI_fwd(f)

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid
        f_grid : ndarray

        Returns
        -------
        ndarray, shape (N_f,)
            G_NLI_bwd(f) [W/Hz]
        """
        G_nli_fwd = self.compute_nli_psd(fiber, wdm_grid, f_grid)
        # C-band: alpha is constant, L_eff = (1 - exp(-αL))/α
        L_eff = (1.0 - np.exp(-fiber.alpha * fiber.L)) / fiber.alpha
        return fiber.rayleigh_coeff * L_eff * G_nli_fwd

    def compute_nli_per_channel(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """每个经典信道积分的 NLI 噪声功率。

        对每个经典信道，在其信号带宽内对 G_NLI(f) 积分：
            P_NLI,ch = ∫_{f_c-B_s/2}^{f_c+B_s/2} G_NLI(f) df

        Parameters
        ----------
        fiber : Fiber
        wdm_grid : WDMGrid
        f_grid : ndarray, shape (N_f,)

        Returns
        -------
        dict[str, ndarray]
            "nli_fwd" : shape (N_c,) 前向 NLI [W]
            "nli_bwd" : shape (N_c,) 后向 NLI [W]
        """
        df = self._validate_frequency_grid(f_grid)

        classical_channels = wdm_grid.get_classical_channels()
        if len(classical_channels) == 0:
            n_c = len(wdm_grid.get_channel_frequencies())
            return {
                "nli_fwd": np.zeros(n_c, dtype=np.float64),
                "nli_bwd": np.zeros(n_c, dtype=np.float64),
            }

        G_nli_fwd = self.compute_nli_psd(fiber, wdm_grid, f_grid)
        G_nli_bwd = self.compute_nli_psd_backward(fiber, wdm_grid, f_grid)

        nli_fwd = np.zeros(len(classical_channels), dtype=np.float64)
        nli_bwd = np.zeros(len(classical_channels), dtype=np.float64)

        for i, ch in enumerate(classical_channels):
            if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
                # 单频信道：带宽 = df，积分变为 G_NLI × df
                idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
                nli_fwd[i] = G_nli_fwd[idx] * df
                nli_bwd[i] = G_nli_bwd[idx] * df
            else:
                # 连续信道：积分带宽 [f_center - B_s/2, f_center + B_s/2]
                f_lo = ch.f_center - ch.B_s / 2.0
                f_hi = ch.f_center + ch.B_s / 2.0
                mask = (f_grid >= f_lo) & (f_grid < f_hi)
                nli_fwd[i] = float(np.sum(G_nli_fwd[mask]) * df)
                nli_bwd[i] = float(np.sum(G_nli_bwd[mask]) * df)

        return {"nli_fwd": nli_fwd, "nli_bwd": nli_bwd}

    # ---- 闭式参考（单信道矩形谱，Eq.120）----
    # 用于交叉验证：数值积分结果 vs Poggiolini 闭式解

    @staticmethod
    def _single_channel_nli_closed_form(
        P_ch: float,
        B_s: float,
        alpha: float,
        gamma: float,
        L: float,
        beta2: float,
    ) -> float:
        r"""单信道矩形谱的 NLI 闭式近似（Poggiolini Eq.120）。

        假设信道为理想矩形 PSD：G_TX = P_ch / B_s（|f-f_c| ≤ B_s/2）。
        闭式积分结果（asinh 形式）为：

            G_NLI = (8/27) × γ² × G³ × L_eff²
                    × asinh(π²|β₂|L_eff_a × B_s² / 2) / (π|β₂|)
            P_NLI = G_NLI × B_s

        其中：
                L_eff = (1 - exp(-αL)) / α
                L_eff_a = 1/(2α)（渐近有效长度）
                G = P_ch / B_s（信道 PSD [W/Hz]）

        Parameters
        ----------
        P_ch : float
            单信道功率 [W]
        B_s : float
            信道带宽 [Hz]
        alpha : float
            衰减系数 [1/m]
        gamma : float
            非线性系数 [1/(W·m)]
        L : float
            光纤长度 [m]
        beta2 : float
            二阶色散系数 [s²/m]

        Returns
        -------
        float
            P_NLI [W]（积分到信道带宽内）

        Notes
        -----
        Eq.120 是近似公式（忽略残留相位失配），
        数值积分结果通常比 Eq.120 高 10-30%，这是正常的。
        """
        if P_ch <= 0 or B_s <= 0:
            return 0.0

        G = P_ch / B_s  # PSD [W/Hz]
        L_eff = (1.0 - np.exp(-alpha * L)) / alpha  # [m]
        L_eff_a = 1.0 / (2.0 * alpha)  # asymptotic [m]

        # asinh argument: π²|β₂|L_eff_a × B_s² / 2
        asinh_arg = (np.pi**2 * abs(beta2) * L_eff_a * B_s**2) / 2.0
        # asinh(x) = log(x + sqrt(x²+1))
        asinh_val = np.log(asinh_arg + np.sqrt(asinh_arg**2 + 1.0))

        # G_NLI PSD at channel center
        G_nli = (
            (8.0 / 27.0)
            * (gamma**2)
            * (G**3)
            * (L_eff**2)
            * asinh_val
            / (np.pi * abs(beta2))
        )
        # Total in-band NLI power
        return G_nli * B_s
