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
计算路径切换至 GPU，消除逐频率 Python 循环，实现 ~10-100x 加速。
"""

from __future__ import annotations

import os

import numpy as np

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import SpectrumType, WDMGrid
from qkd_sim.physical.noise.base import NoiseSolver

# Diagnostic prints (N_active / chunk_size) only emitted when DEBUG_MODE is set.
_DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "").lower() in ("1", "true", "yes", "on")

# GPU acceleration: shared helper from gpu_utils (also used by sprs_solver).
from qkd_sim.utils.gpu_utils import get_gpu_module as _get_gpu_module


# =============================================================================
# Unified η coefficient + F antiderivative helpers (CPU + GPU in one)
# =============================================================================

def _fwm_coefficient(
    delta_alpha: "np.ndarray",
    delta_beta: "np.ndarray",
    L: "float | np.ndarray",
    xp: "type | None" = None,
) -> "np.ndarray":
    """FWM 系数 η。

    公式 2.1.1 (formulas_fwm.md):
        η = [exp(-Δα·L) - 2·exp(-Δα·L/2)·cos(Δβ·L) + 1]
            / [(Δα)²/4 + (Δβ)²]

    L 可以是标量（单长度）或 ndarray（L_arr 广播）。
    当 xp 未指定时自动从 delta_alpha 推断。
    """
    if xp is None:
        xp = np
    L_arr = xp.asarray(L, dtype=xp.float64)
    da = delta_alpha.astype(xp.float64)
    db = delta_beta.astype(xp.float64)
    exp_da_L = xp.exp(-da * L_arr)
    exp_da_L_2 = xp.exp(-da * L_arr / 2.0)
    cos_db_L = xp.cos(db * L_arr)
    numerator = exp_da_L - 2.0 * exp_da_L_2 * cos_db_L + 1.0
    denominator = (da ** 2) / 4.0 + db ** 2
    return numerator / denominator


def _F_antiderivative(
    l: "np.ndarray",
    z_obs: float,
    alpha1: "float | np.ndarray",
    delta_alpha: "np.ndarray",
    delta_beta: "np.ndarray",
    L: "float | np.ndarray",
    xp: "type | None" = None,
) -> "np.ndarray":
    """后向 FWM 积分原函数 F(l)。

    公式 2.2.6 (formulas_fwm.md):
        F(l) = exp(α₁·z_obs) / denom
               × [ -exp(-A·l)/A
                   - exp(-B·l)/(B²+Δβ²)·(-B·cos(Δβ·l)+Δβ·sin(Δβ·l))
                   - exp(-C·l)/C ]

    l 可以是 ndarray（与 delta_alpha 同形）或标量广播。
    alpha1 可以是标量或与 delta_alpha 兼容的数组。
    """
    if xp is None:
        xp = np
    da = delta_alpha.astype(xp.float64)
    db = delta_beta.astype(xp.float64)
    a1 = xp.asarray(alpha1, dtype=xp.float64)
    if a1.ndim == 1 and da.ndim > 1:
        a1 = a1.reshape((a1.shape[0],) + (1,) * (da.ndim - 1))
    A = da + 2.0 * a1
    B = da / 2.0 + 2.0 * a1
    C = xp.ones_like(da) * (2.0 * a1)
    denom = (da ** 2) / 4.0 + db ** 2
    exp_z = xp.exp(a1 * z_obs)
    term_A = -xp.exp(-A * l) / A
    term_B = (
        -xp.exp(-B * l)
        / (B ** 2 + db ** 2)
        * (-B * xp.cos(db * l) + db * xp.sin(db * l))
    )
    term_C = -xp.exp(-C * l) / C
    return exp_z / denom * (term_A + term_B + term_C)


def _phase_mismatch(
    f2: "np.ndarray",
    f3: "np.ndarray",
    f4: "np.ndarray",
    D_c: float,
    D_slope: float,
    xp: "type | None" = None,
    f_ref: float = 193.5e12,
) -> "np.ndarray":
    """相位失配 Δβ。

    Formula 2.2.3 (formulas_fwm.md):
        Δβ = (2π λ² / c) × |f₃-f₂| × |f₄-f₂|
             × [D_c(f₂) + (λ²/(2c)) × (|f₃-f₂|+|f₄-f₂|) × D_slope]
        λ = c / f2
        D_c(f₂) = D_c + D_slope × (λ - λ_ref)  （频率相关色散修正）

    与 fiber.get_phase_mismatch() 公式一致：D_c 取 f₂ 处的色散值（含斜率修正）。
    """
    if xp is None:
        xp = np
    c_light = 299792458.0
    lambda_c = c_light / f2
    lambda_ref = c_light / f_ref
    # Frequency-dependent D_c at f2 (matches fiber.get_dispersion_at_freq)
    D_c_f2 = D_c + D_slope * (lambda_c - lambda_ref)
    df32 = xp.abs(f3 - f2)
    df42 = xp.abs(f4 - f2)
    return (
        (2.0 * xp.pi * lambda_c ** 2 / c_light)
        * df32
        * df42
        * (D_c_f2 + (lambda_c ** 2 / (2.0 * c_light)) * (df32 + df42) * D_slope)
    )


class DiscreteFWMSolver(NoiseSolver):
    """离散 FWM 噪声求解器。

    计算量子信道受经典信道四波混频产生的前向/后向噪声功率。

    公式来源：docs/formulas_fwm.md 第 2-3 节
    算法：方案 B（等间隔索引算术），复杂度 O(N_q × N_c²)。

    Parameters
    ----------
    channel_spacing : float or None
        信道间隔 [Hz]。None 时从 wdm_grid 自动推断（要求等间隔）。
    """

    def __init__(
        self,
        channel_spacing: float | None = None,
        active_threshold_db: float = -50.0,
    ) -> None:
        """Initialize DiscreteFWMSolver.

        Parameters
        ----------
        channel_spacing : float or None
            信道间隔 [Hz]。None 时从 wdm_grid 自动推断（要求等间隔）。
        active_threshold_db : float
            连续谱计算中活跃频率 bin 的相对阈值 [dB]。
            G_min = G_tx.max() * 10^(threshold_db/10)。
            默认 -50.0（即 1e-5）；收紧到 -40.0 可减少 ~50% N_active。
        """
        self.channel_spacing = channel_spacing
        self._active_threshold_db = active_threshold_db
        self._inferred_spacing: float | None = None
        # 拓扑级缓存
        self._cache_key: tuple[int, ...] | None = None
        self._classical_set_cache: np.ndarray | None = None
        self._valid_combo_cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] | None = None

    # -------------------------------------------------------------------------
    # 离散模型（信道级）
    # -------------------------------------------------------------------------

    def _get_valid_combinations(
        self,
        n1: int,
        idx_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """枚举对量子信道 n₁ 产生 FWM 噪声的有效经典信道三元组 (n₂, n₃, n₄)。

        频率匹配：n₃ + n₄ - n₂ = n₁，即 n₂ = n₃ + n₄ - n₁。
        约束：n₂ ∈ idx_c，n₂ ≠ n₃，n₂ ≠ n₄。
        """
        cache_key = tuple(sorted(idx_c.tolist()))

        if (
            self._cache_key == cache_key
            and self._valid_combo_cache is not None
            and n1 in self._valid_combo_cache
        ):
            return self._valid_combo_cache[n1]

        self._cache_key = cache_key

        n3_grid, n4_grid = np.meshgrid(idx_c, idx_c, indexing="ij")
        n2_grid = n3_grid + n4_grid - n1

        sorted_idx_c = np.sort(idx_c)
        positions = np.searchsorted(sorted_idx_c, n2_grid.ravel())
        positions = np.clip(positions, 0, len(sorted_idx_c) - 1)
        in_classical_flat = sorted_idx_c.ravel()[positions] == n2_grid.ravel()
        in_classical = in_classical_flat.reshape(n2_grid.shape)
        not_spm_xpm = (n2_grid != n3_grid) & (n2_grid != n4_grid)
        valid_mask = in_classical & not_spm_xpm

        result = (n2_grid[valid_mask], n3_grid[valid_mask], n4_grid[valid_mask])

        if self._valid_combo_cache is None:
            self._valid_combo_cache = {}
        self._valid_combo_cache[n1] = result

        return result

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
        """Vectorized FWM noise for multiple quantum channels.

        Pre-computes invariants across the N_c×N_c pump grid, then validates
        n2=n3+n4-n1 against the classical set for all n1 simultaneously via np.isin.
        """
        n_q = len(n1_arr)
        n_c = len(idx_c)
        n3_grid, n4_grid = np.meshgrid(idx_c, idx_c, indexing="ij")  # (n_c, n_c)
        n2_base = n3_grid + n4_grid

        # ---- Pre-compute pump-pair invariants (independent of n1) ----
        f3_grid = f_all[n3_grid]
        f4_grid = f_all[n4_grid]
        P3_grid = P_all[n3_grid]
        P4_grid = P_all[n4_grid]
        alpha3_grid = fiber.get_loss_at_freq(f3_grid)
        alpha4_grid = fiber.get_loss_at_freq(f4_grid)
        D_grid = np.where(n3_grid == n4_grid, 3.0, 6.0)

        L = fiber.L
        gamma = fiber.gamma
        rayleigh = fiber.rayleigh_coeff
        is_fwd = (direction == "forward")

        # Build n2 for all n1 and validate in one shot
        n2_3d = n2_base[:, :, np.newaxis] - n1_arr.astype(int)[np.newaxis, np.newaxis, :]
        n3_3d = n3_grid[:, :, np.newaxis]
        n4_3d = n4_grid[:, :, np.newaxis]
        valid_3d = (
            np.isin(n2_3d, idx_c)
            & (n2_3d != n3_3d)
            & (n2_3d != n4_3d)
        )  # (n_c, n_c, n_q)

        P_result = np.zeros(n_q, dtype=np.float64)

        for i in range(n_q):
            mask = valid_3d[:, :, i]
            if not np.any(mask):
                continue

            n2_v = n2_3d[:, :, i][mask]
            f2_v = f_all[n2_v]
            f3_v = f3_grid[mask]
            f4_v = f4_grid[mask]
            P2_v = P_all[n2_v]
            P3_v = P3_grid[mask]
            P4_v = P4_grid[mask]
            D_v = D_grid[mask]
            alpha1 = alpha1_arr[i]

            alpha2_v = fiber.get_loss_at_freq(f2_v)
            delta_alpha = alpha4_grid[mask] + alpha3_grid[mask] + alpha2_v - alpha1
            delta_beta = fiber.get_phase_mismatch(f2=f2_v, f3=f3_v, f4=f4_v)

            if is_fwd:
                eta = _fwm_coefficient(delta_alpha, delta_beta, L)
                contrib = (
                    np.exp(-alpha1 * L) * (gamma ** 2 / 9.0)
                    * D_v ** 2 * eta * P2_v * P3_v * P4_v
                )
            else:
                F_L = _F_antiderivative(
                    np.full_like(delta_alpha, L), 0.0, alpha1,
                    delta_alpha, delta_beta, L,
                )
                F_0 = _F_antiderivative(
                    np.zeros_like(delta_alpha), 0.0, alpha1,
                    delta_alpha, delta_beta, L,
                )
                contrib = (
                    rayleigh * (gamma ** 2 / 9.0)
                    * D_v ** 2 * P2_v * P3_v * P4_v
                    * (F_L - F_0)
                )
            P_result[i] = float(contrib.sum())

        return P_result

    def _prepare_grid(
        self, fiber: Fiber, wdm_grid: WDMGrid
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """从 WDMGrid 构建整数索引网格。

        Returns
        -------
        f_q, n_q, idx_c, f_all, P_all, g
        """
        all_freqs = wdm_grid.get_channel_frequencies()
        all_powers = wdm_grid.get_channel_powers()

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

        f_min = all_freqs.min()
        all_indices = np.round((all_freqs - f_min) / g).astype(int)

        n_max = all_indices.max()
        f_all = f_min + np.arange(n_max + 1) * g
        P_all = np.zeros(n_max + 1)
        for idx, p in zip(all_indices, all_powers):
            P_all[idx] = p

        q_chs = wdm_grid.get_quantum_channels()
        c_chs = wdm_grid.get_classical_channels()
        f_q = np.array([ch.f_center for ch in q_chs])
        n_q = np.round((f_q - f_min) / g).astype(int)

        f_c = np.array([ch.f_center for ch in c_chs])
        idx_c = np.round((f_c - f_min) / g).astype(int)

        for nq in n_q:
            P_all[nq] = 0.0

        return f_q, n_q, idx_c, f_all, P_all, g

    def compute_forward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算各量子信道在光纤接收端（z=L）的前向 FWM 噪声功率。"""
        f_q, n_q, idx_c, f_all, P_all, _ = self._prepare_grid(fiber, wdm_grid)
        alpha1_arr = fiber.get_loss_at_freq(f_q)
        return self._compute_noise_for_channel_vec(
            n_q, idx_c, f_all, P_all, alpha1_arr, fiber, direction="forward",
        )

    def compute_backward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算各量子信道在光纤发射端（z=0）的后向 FWM 噪声功率。"""
        f_q, n_q, idx_c, f_all, P_all, _ = self._prepare_grid(fiber, wdm_grid)
        alpha1_arr = fiber.get_loss_at_freq(f_q)
        return self._compute_noise_for_channel_vec(
            n_q, idx_c, f_all, P_all, alpha1_arr, fiber, direction="backward",
        )

    # -------------------------------------------------------------------------
    # 连续模型（PSD 级）— 统一频谱计算 + 薄封装积分
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_frequency_grid(f_grid: np.ndarray) -> float:
        """验证 1D 积分网格并返回频率步长。"""
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
        """在 f_grid 上构建经典信道总发射 PSD。"""
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

    def compute_forward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续前向 FWM 噪声功率 [公式 2.3.1 + 2.3.6]。

        内部复用 compute_fwm_spectrum_conti 的 PSD 结果，
        在每个量子信道带宽 [f_c - B_s/2, f_c + B_s/2] 内积分得到功率。

        当经典信道全为 SINGLE_FREQ 时退化为离散模型（交叉验证）。
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

        # 核心 PSD 计算（复用 spectrum 路径）
        psd = self.compute_fwm_spectrum_conti(
            fiber, wdm_grid, f_grid, direction="forward"
        )
        return self._integrate_psd_per_channel(psd, f_grid, df, q_chs)

    def compute_backward_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """连续后向 FWM 噪声功率 [公式 2.3.2-2.3.6]。

        内部复用 compute_fwm_spectrum_conti 的 PSD 结果，
        在每个量子信道带宽 [f_c - B_s/2, f_c + B_s/2] 内积分得到功率。

        当经典信道全为 SINGLE_FREQ 时退化为离散模型（交叉验证）。
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

        psd = self.compute_fwm_spectrum_conti(
            fiber, wdm_grid, f_grid, direction="backward"
        )
        return self._integrate_psd_per_channel(psd, f_grid, df, q_chs)

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
            PSD G_fwm(f) [W/Hz]
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

        # Prefix sum: cumsum along frequency axis, scaled by df
        prefix = np.cumsum(psd, axis=0) * df  # (N_f,) or (N_f, N_L)

        if psd_2d:
            n_l = psd.shape[1]
            P = np.zeros((n_q, n_l), dtype=np.float64)
        else:
            P = np.zeros(n_q, dtype=np.float64)

        for i, ch in enumerate(q_chs):
            f_lo = ch.f_center - ch.B_s / 2.0
            f_hi = ch.f_center + ch.B_s / 2.0
            idx_lo = max(0, min(n_f - 1, int(np.round((f_lo - f0) / df))))
            idx_hi = max(0, min(n_f - 1, int(np.round((f_hi - f0) / df))))

            if idx_hi > idx_lo:
                if psd_2d:
                    P[i] = prefix[idx_hi] - prefix[idx_lo]
                else:
                    P[i] = float(prefix[idx_hi] - prefix[idx_lo])
            else:
                # 带宽内无频率点，用中心频率处的 PSD × df 近似
                idx = max(0, min(n_f - 1, int(np.round((ch.f_center - f0) / df))))
                if psd_2d:
                    P[i] = psd[idx] * df
                else:
                    P[i] = float(psd[idx] * df)

        return P

    # -------------------------------------------------------------------------
    # 连续谱计算 — compute_fwm_spectrum_conti (统一方向)
    # -------------------------------------------------------------------------

    def compute_fwm_spectrum_conti(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str = "forward",
        L_arr: np.ndarray | None = None,
    ):
        """计算 FWM 噪声 PSD G_fwm(f) [W/Hz]，在 f_grid 每个频率点评估。

        direction: "forward" | "backward" | "both"
        - "forward" / "backward": 返回 ndarray, shape (N_f,) 或 (N_f, N_L)
        - "both": 返回 (fwd, bwd) tuple

        GPU 加速：当 GPU 可用时自动切换至 GPU 批量计算。
        """
        _, is_gpu = _get_gpu_module()
        return self._compute_fwm_spectrum_conti_impl(
            fiber, wdm_grid, f_grid, direction=direction, L_arr=L_arr,
            use_gpu=is_gpu,
        )

    def _compute_fwm_spectrum_conti_impl(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        direction: str,
        L_arr: np.ndarray | None,
        use_gpu: bool,
    ):
        """Unified FWM spectrum computation (CPU or GPU).

        direction: "forward" | "backward" | "both"
        """
        do_fwd = direction in ("forward", "both")
        do_bwd = direction in ("backward", "both")

        if use_gpu:
            from qkd_sim.utils.gpu_utils import to_host
            xp, _ = _get_gpu_module()
        else:
            xp = np
            to_host = lambda a: np.asarray(a)

        f_grid_cpu = np.asarray(f_grid, dtype=np.float64)
        self._validate_frequency_grid(f_grid_cpu)

        df = float(np.mean(np.diff(f_grid_cpu)))
        G_tx_cpu = self._build_total_classical_psd(wdm_grid, f_grid_cpu, df)
        alpha_grid_cpu = np.asarray(fiber.get_loss_at_freq(f_grid_cpu), dtype=np.float64)

        G_min_cpu = G_tx_cpu.max() * 10 ** (self._active_threshold_db / 10.0)
        active_cpu = G_tx_cpu > G_min_cpu
        n_f = f_grid_cpu.size

        def _zeros(shape):
            return np.zeros(shape, dtype=np.float64)

        if not np.any(active_cpu):
            if L_arr is None:
                z = _zeros(n_f)
                return (z, z.copy()) if direction == "both" else z
            L_values = np.asarray(L_arr, dtype=np.float64).reshape(-1)
            z = _zeros((n_f, L_values.size))
            return (z, z.copy()) if direction == "both" else z

        if use_gpu:
            f_grid = xp.asarray(f_grid_cpu, dtype=xp.float64)
            G_tx = xp.asarray(G_tx_cpu, dtype=xp.float64)
            alpha_grid = xp.asarray(alpha_grid_cpu, dtype=xp.float64)
            active = xp.asarray(active_cpu)
        else:
            f_grid = f_grid_cpu
            G_tx = G_tx_cpu
            alpha_grid = alpha_grid_cpu
            active = active_cpu

        f_active = f_grid[active]
        G_active = G_tx[active]
        alpha_active = alpha_grid[active]
        active_idx_cpu = np.flatnonzero(active_cpu).astype(np.int64)
        active_idx = xp.asarray(active_idx_cpu, dtype=xp.int64) if use_gpu else active_idx_cpu

        Fi, Fj = xp.meshgrid(f_active, f_active, indexing="ij")
        Gi, Gj = xp.meshgrid(G_active, G_active, indexing="ij")
        alpha_i, alpha_j = xp.meshgrid(alpha_active, alpha_active, indexing="ij")
        D = xp.where(xp.abs(Fi - Fj) <= 0.5 * df, 3.0, 6.0)
        idx_i, idx_j = xp.meshgrid(active_idx, active_idx, indexing="ij")
        Fk_idx_ij = idx_i + idx_j

        N_a = Fi.shape[0]
        N_pairs = N_a * N_a
        N_L_for_chunk = int(np.asarray(L_arr).size) if L_arr is not None else 1

        # Dynamic chunk_size
        if use_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                free_bytes, total_bytes = cp.cuda.Device().mem_info
                min_budget = int(total_bytes * 0.35)
                budget_bytes = max(int(free_bytes * 0.4), min_budget)
                bytes_per_chunk_unit = N_a * N_a * max(N_L_for_chunk, 1) * 8 * 6
                chunk_size = max(1, min(50, int(budget_bytes / max(bytes_per_chunk_unit, 1))))
            except Exception:
                chunk_size = 10 if N_pairs * N_a > 50_000_000 else 50
        else:
            chunk_size = 10 if N_pairs * N_a > 50_000_000 else 50

        if _DEBUG_MODE:
            N_L_diag = N_L_for_chunk
            gpu_tag = "gpu" if use_gpu else "cpu"
            print(
                f"[fwm-{gpu_tag}] {direction} N_active={N_a}/{n_f} "
                f"({100.0 * N_a / max(n_f, 1):.1f}%) "
                f"N_L={N_L_diag} chunk_size={chunk_size}"
            )

        D_c = float(fiber._D_c)
        D_slope = float(fiber._D_slope)
        gamma = float(fiber.gamma)
        rayleigh_coeff = float(fiber.rayleigh_coeff)
        L_fiber = float(fiber.L)
        L_values_arr = (
            xp.asarray(L_arr, dtype=xp.float64).reshape(-1) if L_arr is not None else None
        )

        Fi_f = Fi.astype(xp.float64)
        Fj_f = Fj.astype(xp.float64)
        Gi_f = Gi.astype(xp.float64)
        Gj_f = Gj.astype(xp.float64)
        alpha_i_f = alpha_i.astype(xp.float64)
        alpha_j_f = alpha_j.astype(xp.float64)
        D_f = D.astype(xp.float64)
        pair_weight_f = ((D_f ** 2) * Gi_f * Gj_f).astype(xp.float64)

        pump_G_f = G_tx.astype(xp.float64)
        alpha_grid_f = alpha_grid.astype(xp.float64)

        # Valid f1 range: beyond this, Fk = Fi+Fj-f1 can never fall on a pump bin
        f1_lo = 2.0 * float(xp.min(f_active)) - float(xp.max(f_active))
        f1_hi = 2.0 * float(xp.max(f_active)) - float(xp.min(f_active))
        idx_lo = max(0, int(xp.searchsorted(f_grid, f1_lo)))
        idx_hi = min(n_f, int(xp.searchsorted(f_grid, f1_hi, side="right")))

        # ---- inner _batch_fwm: shared prep + direction-branched integration ----
        def _batch_fwm(chunk_start: int, chunk_end: int):
            n_chunk = chunk_end - chunk_start
            alpha1_chunk = alpha_grid_f[chunk_start:chunk_end]

            # Fk = Fi + Fj - f1  →  (n_chunk, N_a, N_a)
            f1_idx_chunk = xp.arange(chunk_start, chunk_end, dtype=xp.int64)
            k_idx = Fk_idx_ij[xp.newaxis, :, :] - f1_idx_chunk[:, xp.newaxis, xp.newaxis]
            in_grid = (k_idx >= 0) & (k_idx < n_f)
            k_idx_safe = xp.clip(k_idx, 0, n_f - 1)

            Gk = pump_G_f[k_idx_safe].astype(xp.float64)
            alpha_k = alpha_grid_f[k_idx_safe].astype(xp.float64)

            f2_3d = f_grid[k_idx_safe].astype(xp.float64)
            f3_3d = xp.broadcast_to(Fi_f[xp.newaxis, :, :], (n_chunk, N_a, N_a))
            f4_3d = xp.broadcast_to(Fj_f[xp.newaxis, :, :], (n_chunk, N_a, N_a))
            delta_beta_3d = _phase_mismatch(f2_3d, f3_3d, f4_3d, D_c, D_slope, xp=xp)

            valid_mask = in_grid & (Gk > 0.0)

            da_3d = (
                alpha_k
                + xp.broadcast_to(alpha_i_f, (n_chunk, N_a, N_a))
                + xp.broadcast_to(alpha_j_f, (n_chunk, N_a, N_a))
                - alpha1_chunk[:, xp.newaxis, xp.newaxis]
            )

            pair_weight_3d = pair_weight_f[xp.newaxis, :, :]

            if L_values_arr is None:
                # ---- scalar L ----
                fwd_res = None
                bwd_res = None

                if do_fwd:
                    eta = _fwm_coefficient(da_3d, delta_beta_3d, L_fiber, xp=xp)
                    eta_m = xp.where(valid_mask, eta, 0.0)
                    exp_a1 = xp.exp(-alpha1_chunk * L_fiber)
                    integrand = pair_weight_3d * eta_m * Gk
                    integral = xp.sum(integrand, axis=(1, 2)) * df * df
                    fwd_res = (gamma ** 2 / 9.0) * exp_a1 * integral

                if do_bwd:
                    F_L = _F_antiderivative(
                        xp.full_like(da_3d, L_fiber), 0.0, alpha1_chunk,
                        da_3d, delta_beta_3d, L_fiber, xp=xp,
                    )
                    F_0 = _F_antiderivative(
                        xp.zeros_like(da_3d), 0.0, alpha1_chunk,
                        da_3d, delta_beta_3d, L_fiber, xp=xp,
                    )
                    diff = xp.where(valid_mask, F_L - F_0, 0.0)
                    integrand = pair_weight_3d * Gk * diff
                    integral = xp.sum(integrand, axis=(1, 2)) * df * df
                    bwd_res = rayleigh_coeff * (gamma ** 2 / 9.0) * integral

                if direction == "both":
                    return fwd_res, bwd_res
                return fwd_res if do_fwd else bwd_res

            else:
                # ---- vectorized L ----
                N_L = L_values_arr.shape[0]
                fwd_res = None
                bwd_res = None

                Gk_4d = Gk[:, :, :, xp.newaxis]
                da_4d = da_3d[:, :, :, xp.newaxis]
                db_4d = delta_beta_3d[:, :, :, xp.newaxis]
                pair_weight_4d = pair_weight_f[xp.newaxis, :, :, xp.newaxis]

                if do_fwd:
                    eta_4d = _fwm_coefficient(da_4d, db_4d, L_values_arr, xp=xp)
                    eta_4d_m = xp.where(valid_mask[:, :, :, xp.newaxis], eta_4d, 0.0)
                    integrand = pair_weight_4d * eta_4d_m * Gk_4d
                    integral = xp.sum(integrand, axis=(1, 2)) * df * df
                    exp_f = xp.exp(-alpha1_chunk[:, xp.newaxis] * L_values_arr[xp.newaxis, :])
                    fwd_res = (gamma ** 2 / 9.0) * exp_f * integral

                if do_bwd:
                    a1_4d = alpha1_chunk[:, xp.newaxis, xp.newaxis, xp.newaxis]
                    a1_bc = xp.broadcast_to(a1_4d, (n_chunk, N_a, N_a, N_L))
                    F_L_4d = _F_antiderivative(
                        L_values_arr, 0.0, a1_bc, da_4d, db_4d, L_values_arr, xp=xp,
                    )
                    F_0_4d = _F_antiderivative(
                        0.0, 0.0, a1_bc, da_4d, db_4d, L_values_arr, xp=xp,
                    )
                    diff = F_L_4d - F_0_4d
                    diff_m = xp.where(valid_mask[:, :, :, xp.newaxis], diff, 0.0)
                    integrand = pair_weight_4d * Gk_4d * diff_m
                    integral = xp.sum(integrand, axis=(1, 2)) * df * df
                    bwd_res = rayleigh_coeff * (gamma ** 2 / 9.0) * integral

                if direction == "both":
                    return fwd_res, bwd_res
                return fwd_res if do_fwd else bwd_res

        # ---- assemble results in chunks ----
        if direction == "both":
            if L_arr is None:
                out_fwd = _zeros(n_f)
                out_bwd = _zeros(n_f)
                for start in range(idx_lo, idx_hi, chunk_size):
                    end = min(start + chunk_size, idx_hi)
                    fwd_c, bwd_c = _batch_fwm(start, end)
                    if use_gpu:
                        out_fwd[start:end] = to_host(fwd_c)
                        out_bwd[start:end] = to_host(bwd_c)
                    else:
                        out_fwd[start:end] = fwd_c
                        out_bwd[start:end] = bwd_c
                # Fill inactive f1 range with zeros (already zero from _zeros)
                return out_fwd, out_bwd
            else:
                N_L = L_values_arr.shape[0]
                out_fwd = _zeros((n_f, N_L))
                out_bwd = _zeros((n_f, N_L))
                for start in range(idx_lo, idx_hi, chunk_size):
                    end = min(start + chunk_size, idx_hi)
                    fwd_c, bwd_c = _batch_fwm(start, end)
                    if use_gpu:
                        out_fwd[start:end, :] = to_host(fwd_c)
                        out_bwd[start:end, :] = to_host(bwd_c)
                    else:
                        out_fwd[start:end, :] = fwd_c
                        out_bwd[start:end, :] = bwd_c
                return out_fwd, out_bwd

        # Single direction
        if L_arr is None:
            out = _zeros(n_f)
            for start in range(idx_lo, idx_hi, chunk_size):
                end = min(start + chunk_size, idx_hi)
                chunk_res = _batch_fwm(start, end)
                if use_gpu:
                    out[start:end] = to_host(chunk_res)
                else:
                    out[start:end] = chunk_res
            return out

        N_L = L_values_arr.shape[0]
        out = _zeros((n_f, N_L))
        for start in range(idx_lo, idx_hi, chunk_size):
            end = min(start + chunk_size, idx_hi)
            chunk_res = _batch_fwm(start, end)
            if use_gpu:
                out[start:end, :] = to_host(chunk_res)
            else:
                out[start:end, :] = chunk_res
        return out

    def compute_fwm_spectrum_conti_pair(
        self,
        fiber: Fiber,
        wdm_grid: WDMGrid,
        f_grid: np.ndarray,
        L_arr: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute FWM forward and backward PSD simultaneously (thin wrapper).

        Equivalent to compute_fwm_spectrum_conti(..., direction="both").
        """
        return self.compute_fwm_spectrum_conti(
            fiber, wdm_grid, f_grid, direction="both", L_arr=L_arr,
        )
