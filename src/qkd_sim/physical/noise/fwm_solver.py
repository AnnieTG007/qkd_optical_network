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
"""

from __future__ import annotations

import numpy as np

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import WDMGrid
from qkd_sim.physical.noise.base import NoiseSolver


def _fwm_efficiency(
    delta_alpha: np.ndarray,
    delta_beta: np.ndarray,
    L: float,
) -> np.ndarray:
    """FWM 效率因子 η。

    公式 2.2.2 (formulas_fwm.md):
        η = [exp(-Δα·L) - 2·exp(-Δα·L/2)·cos(Δβ·L) + 1]
            / [(Δα)²/4 + (Δβ)²]

    Parameters
    ----------
    delta_alpha : ndarray
        衰减失配 Δα = α₃+α₄+α₂-α₁ [1/m]
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
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    denom: np.ndarray,
) -> np.ndarray:
    """后向 FWM 积分原函数 F(l)。

    公式 2.2.6 (formulas_fwm.md):
        F(l) = exp(α₁·z_obs) / denom × [-exp(-A·l)/A - exp(-B·l)/B² - exp(-C·l)/C]

    其中辅助变量：
        A = Δα + 2·α₁
        B = Δα/2 + 2·α₁   （注：分母中为 B²，源自二次项积分，见公式推导）
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
    A, B, C : ndarray
        辅助变量（与 l 广播兼容）
    denom : ndarray
        (Δα)²/4 + (Δβ)² [1/m²]

    Returns
    -------
    ndarray
        F(l) 的值
    """
    exp_z = np.exp(alpha1 * z_obs)
    term_A = -np.exp(-A * l) / A
    term_B = -np.exp(-B * l) / (B ** 2)  # 注：B² 来自二次项的部分分数展开
    term_C = -np.exp(-C * l) / C
    return exp_z / denom * (term_A + term_B + term_C)


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

    def _get_valid_combinations(
        self,
        n1: int,
        idx_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """枚举对量子信道 n₁ 产生 FWM 噪声的有效经典信道三元组 (n₂, n₃, n₄)。

        频率匹配：n₃ + n₄ - n₂ = n₁，即 n₂ = n₃ + n₄ - n₁。
        约束：n₂ ∈ idx_c，n₂ ≠ n₃，n₂ ≠ n₄。

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
        # meshgrid 生成所有 (n₃, n₄) 组合，shape (N_c, N_c)
        n3_grid, n4_grid = np.meshgrid(idx_c, idx_c, indexing="ij")
        n2_grid = n3_grid + n4_grid - n1

        # 有效性掩码：n₂ 是经典信道，且 n₂ ≠ n₃，n₂ ≠ n₄
        in_classical = np.isin(n2_grid, idx_c)
        not_spm_xpm = (n2_grid != n3_grid) & (n2_grid != n4_grid)
        valid = in_classical & not_spm_xpm

        return n2_grid[valid], n3_grid[valid], n4_grid[valid]

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

        # 衰减失配 Δα = α₃+α₄+α₂-α₁（C波段近似：各信道 α 相同，Δα = 2α）
        alpha = fiber.alpha
        delta_alpha = 2.0 * alpha * np.ones(f2.shape)  # Δα ≈ 2α

        # 相位失配 Δβ（调用 Fiber 接口，公式 2.2.3）
        delta_beta = fiber.get_phase_mismatch(f2=f2, f3=f3, f4=f4)

        L = fiber.L
        gamma = fiber.gamma

        if direction == "forward":
            # 公式 2.2.1
            eta = _fwm_efficiency(delta_alpha, delta_beta, L)
            contributions = (
                np.exp(-alpha * L) * (gamma ** 2 / 9.0)
                * D ** 2 * eta * P2 * P3 * P4
            )
            return float(contributions.sum())

        else:  # backward
            # 公式 2.2.4-2.2.7：瑞利散射积分解析式
            # 辅助变量（公式 2.2.5）
            A = delta_alpha + 2.0 * alpha  # shape (N_valid,)
            B = delta_alpha / 2.0 + 2.0 * alpha
            C = 2.0 * alpha * np.ones_like(A)
            denom = delta_alpha ** 2 / 4.0 + delta_beta ** 2

            # z_obs = 0（光纤发射端），exp(α₁·0) = 1
            F_L = _F_antiderivative(
                l=np.full_like(A, L), z_obs=0.0,
                alpha1=alpha, A=A, B=B, C=C, denom=denom,
            )
            F_0 = _F_antiderivative(
                l=np.zeros_like(A), z_obs=0.0,
                alpha1=alpha, A=A, B=B, C=C, denom=denom,
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
        P_fwd = np.array([
            self._compute_noise_for_channel(
                f1=float(f_q[i]), n1=int(n_q[i]),
                idx_c=idx_c, f_all=f_all, P_all=P_all,
                fiber=fiber, direction="forward",
            )
            for i in range(len(f_q))
        ])
        assert P_fwd.shape == (len(f_q),)
        return P_fwd

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
        P_bwd = np.array([
            self._compute_noise_for_channel(
                f1=float(f_q[i]), n1=int(n_q[i]),
                idx_c=idx_c, f_all=f_all, P_all=P_all,
                fiber=fiber, direction="backward",
            )
            for i in range(len(f_q))
        ])
        assert P_bwd.shape == (len(f_q),)
        return P_bwd
