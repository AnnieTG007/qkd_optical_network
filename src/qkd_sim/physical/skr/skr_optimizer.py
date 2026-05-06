"""SKR 参数优化模块。

基于 scipy.optimize.minimize (Nelder-Mead) 在每个距离点优化 4 个自由参数：
  mu_decoy, mu_signal, p_signal, P_X_alice
通过顺序热启动（前一距离的最优解作为下一距离的初始猜测）加速收敛。

不修改 skr_decoy_bb84.py，通过 dataclasses.replace 注入优化参数后调用 strict_finite_key_rate。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from qkd_sim.config.schema import FiberConfig, SKRConfig
from qkd_sim.physical.skr.skr_decoy_bb84 import strict_finite_key_rate

# 优化向量 x 的索引
IDX_MU_DECOY = 0
IDX_MU_SIGNAL = 1
IDX_P_SIGNAL = 2
IDX_PX_ALICE = 3

# 参数搜索边界
BOUNDS = [
    (0.001, 0.5),    # mu_decoy
    (0.01, 2.0),     # mu_signal
    (0.5, 0.99),     # p_signal
    (0.51, 0.999),   # P_X_alice
]

# 默认初始猜测 [mu_decoy, mu_signal, p_signal, P_X_alice]
DEFAULT_X0 = [0.1, 0.5, 0.7, 0.9]

# mu_signal > mu_decoy 的最小裕度（与 GitHub qkdsimulator.py 一致）
EPSILON_FEASIBLE = 0.0001

# 不可行区域惩罚值
PENALTY = 1e30


@dataclass
class OptimizationResult:
    """单距离优化结果。"""
    distance_m: float
    optimal_skr_bps: float
    optimal_params: dict  # mu_signal, mu_decoy, p_signal, P_X_alice, p_decoy, P_X_bob
    qber: float
    success: bool
    nfev: int


class SKROptimizer:
    """SKR 参数优化器。

    在每个距离点用 Nelder-Mead 优化 decoy-state BB84 协议参数，
    使严格有限长密钥率最大化。
    """

    def __init__(
        self,
        fiber_cfg: FiberConfig,
        base_skr_cfg: SKRConfig,
        p_noise: float = 0.0,
    ) -> None:
        self.fiber_cfg = fiber_cfg
        self.base_skr_cfg = base_skr_cfg
        self.p_noise = p_noise

    # ------------------------------------------------------------------
    # 内部：目标函数
    # ------------------------------------------------------------------

    def _decode_params(self, x: np.ndarray) -> tuple[float, float, float, float]:
        mu_decoy = float(x[IDX_MU_DECOY])
        mu_signal = float(x[IDX_MU_SIGNAL])
        p_signal = float(x[IDX_P_SIGNAL])
        px_alice = float(x[IDX_PX_ALICE])
        return mu_decoy, mu_signal, p_signal, px_alice

    def _objective(self, x: np.ndarray, distance_m: float) -> float:
        mu_decoy, mu_signal, p_signal, px_alice = self._decode_params(x)

        # 约束：mu_signal > mu_decoy + margin
        if mu_signal <= mu_decoy + EPSILON_FEASIBLE:
            return PENALTY

        # 边界检查（Nelder-Mead 可能试探边界外）
        for (lo, hi), val in zip(BOUNDS, x):
            if val < lo - 1e-12 or val > hi + 1e-12:
                return PENALTY

        try:
            opt_cfg = replace(
                self.base_skr_cfg,
                mu_signal=mu_signal,
                mu_decoy=mu_decoy,
                p_signal=p_signal,
                p_decoy=1.0 - p_signal,
                P_X_alice=px_alice,
                P_X_bob=px_alice,
            )
            skr_bps, _, qber = strict_finite_key_rate(
                distance_m, self.fiber_cfg, opt_cfg, self.p_noise, optimize_params=False
            )
            if skr_bps <= 0.0 or not math.isfinite(skr_bps):
                return PENALTY
            return -skr_bps
        except (ValueError, ZeroDivisionError):
            return PENALTY

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def optimize_distance(
        self, distance_m: float, x0: list[float] | None = None
    ) -> OptimizationResult:
        """优化单个距离点的 SKR 参数。

        Parameters
        ----------
        distance_m : float
            光纤距离 [m]
        x0 : list[float] | None
            初始猜测 [mu_decoy, mu_signal, p_signal, P_X_alice]，
            None 时使用 DEFAULT_X0

        Returns
        -------
        OptimizationResult
        """
        if x0 is None:
            x0 = list(DEFAULT_X0)

        result = minimize(
            fun=lambda x: self._objective(x, distance_m),
            x0=np.array(x0, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 100000, "adaptive": True, "fatol": 1e-8},
        )

        mu_decoy, mu_signal, p_signal, px_alice = self._decode_params(result.x)
        p_decoy = 1.0 - p_signal

        # 如果优化失败（结果仍为惩罚值），回退到基础配置
        if result.fun >= PENALTY * 0.5:
            return OptimizationResult(
                distance_m=distance_m,
                optimal_skr_bps=0.0,
                optimal_params={
                    "mu_signal": self.base_skr_cfg.mu_signal,
                    "mu_decoy": self.base_skr_cfg.mu_decoy,
                    "p_signal": self.base_skr_cfg.p_signal,
                    "p_decoy": self.base_skr_cfg.p_decoy,
                    "P_X_alice": self.base_skr_cfg.P_X_alice,
                    "P_X_bob": self.base_skr_cfg.P_X_bob,
                },
                qber=0.0,
                success=False,
                nfev=result.nfev,
            )

        # 用最优参数重新计算 SKR 和 QBER
        opt_cfg = replace(
            self.base_skr_cfg,
            mu_signal=mu_signal,
            mu_decoy=mu_decoy,
            p_signal=p_signal,
            p_decoy=p_decoy,
            P_X_alice=px_alice,
            P_X_bob=px_alice,
        )
        skr_bps, _, qber = strict_finite_key_rate(
            distance_m, self.fiber_cfg, opt_cfg, self.p_noise, optimize_params=False
        )

        return OptimizationResult(
            distance_m=distance_m,
            optimal_skr_bps=max(skr_bps, 0.0),
            optimal_params={
                "mu_signal": mu_signal,
                "mu_decoy": mu_decoy,
                "p_signal": p_signal,
                "p_decoy": p_decoy,
                "P_X_alice": px_alice,
                "P_X_bob": px_alice,
            },
            qber=qber,
            success=result.success or skr_bps > 0.0,
            nfev=result.nfev,
        )

    def optimize_over_distances(
        self,
        distances_m: list[float],
        x0_initial: list[float] | None = None,
    ) -> list[OptimizationResult]:
        """批量优化多个距离点，自动热启动。

        Parameters
        ----------
        distances_m : list[float]
            光纤距离列表 [m]，须按升序排列
        x0_initial : list[float] | None
            首个距离的初始猜测，None 时使用 DEFAULT_X0

        Returns
        -------
        list[OptimizationResult]
        """
        if x0_initial is None:
            x0_initial = list(DEFAULT_X0)

        results: list[OptimizationResult] = []
        x0 = list(x0_initial)

        for d in distances_m:
            result = self.optimize_distance(d, x0)
            results.append(result)

            # 热启动：用当前最优参数作为下一距离的初始猜测
            if result.optimal_skr_bps > 0.0:
                p = result.optimal_params
                x0 = [p["mu_decoy"], p["mu_signal"], p["p_signal"], p["P_X_alice"]]

        return results
