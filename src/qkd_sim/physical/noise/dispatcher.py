"""噪声计算统一入口。

提供 compute_noise() 函数，根据 noise_type 调度对应求解器，
返回统一格式的噪声功率字典。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import WDMGrid
from qkd_sim.physical.noise.sprs_solver import DiscreteSPRSSolver
from qkd_sim.physical.noise.fwm_solver import DiscreteFWMSolver


NoiseType = Literal["sprs", "fwm", "all"]


def compute_noise(
    noise_type: NoiseType,
    fiber: Fiber,
    wdm_grid: WDMGrid,
    sprs_solver: DiscreteSPRSSolver | None = None,
    fwm_solver: DiscreteFWMSolver | None = None,
    continuous: bool = False,
) -> dict[str, np.ndarray]:
    """计算量子信道噪声功率。

    Parameters
    ----------
    noise_type : {"sprs", "fwm", "all"}
        噪声类型：
        - "sprs"：仅计算自发拉曼散射噪声
        - "fwm"：仅计算四波混频噪声
        - "all"：计算全部噪声
    fiber : Fiber
        光纤物理参数
    wdm_grid : WDMGrid
        WDM 信道网格（含经典/量子信道频率与功率）
    sprs_solver : DiscreteSPRSSolver or None
        SpRS 求解器实例。None 时使用默认参数构造。
    fwm_solver : DiscreteFWMSolver or None
        FWM 求解器实例。None 时使用默认参数构造。
    continuous : bool
        False（默认）：使用离散模型；
        True：使用连续模型（要求 wdm_grid.f_grid 非空）

    Returns
    -------
    dict
        键值对：
        - "sprs_fwd"  : ndarray, shape (N_q,)  前向 SpRS 噪声 [W]（noise_type in {"sprs","all"}）
        - "sprs_bwd"  : ndarray, shape (N_q,)  后向 SpRS 噪声 [W]（noise_type in {"sprs","all"}）
        - "fwm_fwd"   : ndarray, shape (N_q,)  前向 FWM 噪声 [W]（noise_type in {"fwm","all"}）
        - "fwm_bwd"   : ndarray, shape (N_q,)  后向 FWM 噪声 [W]（noise_type in {"fwm","all"}）

    Raises
    ------
    ValueError
        noise_type 不在允许值内时。
    ValueError
        continuous=True 但 wdm_grid.f_grid 未设置时。

    Examples
    --------
    >>> result = compute_noise("all", fiber, wdm_grid)
    >>> result["sprs_fwd"]  # shape (N_q,)
    >>> result["fwm_fwd"]   # shape (N_q,)
    """
    if noise_type not in ("sprs", "fwm", "all"):
        raise ValueError(f"noise_type 须为 'sprs'/'fwm'/'all'，得到 '{noise_type}'")

    if continuous and wdm_grid.f_grid is None:
        raise ValueError(
            "continuous=True 时需要在 wdm_grid.f_grid 提供频率网格"
        )

    result: dict[str, np.ndarray] = {}

    if noise_type in ("sprs", "all"):
        solver = sprs_solver if sprs_solver is not None else DiscreteSPRSSolver()
        if continuous:
            result["sprs_fwd"] = solver.compute_forward_conti(fiber, wdm_grid, wdm_grid.f_grid)
            result["sprs_bwd"] = solver.compute_backward_conti(fiber, wdm_grid, wdm_grid.f_grid)
        else:
            result["sprs_fwd"] = solver.compute_forward(fiber, wdm_grid)
            result["sprs_bwd"] = solver.compute_backward(fiber, wdm_grid)

    if noise_type in ("fwm", "all"):
        solver_fwm = fwm_solver if fwm_solver is not None else DiscreteFWMSolver()
        if continuous:
            result["fwm_fwd"] = solver_fwm.compute_forward_conti(fiber, wdm_grid, wdm_grid.f_grid)
            result["fwm_bwd"] = solver_fwm.compute_backward_conti(fiber, wdm_grid, wdm_grid.f_grid)
        else:
            result["fwm_fwd"] = solver_fwm.compute_forward(fiber, wdm_grid)
            result["fwm_bwd"] = solver_fwm.compute_backward(fiber, wdm_grid)

    return result
