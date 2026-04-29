"""噪声计算统一入口。

提供 compute_noise() 函数，根据 noise_type 调度对应求解器，
返回统一格式的噪声功率字典。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import WDMGrid, validate_uniform_frequency_grid
from qkd_sim.physical.noise.sprs_solver import DiscreteSPRSSolver
from qkd_sim.physical.noise.fwm_solver import DiscreteFWMSolver
from qkd_sim.physical.noise.gn_solver import GNModelSolver


NoiseType = Literal["sprs", "fwm", "nli", "all"]


def compute_noise(
    noise_type: NoiseType,
    fiber: Fiber,
    wdm_grid: WDMGrid,
    sprs_solver: DiscreteSPRSSolver | None = None,
    fwm_solver: DiscreteFWMSolver | None = None,
    continuous: bool = False,
    gn_solver: GNModelSolver | None = None,
) -> dict[str, np.ndarray]:
    """计算量子信道（或经典信道 NLI）噪声功率。

    Parameters
    ----------
    noise_type : {"sprs", "fwm", "nli", "all"}
        噪声类型：
        - "sprs"：仅计算自发拉曼散射噪声（量子信道受泵浦）
        - "fwm"：仅计算四波混频噪声（量子信道受泵浦）
        - "nli"：计算经典信道的 GN-model 非线性干扰噪声
        - "all"：计算全部量子信道噪声（不含 NLI）
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
        True：使用连续模型（要求 wdm_grid.f_grid 非空）。
        注意：noise_type="nli" 时忽略此参数，始终使用连续模型。
    gn_solver : GNModelSolver or None
        GN-model NLI 求解器。None 时使用默认参数构造。
        仅在 noise_type="nli" 时生效。

    Returns
    -------
    dict
        键值对：
        - "sprs_fwd"  : ndarray, shape (N_q,)  前向 SpRS 噪声 [W]（noise_type in {"sprs","all"}）
        - "sprs_bwd"  : ndarray, shape (N_q,)  后向 SpRS 噪声 [W]（noise_type in {"sprs","all"}）
        - "fwm_fwd"   : ndarray, shape (N_q,)  前向 FWM 噪声 [W]（noise_type in {"fwm","all"}）
        - "fwm_bwd"   : ndarray, shape (N_q,)  后向 FWM 噪声 [W]（noise_type in {"fwm","all"}）
        - "nli_fwd"   : ndarray, shape (N_c,)  前向 NLI 噪声 [W]（noise_type="nli"）
        - "nli_bwd"   : ndarray, shape (N_c,)  后向 NLI 噪声 [W]（noise_type="nli"）

    Raises
    ------
    ValueError
        noise_type 不在允许值内时。
    ValueError
        continuous=True 但 wdm_grid.f_grid 未设置时（sprs/fwm/all 模式）。
    ValueError
        noise_type="nli" 但 wdm_grid.f_grid 未设置时。
    """
    if noise_type not in ("sprs", "fwm", "nli", "all"):
        raise ValueError(f"noise_type 须为 'sprs'/'fwm'/'nli'/'all'，得到 '{noise_type}'")

    if noise_type == "nli":
        if wdm_grid.f_grid is None:
            raise ValueError(
                "noise_type='nli' 时需要在 wdm_grid.f_grid 提供频率网格"
            )
        solver_gn = gn_solver if gn_solver is not None else GNModelSolver()
        result_gn = solver_gn.compute_nli_per_channel(fiber, wdm_grid, wdm_grid.f_grid)
        return result_gn

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


def _resolve_frequency_grid(
    wdm_grid: WDMGrid,
    f_grid: np.ndarray | None,
) -> np.ndarray:
    """解析得到评估频率网格。"""
    if f_grid is not None:
        return np.asarray(f_grid, dtype=np.float64)
    if wdm_grid.f_grid is None:
        raise ValueError("f_grid must be provided, or wdm_grid.f_grid must be set")
    return np.asarray(wdm_grid.f_grid, dtype=np.float64)


def compute_noise_spectrum(
    noise_type: NoiseType,
    fiber: Fiber,
    wdm_grid: WDMGrid,
    f_grid: np.ndarray | None = None,
    sprs_solver: DiscreteSPRSSolver | None = None,
    fwm_solver: DiscreteFWMSolver | None = None,
    gn_solver: GNModelSolver | None = None,
) -> dict[str, np.ndarray]:
    """计算噪声功率谱密度 G_noise(f) [W/Hz]，在 f_grid 每个频率点评估。

    与 compute_noise() 的区别：
    - compute_noise()：返回每个量子信道中心频率处的积分噪声功率 [W]（标量 per channel）
    - compute_noise_spectrum()：返回 f_grid 上每个频率点的噪声 PSD [W/Hz]（连续曲线）

    仅用于连续模型（绘制噪声功率谱曲线）。

    Parameters
    ----------
    noise_type : {"sprs", "fwm", "nli", "all"}
        噪声类型：
        - "sprs"/"fwm"/"all"：量子信道噪声 PSD
        - "nli"：经典信道 GN-model NLI PSD
    fiber : Fiber
        光纤物理参数
    wdm_grid : WDMGrid
        WDM 信道网格
    f_grid : ndarray or None
        输出频率网格 [Hz]。None 时使用 wdm_grid.f_grid。
    sprs_solver : DiscreteSPRSSolver or None
    fwm_solver : DiscreteFWMSolver or None
    gn_solver : GNModelSolver or None

    Returns
    -------
    dict[str, ndarray]
        - "sprs_fwd" : ndarray, shape (N_f,)  前向 SpRS PSD [W/Hz]
        - "sprs_bwd" : ndarray, shape (N_f,)  后向 SpRS PSD [W/Hz]
        - "fwm_fwd"  : ndarray, shape (N_f,)  前向 FWM PSD [W/Hz]
        - "fwm_bwd"  : ndarray, shape (N_f,)  后向 FWM PSD [W/Hz]
        - "sprs"     : ndarray, shape (N_f,)  SpRS 总 PSD [W/Hz]
        - "fwm"      : ndarray, shape (N_f,)  FWM 总 PSD [W/Hz]
        - "nli"      : ndarray, shape (N_f,)  经典信道 NLI PSD [W/Hz]（noise_type="nli"）
        - "total"    : ndarray, shape (N_f,)  总噪声 PSD [W/Hz]
    """
    if noise_type not in ("sprs", "fwm", "nli", "all"):
        raise ValueError(
            f"noise_type must be 'sprs'/'fwm'/'nli'/'all', got {noise_type!r}"
        )

    f_eval = _resolve_frequency_grid(wdm_grid, f_grid)
    validate_uniform_frequency_grid(f_eval)

    result: dict[str, np.ndarray] = {}

    if noise_type == "nli":
        solver_gn = gn_solver if gn_solver is not None else GNModelSolver()
        result["nli_fwd"] = solver_gn.compute_nli_psd(fiber, wdm_grid, f_eval)
        result["nli_bwd"] = solver_gn.compute_nli_psd_backward(fiber, wdm_grid, f_eval)
        result["nli"] = result["nli_fwd"] + result["nli_bwd"]
        result["total"] = result["nli"]
        return result

    if noise_type in ("sprs", "all"):
        solver = sprs_solver if sprs_solver is not None else DiscreteSPRSSolver()
        result["sprs_fwd"], result["sprs_bwd"] = solver.compute_sprs_spectrum_conti(
            fiber, wdm_grid, f_eval, direction="both"
        )
        result["sprs"] = result["sprs_fwd"] + result["sprs_bwd"]

    if noise_type in ("fwm", "all"):
        solver_fwm = fwm_solver if fwm_solver is not None else DiscreteFWMSolver()
        result["fwm_fwd"], result["fwm_bwd"] = solver_fwm.compute_fwm_spectrum_conti(
            fiber, wdm_grid, f_eval, direction="both"
        )
        result["fwm"] = result["fwm_fwd"] + result["fwm_bwd"]

    if noise_type == "sprs":
        result["fwm"] = np.zeros_like(f_eval, dtype=np.float64)
    elif noise_type == "fwm":
        result["sprs"] = np.zeros_like(f_eval, dtype=np.float64)

    result["total"] = result.get("sprs", np.zeros_like(f_eval)) + result.get("fwm", np.zeros_like(f_eval))
    return result
