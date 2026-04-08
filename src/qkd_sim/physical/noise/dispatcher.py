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


def _map_channel_noise_to_psd(
    wdm_grid: WDMGrid,
    f_grid: np.ndarray,
    channel_noise: np.ndarray,
) -> np.ndarray:
    """将信道积分噪声（标量 per channel）映射到 f_grid 最近格点的 PSD。

    用于离散模型的噪声 PSD 谱：将信道中心频率处的积分噪声 per B_s
    转换为等效 PSD 放在最近格点。
    """
    df = validate_uniform_frequency_grid(f_grid)
    q_freqs = np.array(
        [ch.f_center for ch in wdm_grid.get_quantum_channels()], dtype=np.float64
    )
    out = np.zeros_like(f_grid, dtype=np.float64)
    for fq, pq in zip(q_freqs, channel_noise):
        idx = int(np.argmin(np.abs(f_grid - fq)))
        out[idx] += float(pq) / df
    return out


def compute_noise_spectrum(
    noise_type: NoiseType,
    fiber: Fiber,
    wdm_grid: WDMGrid,
    f_grid: np.ndarray | None = None,
    sprs_solver: DiscreteSPRSSolver | None = None,
    fwm_solver: DiscreteFWMSolver | None = None,
    continuous: bool = True,
) -> dict[str, np.ndarray]:
    """计算噪声功率谱密度 G_noise(f) [W/Hz]，在 f_grid 每个频率点评估。

    与 compute_noise() 的区别：
    - compute_noise()：返回每个量子信道中心频率处的积分噪声功率 [W]（标量 per channel）
    - compute_noise_spectrum()：返回 f_grid 上每个频率点的噪声 PSD [W/Hz]（连续曲线）

    用于绘制噪声功率谱（连续曲线），而非 BER/SNR 计算。

    Parameters
    ----------
    noise_type : {"sprs", "fwm", "all"}
        噪声类型
    fiber : Fiber
        光纤物理参数
    wdm_grid : WDMGrid
        WDM 信道网格
    f_grid : ndarray or None
        输出频率网格 [Hz]。None 时使用 wdm_grid.f_grid。
    sprs_solver : DiscreteSPRSSolver or None
    fwm_solver : DiscreteFWMSolver or None
    continuous : bool
        True（默认）：使用连续模型；
        False：使用离散模型（信道积分噪声映射到 f_grid 的 PSD）

    Returns
    -------
    dict[str, ndarray]
        - "sprs_fwd" : ndarray, shape (N_f,)  前向 SpRS PSD [W/Hz]
        - "sprs_bwd" : ndarray, shape (N_f,)  后向 SpRS PSD [W/Hz]
        - "fwm_fwd"  : ndarray, shape (N_f,)  前向 FWM PSD [W/Hz]
        - "fwm_bwd"  : ndarray, shape (N_f,)  后向 FWM PSD [W/Hz]
        - "sprs"     : ndarray, shape (N_f,)  SpRS 总 PSD [W/Hz]
        - "fwm"      : ndarray, shape (N_f,)  FWM 总 PSD [W/Hz]
        - "total"    : ndarray, shape (N_f,)  总噪声 PSD [W/Hz]
    """
    if noise_type not in ("sprs", "fwm", "all"):
        raise ValueError(f"noise_type must be 'sprs'/'fwm'/'all', got {noise_type!r}")

    f_eval = _resolve_frequency_grid(wdm_grid, f_grid)
    validate_uniform_frequency_grid(f_eval)

    result: dict[str, np.ndarray] = {}

    if noise_type in ("sprs", "all"):
        solver = sprs_solver if sprs_solver is not None else DiscreteSPRSSolver()
        if continuous:
            result["sprs_fwd"] = solver.compute_sprs_spectrum_conti(
                fiber, wdm_grid, f_eval, direction="forward"
            )
            result["sprs_bwd"] = solver.compute_sprs_spectrum_conti(
                fiber, wdm_grid, f_eval, direction="backward"
            )
        else:
            ch_noise = compute_noise(
                "sprs", fiber, wdm_grid, sprs_solver=solver, continuous=False
            )
            result["sprs_fwd"] = _map_channel_noise_to_psd(
                wdm_grid, f_eval, ch_noise["sprs_fwd"]
            )
            result["sprs_bwd"] = _map_channel_noise_to_psd(
                wdm_grid, f_eval, ch_noise["sprs_bwd"]
            )
        result["sprs"] = result["sprs_fwd"] + result["sprs_bwd"]

    if noise_type in ("fwm", "all"):
        solver_fwm = fwm_solver if fwm_solver is not None else DiscreteFWMSolver()
        if continuous:
            result["fwm_fwd"] = solver_fwm.compute_fwm_spectrum_conti(
                fiber, wdm_grid, f_eval, direction="forward"
            )
            result["fwm_bwd"] = solver_fwm.compute_fwm_spectrum_conti(
                fiber, wdm_grid, f_eval, direction="backward"
            )
        else:
            ch_noise = compute_noise(
                "fwm", fiber, wdm_grid, fwm_solver=solver_fwm, continuous=False
            )
            result["fwm_fwd"] = _map_channel_noise_to_psd(
                wdm_grid, f_eval, ch_noise["fwm_fwd"]
            )
            result["fwm_bwd"] = _map_channel_noise_to_psd(
                wdm_grid, f_eval, ch_noise["fwm_bwd"]
            )
        result["fwm"] = result["fwm_fwd"] + result["fwm_bwd"]

    if noise_type == "sprs":
        result["fwm"] = np.zeros_like(f_eval, dtype=np.float64)
    elif noise_type == "fwm":
        result["sprs"] = np.zeros_like(f_eval, dtype=np.float64)

    result["total"] = result["sprs"] + result["fwm"]
    return result
