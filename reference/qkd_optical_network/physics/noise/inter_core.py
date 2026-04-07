"""
芯间噪声模块（预留接口）

当前版本仅实现单芯光纤的芯内噪声（intra-core noise）。
本模块用于未来扩展多芯光纤的芯间噪声（inter-core noise）：
- 芯间 FWM（Inter-core Four-Wave Mixing）
- 芯间拉曼散射（Inter-core Raman Scattering）
- 芯间串扰（Inter-core Crosstalk）

References
----------
- MCF.py 中的芯间 FWM 和芯间拉曼实现
- 多芯光纤耦合模理论
"""

from typing import List, Optional
import numpy as np

from physics.signal import WDMChannel
from physics.fiber import Fiber, MultiCoreFiber


def compute_inter_core_fwm_noise(
    fiber: MultiCoreFiber,
    channels: List[WDMChannel],
    core_i: int,
    core_j: int,
    compute_at_length: bool = True
) -> np.ndarray:
    """
    计算芯间四波混频噪声（预留接口）

    Parameters
    ----------
    fiber : MultiCoreFiber
        多芯光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    core_i, core_j : int
        相互作用的两个纤芯编号
    compute_at_length : bool, optional
        如果 True，只计算光纤末端 (z=L) 的噪声
        如果 False，返回沿光纤的噪声分布

    Returns
    -------
    noise_powers : np.ndarray
        各信道的芯间 FWM 噪声功率 [W]

    Raises
    ------
    NotImplementedError
        当前版本未实现

    Notes
    -----
    未来实现时参考：
    - MCF.py 中的 get_intercore_four_wave_mixing 方法
    - 芯间耦合系数 h_mn 的影响
    """
    raise NotImplementedError(
        "Inter-core FWM noise calculation is not implemented yet. "
        "This is a placeholder for future multi-core fiber support."
    )


def compute_inter_core_raman_noise(
    fiber: MultiCoreFiber,
    channels: List[WDMChannel],
    core_i: int,
    core_j: int,
    compute_at_length: bool = True
) -> np.ndarray:
    """
    计算芯间拉曼散射噪声（预留接口）

    Parameters
    ----------
    fiber : MultiCoreFiber
        多芯光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    core_i, core_j : int
        相互作用的两个纤芯编号
    compute_at_length : bool, optional
        如果 True，只计算光纤末端 (z=L) 的噪声

    Returns
    -------
    noise_powers : np.ndarray
        各信道的芯间拉曼噪声功率 [W]

    Raises
    ------
    NotImplementedError
        当前版本未实现

    Notes
    -----
    未来实现时参考：
    - MCF.py 中的 get_inter_forward_raman_scatter 等方法
    - 芯间耦合系数 h_mn 的影响
    """
    raise NotImplementedError(
        "Inter-core Raman noise calculation is not implemented yet. "
        "This is a placeholder for future multi-core fiber support."
    )


def compute_inter_core_crosstalk(
    fiber: MultiCoreFiber,
    channels: List[WDMChannel],
    length: Optional[float] = None
) -> np.ndarray:
    """
    计算芯间串扰（预留接口）

    Parameters
    ----------
    fiber : MultiCoreFiber
        多芯光纤对象
    channels : List[WDMChannel]
        WDM 信道列表
    length : float, optional
        光纤长度 [m]。如果为 None，使用 fiber.length。

    Returns
    -------
    xt_matrix : np.ndarray
        芯间串扰矩阵 [dB]，shape: (n_cores, n_cores)

    Raises
    ------
    NotImplementedError
        当前版本未实现
    """
    raise NotImplementedError(
        "Inter-core crosstalk calculation is not implemented yet. "
        "This is a placeholder for future multi-core fiber support."
    )
