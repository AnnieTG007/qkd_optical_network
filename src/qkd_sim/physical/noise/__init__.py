"""噪声求解器：FWM、SpRS、GN-model。

公开接口
--------
compute_noise       — 统一入口（dispatcher）
compute_noise_spectrum — 噪声 PSD 谱 dispatcher
DiscreteSPRSSolver  — 离散 SpRS 求解器
DiscreteFWMSolver   — 离散 FWM 求解器
GNModelSolver       — GN-model 经典信道 NLI 求解器
NoiseSolver         — 抽象基类（用于自定义求解器）
"""

from qkd_sim.physical.noise.base import NoiseSolver
from qkd_sim.physical.noise.sprs_solver import DiscreteSPRSSolver
from qkd_sim.physical.noise.fwm_solver import DiscreteFWMSolver
from qkd_sim.physical.noise.gn_solver import GNModelSolver
from qkd_sim.physical.noise.dispatcher import (
    compute_noise,
    compute_noise_spectrum,
    NoiseType,
)

__all__ = [
    "NoiseSolver",
    "DiscreteSPRSSolver",
    "DiscreteFWMSolver",
    "GNModelSolver",
    "compute_noise",
    "compute_noise_spectrum",
    "NoiseType",
]
