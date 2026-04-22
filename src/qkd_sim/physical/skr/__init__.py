"""SKR 计算子包。

当前模块：
  skr_decoy_bb84 : 诱骗态 BB84 协议安全码率（无限长、近似有限长、严格有限长）
  skr_optimizer  : SKR 参数优化（Nelder-Mead + 顺序热启动）
"""

from qkd_sim.physical.skr.skr_decoy_bb84 import (
    H2,
    infinite_key_rate,
    approx_finite_key_rate,
    strict_finite_key_rate,
)
from qkd_sim.physical.skr.skr_optimizer import (
    OptimizationResult,
    SKROptimizer,
)

__all__ = [
    "H2",
    "infinite_key_rate",
    "approx_finite_key_rate",
    "strict_finite_key_rate",
    "OptimizationResult",
    "SKROptimizer",
]
