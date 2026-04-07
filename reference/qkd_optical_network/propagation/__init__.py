"""
propagation 模块

单跨段光纤传输求解器：
- 噪声传播（FWM + 拉曼）
- 有效长度计算
"""

from .solver import SingleSpanSolver
from .effective_length import EffectiveLengthCalculator

__all__ = [
    'SingleSpanSolver',
    'EffectiveLengthCalculator',
]
