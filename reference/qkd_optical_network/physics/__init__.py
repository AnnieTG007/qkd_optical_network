"""
physics 模块

物理层建模：信号表示、光纤参数、噪声模型（FWM + 拉曼）。
"""

from .signal import WDMChannel, SignalState
from .fiber import Fiber

__all__ = [
    'WDMChannel',
    'SignalState',
    'Fiber',
]
