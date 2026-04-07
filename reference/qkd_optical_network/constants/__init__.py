"""
constants 模块

提供物理常数、单位转换、光纤标准参数等。
"""

from .fiber_parameters import (
    # 光纤类型
    FiberType,
    # 标准光纤参数
    SSMF_PARAMETERS,
    HCF_PARAMETERS,
    get_fiber_parameters,
    # 拉曼系数表（GNpy 来源）
    GNPY_RAMAN_COEFFICIENT,
)

__all__ = [
    'FiberType',
    'SSMF_PARAMETERS',
    'HCF_PARAMETERS',
    'get_fiber_parameters',
    'GNPY_RAMAN_COEFFICIENT',
]
