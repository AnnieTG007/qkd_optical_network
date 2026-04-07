"""
noise 模块

物理层噪声模型：
- FWM（四波混频）
- Raman（拉曼散射：自发 + 受激）
- 芯间噪声（预留）

推荐入口
--------
优先使用统一分派接口：

    from physics.noise import compute_noise, NoiseResult

compute_noise() 根据 channels[*].spectrum_type 自动选择离散或连续模型。
底层模型函数保留用于直接调用（测试、对比脚本等场景）。
"""

# ── 统一分派接口（推荐使用）──────────────────────────────────────────────────
from .dispatcher import compute_noise, NoiseResult

# ── 底层离散模型 ──────────────────────────────────────────────────────────────
from .fwm import compute_fwm_noise
from .raman import compute_raman_noise

# ── 底层连续模型 ──────────────────────────────────────────────────────────────
from .fwm_continuous import (
    compute_fwm_noise_continuous,
    compute_fwm_noise_vectorized,
    compute_fwm_noise_discrete
)
from .raman_continuous import (
    compute_raman_noise_continuous,
    compute_raman_noise_vectorized,
    compute_raman_noise_discrete
)

__all__ = [
    # 统一接口
    'compute_noise',
    'NoiseResult',
    # 离散底层
    'compute_fwm_noise',
    'compute_raman_noise',
    # 连续底层
    'compute_fwm_noise_continuous',
    'compute_fwm_noise_vectorized',
    'compute_fwm_noise_discrete',
    'compute_raman_noise_continuous',
    'compute_raman_noise_vectorized',
    'compute_raman_noise_discrete',
]
