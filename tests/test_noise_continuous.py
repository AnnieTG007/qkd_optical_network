"""连续噪声求解器测试（SpRS + FWM）。

覆盖：
- 连续输出形状与离散一致
- 单频极限下连续/离散交叉验证（<= 5%）
- 连续噪声功率非负
"""

from __future__ import annotations

import numpy as np

from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import SpectrumType, build_wdm_grid, build_frequency_grid
from qkd_sim.physical.noise import compute_noise


def make_fiber() -> Fiber:
    cfg = FiberConfig(
        alpha_dB_per_km=0.2,
        gamma_per_W_km=1.3,
        D_ps_nm_km=17.0,
        D_slope_ps_nm2_km=0.056,
        L_km=50.0,
        A_eff=80e-12,
        rayleigh_coeff=4.8e-8,
        T_kelvin=300.0,
    )
    return Fiber(cfg)


def make_wdm_single_freq() -> tuple:
    """构建 SINGLE_FREQ WDMGrid（用于交叉验证极限）。
    同时设置 f_grid。
    """
    cfg = WDMConfig(
        f_center=193.5e12,
        N_ch=7,
        channel_spacing=50e9,
        B_s=32e9,
        P0=1e-3,
        quantum_channel_indices=[3],  # 居中量子信道
    )
    all_f = np.sort(cfg.f_center + np.arange(-(cfg.N_ch - 1) / 2, (cfg.N_ch + 1) / 2) * cfg.channel_spacing)
    f_min = all_f.min() - 2 * cfg.channel_spacing
    f_max = all_f.max() + 2 * cfg.channel_spacing
    f_grid = np.arange(f_min, f_max + 0.5 * cfg.channel_spacing, cfg.channel_spacing)
    grid = build_wdm_grid(cfg, spectrum_type=SpectrumType.SINGLE_FREQ, f_grid=f_grid)
    return grid


def _relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(b), 1e-30)
    return np.abs(a - b) / denom


def test_continuous_shape_consistency():
    """连续模型输出 shape 必须与离散模型一致。"""
    fiber = make_fiber()
    grid = make_wdm_single_freq()

    r_disc = compute_noise("all", fiber, grid, continuous=False)
    r_cont = compute_noise("all", fiber, grid, continuous=True)

    n_q = len(grid.get_quantum_channels())
    for key in ("sprs_fwd", "sprs_bwd", "fwm_fwd", "fwm_bwd"):
        assert key in r_disc and key in r_cont
        assert r_disc[key].shape == (n_q,), f"{key} discrete shape mismatch"
        assert r_cont[key].shape == (n_q,), f"{key} continuous shape mismatch"


def test_continuous_cross_validation_single_frequency_limit():
    """单频极限：连续结果应与离散结果相差 <= 5%（交叉验证）。"""
    fiber = make_fiber()
    grid = make_wdm_single_freq()

    r_disc = compute_noise("all", fiber, grid, continuous=False)
    r_cont = compute_noise("all", fiber, grid, continuous=True)

    for key in ("sprs_fwd", "sprs_bwd", "fwm_fwd", "fwm_bwd"):
        err = _relative_error(r_cont[key], r_disc[key])
        assert np.all(err <= 0.05), (
            f"{key} continuous/discrete relative error > 5%: max={err.max():.4f}"
        )


def test_continuous_noise_positive():
    """连续模型噪声功率必须非负。"""
    fiber = make_fiber()
    grid = make_wdm_single_freq()

    r_cont = compute_noise("all", fiber, grid, continuous=True)

    for key in ("sprs_fwd", "sprs_bwd", "fwm_fwd", "fwm_bwd"):
        assert np.all(r_cont[key] >= 0.0), f"{key} has negative values: {r_cont[key]}"


def test_continuous_requires_f_grid():
    """continuous=True 但 f_grid=None 时应抛出 ValueError。"""
    fiber = make_fiber()
    cfg = WDMConfig(
        f_center=193.5e12, N_ch=7, channel_spacing=50e9,
        B_s=32e9, P0=1e-3, quantum_channel_indices=[3],
    )
    grid = build_wdm_grid(cfg, spectrum_type=SpectrumType.SINGLE_FREQ, f_grid=None)

    try:
        compute_noise("all", fiber, grid, continuous=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "f_grid" in str(e)


def test_continuous_sprs_only():
    """仅 SpRS 连续计算。"""
    fiber = make_fiber()
    grid = make_wdm_single_freq()

    r_cont = compute_noise("sprs", fiber, grid, continuous=True)

    assert "sprs_fwd" in r_cont
    assert "sprs_bwd" in r_cont
    assert "fwm_fwd" not in r_cont
    assert "fwm_bwd" not in r_cont


def test_continuous_fwm_only():
    """仅 FWM 连续计算。"""
    fiber = make_fiber()
    grid = make_wdm_single_freq()

    r_cont = compute_noise("fwm", fiber, grid, continuous=True)

    assert "fwm_fwd" in r_cont
    assert "fwm_bwd" in r_cont
    assert "sprs_fwd" not in r_cont
    assert "sprs_bwd" not in r_cont
