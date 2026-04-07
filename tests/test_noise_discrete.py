"""离散噪声求解器测试。

覆盖：
  - 维度检查（输出 shape 与量子信道数一致）
  - 极限情况（泵浦→0、L→0、T→0K、强色散）
  - 物理一致性（SpRS 与泵浦功率线性关系；FWM 与泵浦功率三次方关系；
                  Stokes > anti-Stokes）
  - compute_noise 调度入口覆盖

测试使用的默认场景：
  - 光纤：50 km SSMF（fiber_smf.yaml）
  - WDM：16 信道 50 GHz 间隔，信道 8 为量子信道，其余为经典信道（各 1 mW）
"""

from __future__ import annotations

import numpy as np
import pytest

from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import WDMGrid, WDMChannel, SpectrumType, build_wdm_grid
from qkd_sim.physical.noise import (
    DiscreteSPRSSolver,
    DiscreteFWMSolver,
    compute_noise,
)


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

def make_fiber(
    alpha_dB_per_km: float = 0.2,
    gamma_per_W_km: float = 1.3,
    L_km: float = 50.0,
    T_kelvin: float = 300.0,
    rayleigh_coeff: float = 4.8e-8,
) -> Fiber:
    """构造 SSMF Fiber 实例（使用常用单位）。"""
    cfg = FiberConfig(
        alpha_dB_per_km=alpha_dB_per_km,
        gamma_per_W_km=gamma_per_W_km,
        D_ps_nm_km=17.0,
        D_slope_ps_nm2_km=0.056,
        L_km=L_km,
        A_eff=80e-12,
        rayleigh_coeff=rayleigh_coeff,
        T_kelvin=T_kelvin,
    )
    return Fiber(cfg)


def make_wdm_grid(
    N_ch: int = 16,
    quantum_indices: list[int] | None = None,
    P0: float = 1e-3,
    f_center: float = 193.5e12,
    channel_spacing: float = 50e9,
) -> WDMGrid:
    """构造 WDMGrid，使用 SINGLE_FREQ（离散模型）频谱类型。"""
    if quantum_indices is None:
        quantum_indices = [8]
    cfg = WDMConfig(
        f_center=f_center,
        N_ch=N_ch,
        channel_spacing=channel_spacing,
        B_s=32e9,
        P0=P0,
        quantum_channel_indices=quantum_indices,
    )
    return build_wdm_grid(cfg, spectrum_type=SpectrumType.SINGLE_FREQ)


# ---------------------------------------------------------------------------
# 辅助：默认场景
# ---------------------------------------------------------------------------

FIBER = make_fiber()
GRID = make_wdm_grid()
N_Q = len(GRID.get_quantum_channels())  # = 1


# ===========================================================================
# SpRS 测试
# ===========================================================================

class TestDiscreteSPRS:

    def test_forward_shape(self):
        """前向噪声输出 shape == (N_q,)。"""
        solver = DiscreteSPRSSolver()
        P = solver.compute_forward(FIBER, GRID)
        assert P.shape == (N_Q,), f"expected ({N_Q},), got {P.shape}"

    def test_backward_shape(self):
        """后向噪声输出 shape == (N_q,)。"""
        solver = DiscreteSPRSSolver()
        P = solver.compute_backward(FIBER, GRID)
        assert P.shape == (N_Q,), f"expected ({N_Q},), got {P.shape}"

    def test_forward_positive(self):
        """前向噪声功率 > 0。"""
        solver = DiscreteSPRSSolver()
        P = solver.compute_forward(FIBER, GRID)
        assert np.all(P > 0), f"非正值：{P}"

    def test_backward_positive(self):
        """后向噪声功率 > 0。"""
        solver = DiscreteSPRSSolver()
        P = solver.compute_backward(FIBER, GRID)
        assert np.all(P > 0)

    def test_zero_pump_power(self):
        """泵浦功率为 0 时，SpRS 噪声为 0。"""
        grid_zero = make_wdm_grid(P0=0.0)
        solver = DiscreteSPRSSolver()
        assert np.all(solver.compute_forward(FIBER, grid_zero) == 0.0)
        assert np.all(solver.compute_backward(FIBER, grid_zero) == 0.0)

    def test_zero_fiber_length(self):
        """光纤长度越短，SpRS 前向噪声越小（线性缩减）。"""
        solver = DiscreteSPRSSolver()
        fiber_50km = make_fiber(L_km=50.0)
        fiber_1km = make_fiber(L_km=1.0)
        fiber_01km = make_fiber(L_km=0.1)
        P_50 = solver.compute_forward(fiber_50km, GRID)
        P_1 = solver.compute_forward(fiber_1km, GRID)
        P_01 = solver.compute_forward(fiber_01km, GRID)
        # 随 L 减小，噪声单调递减
        assert np.all(P_50 > P_1), "50km 噪声应大于 1km 噪声"
        assert np.all(P_1 > P_01), "1km 噪声应大于 0.1km 噪声"

    def test_linear_in_pump_power(self):
        """SpRS 噪声与泵浦功率成线性关系（1次方）。"""
        solver = DiscreteSPRSSolver()
        grid_1x = make_wdm_grid(P0=1e-3)
        grid_2x = make_wdm_grid(P0=2e-3)
        P1 = solver.compute_forward(FIBER, grid_1x)
        P2 = solver.compute_forward(FIBER, grid_2x)
        ratio = P2 / P1
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10,
            err_msg="SpRS 前向噪声应与泵浦功率成线性关系")

    def test_linear_backward(self):
        """后向 SpRS 噪声同样与泵浦功率线性。"""
        solver = DiscreteSPRSSolver()
        grid_1x = make_wdm_grid(P0=1e-3)
        grid_3x = make_wdm_grid(P0=3e-3)
        P1 = solver.compute_backward(FIBER, grid_1x)
        P3 = solver.compute_backward(FIBER, grid_3x)
        np.testing.assert_allclose(P3 / P1, 3.0, rtol=1e-10,
            err_msg="SpRS 后向噪声应与泵浦功率成线性关系")

    def test_stokes_gt_antistokes(self):
        """量子信道左侧（Stokes 侧）经典信道贡献 > 右侧（anti-Stokes 侧）。

        使用单泵浦在量子信道左右各放一个经典信道，对比贡献大小。
        Stokes：f_c > f_q（泵浦频率更高），对应红移散射。
        anti-Stokes：f_c < f_q（泵浦频率更低），对应蓝移散射。
        """
        # 量子信道 f_q = 193.5 THz
        # Stokes 泵浦 f_c = 193.5 + 2 THz（频率高于量子信道）
        # anti-Stokes 泵浦 f_c = 193.5 - 2 THz（频率低于量子信道）
        f_q = 193.5e12
        delta = 2e12  # 2 THz 频移

        def single_pump_grid(f_pump: float) -> WDMGrid:
            channels = [
                WDMChannel(
                    f_center=f_pump, power=1e-3,
                    channel_type="classical",
                    spectrum_type=SpectrumType.SINGLE_FREQ,
                    B_s=32e9,
                ),
                WDMChannel(
                    f_center=f_q, power=0.0,
                    channel_type="quantum",
                    spectrum_type=SpectrumType.SINGLE_FREQ,
                    B_s=32e9,
                ),
            ]
            return WDMGrid(channels=channels)

        solver = DiscreteSPRSSolver()
        fiber = make_fiber()

        P_stokes = solver.compute_forward(fiber, single_pump_grid(f_q + delta))
        P_antistokes = solver.compute_forward(fiber, single_pump_grid(f_q - delta))

        assert P_stokes[0] > P_antistokes[0], (
            f"Stokes 贡献 {P_stokes[0]:.3e} W 应大于 "
            f"anti-Stokes 贡献 {P_antistokes[0]:.3e} W"
        )

    def test_cold_temperature_antistokes_vanishes(self):
        """T → 0K 时，anti-Stokes 分量（n_th → 0）极小。"""
        fiber_cold = make_fiber(T_kelvin=1.0)  # 近似 0K

        f_q = 193.5e12
        delta = 2e12

        def single_pump_grid(f_pump: float) -> WDMGrid:
            return WDMGrid(channels=[
                WDMChannel(f_center=f_pump, power=1e-3,
                           channel_type="classical",
                           spectrum_type=SpectrumType.SINGLE_FREQ, B_s=32e9),
                WDMChannel(f_center=f_q, power=0.0,
                           channel_type="quantum",
                           spectrum_type=SpectrumType.SINGLE_FREQ, B_s=32e9),
            ])

        solver = DiscreteSPRSSolver()
        P_antistokes_cold = solver.compute_forward(
            fiber_cold, single_pump_grid(f_q - delta)
        )
        P_stokes_cold = solver.compute_forward(
            fiber_cold, single_pump_grid(f_q + delta)
        )
        # 低温下反 Stokes 应比 Stokes 小得多（至少 6 个数量级）
        assert P_antistokes_cold[0] < P_stokes_cold[0] * 1e-6, (
            f"T=1K 时 anti-Stokes ({P_antistokes_cold[0]:.2e}) 应远小于 "
            f"Stokes ({P_stokes_cold[0]:.2e})"
        )

    def test_multi_quantum_channels(self):
        """多个量子信道时，输出 shape 正确。"""
        grid_multi = make_wdm_grid(N_ch=16, quantum_indices=[4, 8, 12])
        solver = DiscreteSPRSSolver()
        P_fwd = solver.compute_forward(FIBER, grid_multi)
        P_bwd = solver.compute_backward(FIBER, grid_multi)
        assert P_fwd.shape == (3,)
        assert P_bwd.shape == (3,)
        assert np.all(P_fwd > 0)
        assert np.all(P_bwd > 0)


# ===========================================================================
# FWM 测试
# ===========================================================================

class TestDiscreteFWM:

    # 使用对称布局：量子信道在中间，两侧各有经典信道
    GRID_FWM = make_wdm_grid(N_ch=7, quantum_indices=[3], P0=1e-3)

    def test_forward_shape(self):
        """前向 FWM 噪声 shape == (N_q,)。"""
        solver = DiscreteFWMSolver()
        P = solver.compute_forward(FIBER, self.GRID_FWM)
        assert P.shape == (1,), f"expected (1,), got {P.shape}"

    def test_backward_shape(self):
        """后向 FWM 噪声 shape == (N_q,)。"""
        solver = DiscreteFWMSolver()
        P = solver.compute_backward(FIBER, self.GRID_FWM)
        assert P.shape == (1,)

    def test_forward_non_negative(self):
        """前向 FWM 噪声 >= 0。"""
        solver = DiscreteFWMSolver()
        P = solver.compute_forward(FIBER, self.GRID_FWM)
        assert np.all(P >= 0)

    def test_backward_non_negative(self):
        """后向 FWM 噪声 >= 0。"""
        solver = DiscreteFWMSolver()
        P = solver.compute_backward(FIBER, self.GRID_FWM)
        assert np.all(P >= 0)

    def test_zero_pump_power(self):
        """泵浦功率为 0 时 FWM 噪声为 0。"""
        grid_zero = make_wdm_grid(N_ch=7, quantum_indices=[3], P0=0.0)
        solver = DiscreteFWMSolver()
        assert np.all(solver.compute_forward(FIBER, grid_zero) == 0.0)
        assert np.all(solver.compute_backward(FIBER, grid_zero) == 0.0)

    def test_cubic_in_pump_power(self):
        """FWM 前向噪声与泵浦功率成三次方关系（P₂×P₃×P₄）。"""
        solver = DiscreteFWMSolver()
        grid_1x = make_wdm_grid(N_ch=7, quantum_indices=[3], P0=1e-3)
        grid_2x = make_wdm_grid(N_ch=7, quantum_indices=[3], P0=2e-3)
        P1 = solver.compute_forward(FIBER, grid_1x)
        P2 = solver.compute_forward(FIBER, grid_2x)
        # 对于所有泵浦功率相等的情况，噪声 ∝ P₀³
        if np.any(P1 > 0):
            np.testing.assert_allclose(P2 / P1, 8.0, rtol=1e-10,
                err_msg="FWM 前向噪声应与泵浦功率成三次方关系")

    def test_large_dispersion_reduces_fwm(self):
        """高色散削弱 FWM（相位失配增大，η 减小）。"""
        solver = DiscreteFWMSolver()
        fiber_low_D = make_fiber()  # D=17 ps/(nm·km)

        cfg_high_D = FiberConfig(
            alpha_dB_per_km=0.2, gamma_per_W_km=1.3,
            D_ps_nm_km=100.0, D_slope_ps_nm2_km=0.056,
            L_km=50.0, A_eff=80e-12, rayleigh_coeff=4.8e-8,
        )
        fiber_high_D = Fiber(cfg_high_D)

        P_low = solver.compute_forward(fiber_low_D, self.GRID_FWM)
        P_high = solver.compute_forward(fiber_high_D, self.GRID_FWM)

        if np.any(P_low > 0):
            assert np.all(P_high <= P_low), (
                f"高色散应使 FWM 减弱：low_D={P_low}, high_D={P_high}"
            )

    def test_no_fwm_without_classical_channels(self):
        """无经典信道时 FWM 为 0（无泵浦）。"""
        # 全部信道设为量子信道时，P_pump 全为 0
        grid_all_quantum = make_wdm_grid(
            N_ch=5, quantum_indices=[0, 1, 2, 3, 4], P0=1e-3
        )
        solver = DiscreteFWMSolver()
        P = solver.compute_forward(FIBER, grid_all_quantum)
        assert np.all(P == 0.0), f"无经典信道时 FWM 应为 0，得到 {P}"


# ===========================================================================
# dispatcher 测试
# ===========================================================================

class TestDispatcher:

    GRID = make_wdm_grid(N_ch=7, quantum_indices=[3])

    def test_sprs_only(self):
        """noise_type='sprs' 仅返回 sprs 键。"""
        result = compute_noise("sprs", FIBER, self.GRID)
        assert "sprs_fwd" in result and "sprs_bwd" in result
        assert "fwm_fwd" not in result and "fwm_bwd" not in result

    def test_fwm_only(self):
        """noise_type='fwm' 仅返回 fwm 键。"""
        result = compute_noise("fwm", FIBER, self.GRID)
        assert "fwm_fwd" in result and "fwm_bwd" in result
        assert "sprs_fwd" not in result

    def test_all_keys(self):
        """noise_type='all' 返回全部 4 个键。"""
        result = compute_noise("all", FIBER, self.GRID)
        for key in ("sprs_fwd", "sprs_bwd", "fwm_fwd", "fwm_bwd"):
            assert key in result, f"缺少键 '{key}'"

    def test_invalid_type_raises(self):
        """非法 noise_type 抛出 ValueError。"""
        with pytest.raises(ValueError):
            compute_noise("invalid", FIBER, self.GRID)  # type: ignore

    def test_shapes_consistent(self):
        """所有输出 shape 与量子信道数一致。"""
        result = compute_noise("all", FIBER, self.GRID)
        n_q = len(self.GRID.get_quantum_channels())
        for key, arr in result.items():
            assert arr.shape == (n_q,), f"键 '{key}' shape {arr.shape} != ({n_q},)"
