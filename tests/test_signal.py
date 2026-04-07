"""Phase 1 测试：单位转换、配置加载、WDM网格、G_TX 功率归一化。"""

import numpy as np
import pytest
from pathlib import Path

# ============================================================
# 1. 单位转换往返测试
# ============================================================

from qkd_sim.utils.units import (
    alpha_dB_km_to_per_m,
    alpha_per_m_to_dB_km,
    gamma_per_W_km_to_per_W_m,
    gamma_per_W_m_to_per_W_km,
    D_ps_nm_km_to_s_m2,
    D_s_m2_to_ps_nm_km,
    D_slope_ps_nm2_km_to_s_m3,
    D_slope_s_m3_to_ps_nm2_km,
    L_km_to_m,
    L_m_to_km,
    power_W_to_dBm,
    power_dBm_to_W,
    freq_Hz_to_wavelength_m,
    wavelength_m_to_freq_Hz,
)


class TestUnitConversions:
    """单位转换的往返一致性测试。"""

    def test_alpha_roundtrip(self):
        alpha_dB = 0.2  # dB/km
        alpha_si = alpha_dB_km_to_per_m(alpha_dB)
        assert alpha_si > 0
        assert np.isclose(alpha_per_m_to_dB_km(alpha_si), alpha_dB, rtol=1e-10)

    def test_alpha_known_value(self):
        """0.2 dB/km ≈ 4.606e-5 /m。
        量纲：0.2 [dB/km] × 1e-3 [km/m] / (10×log10(e) [dB/Np]) = 4.606e-5 [Np/m]
        物理验证：exp(-alpha * 50e3) = exp(-2.303) ≈ 0.1 (对应 50km @ 0.2dB/km = 10dB 损耗)
        """
        alpha_si = alpha_dB_km_to_per_m(0.2)
        expected = 0.2 * 1e-3 / (10 * np.log10(np.e))  # ≈ 4.606e-5 /m
        assert np.isclose(alpha_si, expected, rtol=1e-10)
        # 物理自洽性：50km 处功率损耗应约为 10 dB
        loss_dB_50km = -10 * np.log10(np.exp(-alpha_si * 50e3))
        assert np.isclose(loss_dB_50km, 10.0, rtol=1e-6), f"50km loss = {loss_dB_50km:.3f} dB，应约为 10 dB"

    def test_gamma_roundtrip(self):
        gamma_km = 1.3  # 1/(W·km)
        gamma_si = gamma_per_W_km_to_per_W_m(gamma_km)
        assert np.isclose(gamma_si, 1.3e-3)
        assert np.isclose(gamma_per_W_m_to_per_W_km(gamma_si), gamma_km)

    def test_D_roundtrip(self):
        D = 17.0  # ps/(nm·km)
        D_si = D_ps_nm_km_to_s_m2(D)
        assert np.isclose(D_si, 17.0e-6)
        assert np.isclose(D_s_m2_to_ps_nm_km(D_si), D, rtol=1e-10)

    def test_D_slope_roundtrip(self):
        S = 0.056  # ps/(nm²·km)
        S_si = D_slope_ps_nm2_km_to_s_m3(S)
        assert np.isclose(S_si, 0.056 * 1e3)
        assert np.isclose(D_slope_s_m3_to_ps_nm2_km(S_si), S, rtol=1e-10)

    def test_L_roundtrip(self):
        L_km_val = 50.0
        L_m_val = L_km_to_m(L_km_val)
        assert np.isclose(L_m_val, 50e3)
        assert np.isclose(L_m_to_km(L_m_val), L_km_val)

    def test_power_roundtrip(self):
        P_W = 1e-3  # 1 mW = 0 dBm
        P_dBm = power_W_to_dBm(P_W)
        assert np.isclose(P_dBm, 0.0, atol=1e-10)
        assert np.isclose(power_dBm_to_W(P_dBm), P_W, rtol=1e-10)

    def test_power_array(self):
        P_W = np.array([1e-3, 1e-2, 1e-1])
        P_dBm = power_W_to_dBm(P_W)
        assert np.allclose(power_dBm_to_W(P_dBm), P_W, rtol=1e-10)

    def test_freq_wavelength_roundtrip(self):
        f = 193.5e12  # Hz
        lam = freq_Hz_to_wavelength_m(f)
        assert np.isclose(lam, 1549.3e-9, rtol=1e-3)  # ~1549.3 nm
        assert np.isclose(wavelength_m_to_freq_Hz(lam), f, rtol=1e-10)


# ============================================================
# 2. 配置加载测试
# ============================================================

from qkd_sim.config.schema import (
    FiberConfig,
    WDMConfig,
    load_fiber_config,
    load_wdm_config,
)

DEFAULTS_DIR = Path(__file__).resolve().parents[1] / "src" / "qkd_sim" / "config" / "defaults"


class TestFiberConfig:
    """FiberConfig 的 SI 转换测试。"""

    def test_post_init_conversion(self):
        cfg = FiberConfig(
            alpha_dB_per_km=0.2,
            gamma_per_W_km=1.3,
            D_ps_nm_km=17.0,
            D_slope_ps_nm2_km=0.056,
            L_km=50.0,
            A_eff=80e-12,
            rayleigh_coeff=4.8e-8,
        )
        assert np.isclose(cfg.alpha, alpha_dB_km_to_per_m(0.2))
        assert np.isclose(cfg.gamma, 1.3e-3)
        assert np.isclose(cfg.D_c, 17.0e-6)
        assert np.isclose(cfg.D_slope, 0.056 * 1e3)
        assert np.isclose(cfg.L, 50e3)

    def test_load_yaml(self):
        yaml_path = DEFAULTS_DIR / "fiber_smf.yaml"
        if not yaml_path.exists():
            pytest.skip("Default YAML not found")
        cfg = load_fiber_config(yaml_path)
        assert cfg.alpha > 0
        assert cfg.L > 0


class TestWDMConfig:
    """WDMConfig 加载测试。"""

    def test_load_yaml(self):
        yaml_path = DEFAULTS_DIR / "wdm_50ghz.yaml"
        if not yaml_path.exists():
            pytest.skip("Default YAML not found")
        cfg = load_wdm_config(yaml_path)
        assert cfg.N_ch == 16
        assert np.isclose(cfg.channel_spacing, 50e9)


# ============================================================
# 3. WDM 网格生成测试
# ============================================================

from qkd_sim.physical.signal import (
    SpectrumType,
    WDMChannel,
    WDMGrid,
    build_wdm_grid,
    build_frequency_grid,
)


class TestWDMGrid:
    """WDM 网格构建测试。"""

    @pytest.fixture
    def wdm_config(self):
        return WDMConfig(
            f_center=193.5e12,
            N_ch=5,
            channel_spacing=50e9,
            B_s=32e9,
            P0=1e-3,
            beta_rolloff=0.0,
            quantum_channel_indices=[2],  # 中间信道为量子
        )

    def test_channel_count(self, wdm_config):
        grid = build_wdm_grid(wdm_config, SpectrumType.SINGLE_FREQ)
        assert len(grid.channels) == 5

    def test_channel_frequencies_symmetric(self, wdm_config):
        grid = build_wdm_grid(wdm_config, SpectrumType.SINGLE_FREQ)
        freqs = grid.get_channel_frequencies()
        # 应关于中心对称
        center = np.mean(freqs)
        assert np.isclose(center, wdm_config.f_center, rtol=1e-10)
        # 间隔应为 50 GHz
        diffs = np.diff(freqs)
        assert np.allclose(diffs, 50e9, rtol=1e-10)

    def test_channel_frequencies_formula(self, wdm_config):
        """验证公式: f_n = f_center + arange(-(N-1)/2, (N+1)/2) * g"""
        grid = build_wdm_grid(wdm_config, SpectrumType.SINGLE_FREQ)
        freqs = grid.get_channel_frequencies()
        expected = wdm_config.f_center + np.arange(-2, 3) * 50e9
        assert np.allclose(freqs, expected)

    def test_quantum_channel_assignment(self, wdm_config):
        grid = build_wdm_grid(wdm_config, SpectrumType.SINGLE_FREQ)
        assert grid.channels[2].channel_type == "quantum"
        assert grid.channels[0].channel_type == "classical"
        assert len(grid.get_quantum_channels()) == 1
        assert len(grid.get_classical_channels()) == 4

    def test_frequency_grid_covers_channels(self, wdm_config):
        f_grid = build_frequency_grid(wdm_config, resolution=1e9)
        grid = build_wdm_grid(wdm_config, SpectrumType.RECTANGULAR, f_grid=f_grid)
        freqs = grid.get_channel_frequencies()
        # 频率网格应覆盖所有信道
        assert f_grid[0] < freqs[0]
        assert f_grid[-1] > freqs[-1]


# ============================================================
# 4. G_TX 功率归一化测试
# ============================================================


class TestGTXNormalization:
    """验证 ∫G_TX(f) df = P_ch。"""

    @pytest.fixture
    def f_grid(self):
        """高分辨率频率网格用于积分精度。"""
        return np.linspace(193.0e12, 194.0e12, 100001)

    def test_rectangular_normalization(self, f_grid):
        """矩形谱: ∫G_TX df ≈ P_ch"""
        P_ch = 1e-3  # 1 mW
        B_s = 32e9
        ch = WDMChannel(
            f_center=193.5e12, power=P_ch,
            channel_type="classical", spectrum_type=SpectrumType.RECTANGULAR,
            B_s=B_s,
        )
        psd = ch.get_psd(f_grid)
        df = f_grid[1] - f_grid[0]
        P_integrated = np.trapezoid(psd, f_grid)
        assert np.isclose(P_integrated, P_ch, rtol=1e-3), \
            f"Rectangular: integrated {P_integrated:.6e} vs expected {P_ch:.6e}"

    def test_raised_cosine_normalization(self, f_grid):
        """升余弦谱: ∫G_TX df ≈ P_ch"""
        P_ch = 2e-3
        B_s = 32e9
        for beta in [0.1, 0.3, 0.5, 1.0]:
            ch = WDMChannel(
                f_center=193.5e12, power=P_ch,
                channel_type="classical", spectrum_type=SpectrumType.RAISED_COSINE,
                B_s=B_s, beta_rolloff=beta,
            )
            psd = ch.get_psd(f_grid)
            P_integrated = np.trapezoid(psd, f_grid)
            assert np.isclose(P_integrated, P_ch, rtol=1e-2), \
                f"RC(β={beta}): integrated {P_integrated:.6e} vs expected {P_ch:.6e}"

    def test_raised_cosine_beta0_equals_rectangular(self, f_grid):
        """β = 0 时升余弦谱退化为矩形谱。"""
        P_ch = 1e-3
        B_s = 32e9
        ch_rect = WDMChannel(
            f_center=193.5e12, power=P_ch,
            channel_type="classical", spectrum_type=SpectrumType.RECTANGULAR,
            B_s=B_s,
        )
        ch_rc = WDMChannel(
            f_center=193.5e12, power=P_ch,
            channel_type="classical", spectrum_type=SpectrumType.RAISED_COSINE,
            B_s=B_s, beta_rolloff=0.0,
        )
        psd_rect = ch_rect.get_psd(f_grid)
        psd_rc = ch_rc.get_psd(f_grid)
        assert np.allclose(psd_rect, psd_rc)

    def test_single_freq_psd_is_zero(self, f_grid):
        """离散模型 PSD 应全为零（功率在离散频率点）。"""
        ch = WDMChannel(
            f_center=193.5e12, power=1e-3,
            channel_type="classical", spectrum_type=SpectrumType.SINGLE_FREQ,
            B_s=32e9,
        )
        psd = ch.get_psd(f_grid)
        assert np.all(psd == 0.0)

    def test_total_psd_multi_channel(self, f_grid):
        """多信道总 PSD 的功率守恒。"""
        P_ch = 1e-3
        N_ch = 3
        channels = []
        for i, f_c in enumerate([193.4e12, 193.5e12, 193.6e12]):
            ch = WDMChannel(
                f_center=f_c, power=P_ch,
                channel_type="classical", spectrum_type=SpectrumType.RECTANGULAR,
                B_s=32e9,
            )
            channels.append(ch)

        grid = WDMGrid(channels=channels, f_grid=f_grid)
        total_psd = grid.get_total_psd()
        P_total = np.trapezoid(total_psd, f_grid)
        assert np.isclose(P_total, N_ch * P_ch, rtol=1e-3), \
            f"Total: {P_total:.6e} vs expected {N_ch * P_ch:.6e}"


# ============================================================
# 5. Fiber 类测试
# ============================================================

from qkd_sim.physical.fiber import Fiber


class TestFiber:
    """Fiber 类基本功能测试。"""

    @pytest.fixture
    def fiber(self):
        cfg = FiberConfig(
            alpha_dB_per_km=0.2,
            gamma_per_W_km=1.3,
            D_ps_nm_km=17.0,
            D_slope_ps_nm2_km=0.056,
            L_km=50.0,
            A_eff=80e-12,
            rayleigh_coeff=4.8e-8,
        )
        return Fiber(cfg)

    def test_basic_properties(self, fiber):
        assert fiber.alpha > 0
        assert fiber.gamma > 0
        assert fiber.L == 50e3

    def test_loss_at_freq_constant(self, fiber):
        """当前 C 波段实现：衰减为常数。"""
        freqs = np.array([192e12, 193e12, 194e12, 195e12])
        losses = fiber.get_loss_at_freq(freqs)
        assert np.allclose(losses, fiber.alpha)

    def test_dispersion_at_ref_freq(self, fiber):
        """参考频率处色散应等于 D_c。"""
        D_at_ref = fiber.get_dispersion_at_freq(193.5e12)
        assert np.isclose(D_at_ref, fiber.config.D_c, rtol=1e-6)

    def test_dispersion_slope(self, fiber):
        """不同频率的色散差应由斜率决定。"""
        f1 = 193.0e12
        f2 = 194.0e12
        D1 = fiber.get_dispersion_at_freq(f1)
        D2 = fiber.get_dispersion_at_freq(f2)
        # D2 > D1 因为更高频率对应更短波长，色散应减小
        # （更短波长 = 更小 λ，D_c + D_slope*(λ-λ₀) 中 λ < λ₀）
        assert D1 != D2

    def test_effective_length(self, fiber):
        """L_eff < L，且 L_eff > 0。"""
        L_eff = fiber.get_effective_length()
        assert 0 < L_eff < fiber.L

    def test_phase_mismatch_self(self, fiber):
        """当 f₃ = f₂ 或 f₄ = f₂ 时，Δβ = 0。"""
        f = 193.5e12
        delta_beta = fiber.get_phase_mismatch(f2=f, f3=f, f4=194.0e12)
        assert np.isclose(delta_beta, 0.0)

    def test_phase_mismatch_nonzero(self, fiber):
        """不同频率组合应给出非零 Δβ。"""
        delta_beta = fiber.get_phase_mismatch(
            f2=193.5e12, f3=193.6e12, f4=193.7e12,
        )
        assert delta_beta != 0.0
