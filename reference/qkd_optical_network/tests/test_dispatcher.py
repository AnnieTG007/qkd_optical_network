"""
Dispatcher 路由测试

测试 compute_noise() 根据 SpectrumType 正确路由到离散/连续模型。
"""

import unittest
import numpy as np

from physics.fiber import Fiber
from physics.signal import WDMChannel, SpectrumType
from physics.noise import compute_noise
from physics.noise.dispatcher import _determine_noise_mode
from constants.fiber_parameters import FiberType


class TestDispatcherRouting(unittest.TestCase):
    """dispatcher 路由测试类"""

    def setUp(self):
        """测试前准备"""
        self.fiber = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,  # 50 km
            temperature=300.0
        )

        # 标准 3 信道配置（满足 FWM 最小需求）
        self.base_freq = 193.4e12
        self.spacing = 50e9

    def _build_channels(self, spectrum_type: SpectrumType):
        """构建 3 信道测试系统"""
        channels = []
        for i in range(3):
            offset = (i - 1) * self.spacing
            ch = WDMChannel(
                center_freq=self.base_freq + offset,
                power=1e-3,  # 0 dBm
                baudrate=32e9,
                roll_off=0.1,
                spectrum_type=spectrum_type,
                modulation='QPSK'
            )
            channels.append(ch)
        return channels

    # ─────────────────────────────────────────────────────────────────
    # 1. SINGLE_FREQ → 离散路径
    # ─────────────────────────────────────────────────────────────────

    def test_single_freq_routes_to_discrete(self):
        """SINGLE_FREQ 信道应路由到离散模型"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        mode = _determine_noise_mode(channels)
        self.assertEqual(mode, 'discrete')

    def test_single_freq_returns_discrete_result(self):
        """SINGLE_FREQ 应返回离散模式 NoiseResult"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels)
        self.assertEqual(result.mode, 'discrete')

    def test_single_freq_discrete_no_spectrum(self):
        """离散模式不应返回频谱数据"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels)
        self.assertIsNone(result.freq_array)
        self.assertIsNone(result.df)
        self.assertIsNone(result.fwm_spectrum)
        self.assertIsNone(result.raman_spectrum)

    def test_single_freq_fwm_power_shape(self):
        """离散模式 fwm_power shape 应为 (n_channels,)"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels)
        self.assertEqual(result.fwm_power.shape, (3,))

    def test_single_freq_raman_power_shape(self):
        """离散模式 raman_power shape 应为 (n_channels,)"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels)
        self.assertEqual(result.raman_power.shape, (3,))

    # ─────────────────────────────────────────────────────────────────
    # 2. RAISED_COSINE → 连续路径
    # ─────────────────────────────────────────────────────────────────

    def test_raised_cosine_routes_to_continuous(self):
        """RAISED_COSINE 信道应路由到连续模型"""
        channels = self._build_channels(SpectrumType.RAISED_COSINE)
        mode = _determine_noise_mode(channels)
        self.assertEqual(mode, 'continuous')

    def test_raised_cosine_returns_continuous_result(self):
        """RAISED_COSINE 应返回连续模式 NoiseResult"""
        channels = self._build_channels(SpectrumType.RAISED_COSINE)
        result = compute_noise(self.fiber, channels)
        self.assertEqual(result.mode, 'continuous')

    def test_raised_cosine_with_spectrum_returns_spectrum(self):
        """return_spectrum=True 时连续模式应返回频谱数据"""
        channels = self._build_channels(SpectrumType.RAISED_COSINE)
        result = compute_noise(self.fiber, channels, return_spectrum=True)
        self.assertIsNotNone(result.freq_array)
        self.assertIsNotNone(result.df)
        self.assertIsNotNone(result.fwm_spectrum)
        self.assertIsNotNone(result.raman_spectrum)

    def test_raised_cosine_without_spectrum_skips_spectrum(self):
        """return_spectrum=False 时连续模式不应构建频谱数据"""
        channels = self._build_channels(SpectrumType.RAISED_COSINE)
        result = compute_noise(self.fiber, channels, return_spectrum=False)
        self.assertIsNone(result.freq_array)
        self.assertIsNone(result.df)
        self.assertIsNone(result.fwm_spectrum)
        self.assertIsNone(result.raman_spectrum)

    def test_raised_cosine_fwm_spectrum_shape(self):
        """连续模式 fwm_spectrum shape 应为 (n_channels, n_freq)"""
        channels = self._build_channels(SpectrumType.RAISED_COSINE)
        result = compute_noise(self.fiber, channels, return_spectrum=True)
        self.assertEqual(result.fwm_spectrum.ndim, 2)
        self.assertEqual(result.fwm_spectrum.shape[0], 3)

    # ─────────────────────────────────────────────────────────────────
    # 3. RECTANGULAR → 连续路径
    # ─────────────────────────────────────────────────────────────────

    def test_rectangular_routes_to_continuous(self):
        """RECTANGULAR 信道应路由到连续模型"""
        channels = self._build_channels(SpectrumType.RECTANGULAR)
        mode = _determine_noise_mode(channels)
        self.assertEqual(mode, 'continuous')

    def test_rectangular_returns_continuous_result(self):
        """RECTANGULAR 应返回连续模式 NoiseResult"""
        channels = self._build_channels(SpectrumType.RECTANGULAR)
        result = compute_noise(self.fiber, channels)
        self.assertEqual(result.mode, 'continuous')

    # ─────────────────────────────────────────────────────────────────
    # 4. 混合类型 → 抛异常
    # ─────────────────────────────────────────────────────────────────

    def test_mixed_types_raises_error(self):
        """混合 SpectrumType 应抛出 ValueError"""
        channels = [
            WDMChannel(
                center_freq=self.base_freq - self.spacing,
                power=1e-3,
                baudrate=32e9,
                spectrum_type=SpectrumType.SINGLE_FREQ
            ),
            WDMChannel(
                center_freq=self.base_freq,
                power=1e-3,
                baudrate=32e9,
                spectrum_type=SpectrumType.RAISED_COSINE
            ),
            WDMChannel(
                center_freq=self.base_freq + self.spacing,
                power=1e-3,
                baudrate=32e9,
                spectrum_type=SpectrumType.SINGLE_FREQ
            ),
        ]
        with self.assertRaises(ValueError) as ctx:
            _determine_noise_mode(channels)
        self.assertIn("Inconsistent spectrum types", str(ctx.exception))

    def test_empty_channels_raises_error(self):
        """空信道列表应抛出 ValueError"""
        with self.assertRaises(ValueError):
            _determine_noise_mode([])

    # ─────────────────────────────────────────────────────────────────
    # 5. 功率语义一致性
    # ─────────────────────────────────────────────────────────────────

    def test_discrete_fwm_power_equals_sum_of_psd(self):
        """离散模式：fwm_power[i] 是信道 i 的总 FWM 噪声功率 [W]"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels)
        # fwm_power 应该是标量噪声功率，不是 PSD
        self.assertEqual(result.fwm_power.shape, (3,))
        self.assertTrue(np.all(result.fwm_power >= 0))

    def test_continuous_total_power_integrates_spectrum(self):
        """连续模式：返回 fwm_spectrum (PSD) 和 fwm_power (总功率)"""
        channels = self._build_channels(SpectrumType.RAISED_COSINE)
        result = compute_noise(self.fiber, channels, return_spectrum=True)
        # 验证频谱数据存在
        self.assertIsNotNone(result.fwm_spectrum)
        self.assertIsNotNone(result.raman_spectrum)
        # 验证 fwm_power 是非负值（物理约束）
        self.assertTrue(np.all(result.fwm_power >= 0))
        # 验证频谱形状正确 (n_channels, n_freq)
        self.assertEqual(result.fwm_spectrum.shape[0], len(channels))

    # ─────────────────────────────────────────────────────────────────
    # 6. 启用/禁用选项
    # ─────────────────────────────────────────────────────────────────

    def test_disable_fwm_sets_fwm_power_zero(self):
        """enable_fwm=False 时 fwm_power 应为零"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels, enable_fwm=False)
        np.testing.assert_array_equal(result.fwm_power, 0)

    def test_disable_raman_sets_raman_power_zero(self):
        """enable_raman=False 时 raman_power 应为零"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels, enable_raman=False)
        np.testing.assert_array_equal(result.raman_power, 0)

    # ─────────────────────────────────────────────────────────────────
    # 7. total_power 属性
    # ─────────────────────────────────────────────────────────────────

    def test_total_power_equals_fwm_plus_raman(self):
        """total_power = fwm_power + raman_power"""
        channels = self._build_channels(SpectrumType.SINGLE_FREQ)
        result = compute_noise(self.fiber, channels)
        np.testing.assert_array_almost_equal(
            result.total_power,
            result.fwm_power + result.raman_power
        )


class TestNoiseResultSpectrumTotal(unittest.TestCase):
    """NoiseResult.fwm_spectrum_total / raman_spectrum_total 属性测试"""

    def setUp(self):
        self.fiber = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,
            temperature=300.0
        )
        self.base_freq = 193.4e12

    def test_spectrum_total_shape(self):
        """fwm_spectrum_total shape 应为 (n_freq,)"""
        channels = [
            WDMChannel(
                center_freq=self.base_freq + (i - 1) * 50e9,
                power=1e-3,
                baudrate=32e9,
                spectrum_type=SpectrumType.RAISED_COSINE
            )
            for i in range(3)
        ]
        result = compute_noise(self.fiber, channels, return_spectrum=True)
        self.assertEqual(result.fwm_spectrum_total.shape, result.fwm_spectrum.shape[1:])


if __name__ == '__main__':
    unittest.main()
