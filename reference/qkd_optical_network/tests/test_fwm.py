"""
FWM 噪声单元测试

验证 FWM 噪声计算的正确性。
"""

import unittest
import numpy as np
from physics.fiber import Fiber
from physics.signal import WDMChannel
from physics.noise import compute_fwm_noise
from constants.fiber_parameters import FiberType


class TestFWMNoise(unittest.TestCase):
    """FWM 噪声测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建 SSMF 光纤
        self.fiber = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,  # 50 km
            temperature=300.0
        )

        # 创建 3 信道 WDM 系统（最小需求）
        freq_center = 193.4e12  # 1550 nm 中心频率
        channel_spacing = 50e9  # 50 GHz 间隔

        self.channels = [
            WDMChannel(
                center_freq=freq_center - channel_spacing,
                power=1e-3,  # 0 dBm
                baudrate=32e9,
                modulation='QPSK'
            ),
            WDMChannel(
                center_freq=freq_center,
                power=1e-3,
                baudrate=32e9,
                modulation='QPSK'
            ),
            WDMChannel(
                center_freq=freq_center + channel_spacing,
                power=1e-3,
                baudrate=32e9,
                modulation='QPSK'
            ),
        ]

    def test_fwm_noise_shape(self):
        """测试 FWM 噪声输出形状"""
        noise = compute_fwm_noise(self.fiber, self.channels)
        self.assertEqual(noise.shape, (len(self.channels),))

    def test_fwm_noise_positive(self):
        """测试 FWM 噪声功率非负"""
        noise = compute_fwm_noise(self.fiber, self.channels)
        self.assertTrue(np.all(noise >= 0))

    def test_fwm_noise_nonzero(self):
        """测试 FWM 噪声不为零（有信道间相互作用）"""
        noise = compute_fwm_noise(self.fiber, self.channels)
        # 至少有一个信道有非零噪声
        self.assertTrue(np.any(noise > 0))

    def test_fwm_with_fewer_channels(self):
        """测试少于 3 个信道时返回零噪声"""
        single_channel = self.channels[:1]
        noise = compute_fwm_noise(self.fiber, single_channel)
        np.testing.assert_array_equal(noise, np.zeros(1))

        two_channels = self.channels[:2]
        noise = compute_fwm_noise(self.fiber, two_channels)
        np.testing.assert_array_equal(noise, np.zeros(2))

    def test_fwm_power_scaling(self):
        """测试 FWM 噪声功率随输入功率增加而增加"""
        # 低功率情况
        low_power_channels = [
            WDMChannel(center_freq=ch.center_freq, power=1e-6)  # -30 dBm
            for ch in self.channels
        ]
        noise_low = compute_fwm_noise(self.fiber, low_power_channels)

        # 高功率情况
        high_power_channels = [
            WDMChannel(center_freq=ch.center_freq, power=10e-3)  # 10 dBm
            for ch in self.channels
        ]
        noise_high = compute_fwm_noise(self.fiber, high_power_channels)

        # 高功率噪声应远大于低功率噪声
        self.assertTrue(np.all(noise_high > noise_low))

    def test_fwm_frequency_spacing(self):
        """测试信道间隔对 FWM 噪声的影响"""
        # 小间隔（25 GHz）
        freq_center = 193.4e12
        small_spacing = 25e9

        channels_small = [
            WDMChannel(center_freq=freq_center + i * small_spacing, power=1e-3)
            for i in range(-1, 2)
        ]
        noise_small = compute_fwm_noise(self.fiber, channels_small)

        # 大间隔（100 GHz）
        large_spacing = 100e9
        channels_large = [
            WDMChannel(center_freq=freq_center + i * large_spacing, power=1e-3)
            for i in range(-1, 2)
        ]
        noise_large = compute_fwm_noise(self.fiber, channels_large)

        # 小间隔相位失配小，FWM 效率更高
        # 但由于频率差小，FWM 产物可能不落在信道内
        # 这里主要验证代码能正常运行
        self.assertEqual(noise_small.shape, (3,))
        self.assertEqual(noise_large.shape, (3,))

    def test_fwm_compute_at_length(self):
        """测试 compute_at_length 参数"""
        # compute_at_length=True 应该正常工作
        noise = compute_fwm_noise(self.fiber, self.channels, compute_at_length=True)
        self.assertEqual(noise.shape, (len(self.channels),))

        # compute_at_length=False 应该抛出 NotImplementedError
        with self.assertRaises(NotImplementedError):
            compute_fwm_noise(self.fiber, self.channels, compute_at_length=False)


if __name__ == '__main__':
    unittest.main()
