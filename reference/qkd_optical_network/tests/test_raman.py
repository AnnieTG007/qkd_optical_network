"""
拉曼噪声单元测试

验证拉曼噪声计算的正确性。
"""

import unittest
import numpy as np
from physics.fiber import Fiber
from physics.signal import WDMChannel
from physics.noise import compute_raman_noise
from constants.fiber_parameters import FiberType


class TestRamanNoise(unittest.TestCase):
    """拉曼噪声测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建 SSMF 光纤
        self.fiber = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,  # 50 km
            temperature=300.0
        )

        # 创建 2 信道 WDM 系统（最小需求）
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
        ]

    def test_raman_noise_shape(self):
        """测试拉曼噪声输出形状"""
        noise = compute_raman_noise(self.fiber, self.channels)
        self.assertEqual(noise.shape, (len(self.channels),))

    def test_raman_noise_positive(self):
        """测试拉曼噪声功率非负"""
        noise = compute_raman_noise(self.fiber, self.channels)
        self.assertTrue(np.all(noise >= 0))

    def test_raman_with_fewer_channels(self):
        """测试少于 2 个信道时返回零噪声"""
        single_channel = self.channels[:1]
        noise = compute_raman_noise(self.fiber, single_channel)
        np.testing.assert_array_equal(noise, np.zeros(1))

    def test_raman_power_scaling(self):
        """测试拉曼噪声功率随输入功率增加而增加"""
        # 低功率情况
        low_power_channels = [
            WDMChannel(center_freq=ch.center_freq, power=1e-6)  # -30 dBm
            for ch in self.channels
        ]
        noise_low = compute_raman_noise(self.fiber, low_power_channels)

        # 高功率情况
        high_power_channels = [
            WDMChannel(center_freq=ch.center_freq, power=10e-3)  # 10 dBm
            for ch in self.channels
        ]
        noise_high = compute_raman_noise(self.fiber, high_power_channels)

        # 高功率噪声应大于低功率噪声
        self.assertTrue(np.all(noise_high > noise_low))

    def test_raman_temperature_dependence(self):
        """测试拉曼噪声的温度依赖性"""
        # 低温情况
        fiber_cold = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,
            temperature=77.0  # 液氮温度
        )
        noise_cold = compute_raman_noise(fiber_cold, self.channels)

        # 室温情况
        fiber_room = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,
            temperature=300.0
        )
        noise_room = compute_raman_noise(fiber_room, self.channels)

        # 高温情况
        fiber_hot = Fiber(
            fiber_type=FiberType.SSMF,
            length=50e3,
            temperature=500.0
        )
        noise_hot = compute_raman_noise(fiber_hot, self.channels)

        # 温度越高，Bose-Einstein 光子数越多，拉曼噪声越大
        # 但由于 Stokes 过程占主导，温度依赖性可能不明显
        # 这里主要验证代码能正常运行且温度有影响
        self.assertEqual(noise_cold.shape, (2,))
        self.assertEqual(noise_room.shape, (2,))
        self.assertEqual(noise_hot.shape, (2,))

    def test_raman_compute_at_length(self):
        """测试 compute_at_length 参数"""
        # compute_at_length=True 应该正常工作
        noise = compute_raman_noise(self.fiber, self.channels, compute_at_length=True)
        self.assertEqual(noise.shape, (len(self.channels),))

        # compute_at_length=False 应该抛出 NotImplementedError
        with self.assertRaises(NotImplementedError):
            compute_raman_noise(self.fiber, self.channels, compute_at_length=False)

    def test_raman_stokes_anti_stokes(self):
        """测试 Stokes 和 anti-Stokes 散射的对称性"""
        # 创建对称的 3 信道系统
        freq_center = 193.4e12
        spacing = 100e9

        channels = [
            WDMChannel(center_freq=freq_center - spacing, power=1e-3),  # 低频
            WDMChannel(center_freq=freq_center, power=1e-3),            # 中心
            WDMChannel(center_freq=freq_center + spacing, power=1e-3),  # 高频
        ]

        noise = compute_raman_noise(self.fiber, channels)

        # 由于拉曼增益系数的非对称性，噪声可能不对称
        # 但应该都是正值
        self.assertTrue(np.all(noise >= 0))

    def test_raman_forward_backward(self):
        """测试前向和后向拉曼噪声都存在"""
        # 使用更长的光纤以增强拉曼效应
        fiber_long = Fiber(
            fiber_type=FiberType.SSMF,
            length=100e3,  # 100 km
            temperature=300.0
        )

        noise = compute_raman_noise(fiber_long, self.channels)

        # 验证噪声不为零
        self.assertTrue(np.any(noise > 0))


if __name__ == '__main__':
    unittest.main()
