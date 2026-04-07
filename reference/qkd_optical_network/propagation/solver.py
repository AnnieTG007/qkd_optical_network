"""
单跨段噪声求解器

集成 FWM 和拉曼噪声计算，输出各信道的总噪声功率。

统一使用 compute_noise() 接口，根据 channels[*].spectrum_type
自动选择离散或连续噪声模型。
"""

from typing import List, Optional
import numpy as np

from physics.fiber import Fiber
from physics.signal import WDMChannel
from physics.noise import compute_noise
from physics.noise.dispatcher import NoiseResult


class SingleSpanSolver:
    """
    单跨段光纤传输求解器

    计算单根光纤中传输后的总噪声功率（FWM + 拉曼）。
    噪声模型由 channels[*].spectrum_type 自动决定：
    - SINGLE_FREQ → 离散模型
    - 连续类型（RECTANGULAR / RAISED_COSINE / OSA_SAMPLED）→ 连续模型

    Attributes
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        输入 WDM 信道列表
    enable_fwm : bool
        是否启用 FWM 噪声计算
    enable_raman : bool
        是否启用拉曼噪声计算
    freq_resolution : float
        连续模式频率分辨率 [Hz]（仅对连续模型有效）

    Examples
    --------
    >>> fiber = Fiber(fiber_type=FiberType.SSMF, length=50e3)
    >>> channels = [...]  # 构建 WDM 信道
    >>> solver = SingleSpanSolver(fiber, channels)
    >>> result = solver.compute()
    >>> print(result.fwm_power)   # 每信道 FWM 噪声功率 [W]
    >>> print(result.raman_power) # 每信道拉曼噪声功率 [W]
    """

    def __init__(
        self,
        fiber: Fiber,
        channels: List[WDMChannel],
        enable_fwm: bool = True,
        enable_raman: bool = True,
        freq_resolution: float = 1e9,
    ):
        """
        初始化求解器

        Parameters
        ----------
        fiber : Fiber
            光纤对象
        channels : List[WDMChannel]
            输入 WDM 信道列表
        enable_fwm : bool, optional
            是否启用 FWM 噪声计算，默认 True
        enable_raman : bool, optional
            是否启用拉曼噪声计算，默认 True
        freq_resolution : float, optional
            连续模式频率分辨率 [Hz]，默认 1e9（1 GHz）。
            仅对 RECTANGULAR / RAISED_COSINE / OSA_SAMPLED 信道有效。
        """
        self.fiber = fiber
        self.channels = channels
        self.enable_fwm = enable_fwm
        self.enable_raman = enable_raman
        self.freq_resolution = freq_resolution

        # 缓存计算结果（NoiseResult）
        self._noise_result: Optional[NoiseResult] = None

    def compute(self, reset_cache: bool = True) -> NoiseResult:
        """
        计算总噪声功率

        Parameters
        ----------
        reset_cache : bool, optional
            是否重置缓存并重新计算，默认 True

        Returns
        -------
        NoiseResult
            统一噪声计算结果，包含：
            - fwm_power: 每信道 FWM 噪声功率 [W]
            - raman_power: 每信道拉曼噪声功率 [W]
            - total_power: 每信道总噪声功率 [W]
            - 连续模式额外包含 freq_array, df, fwm_spectrum, raman_spectrum
        """
        if reset_cache or self._noise_result is None:
            self._noise_result = compute_noise(
                fiber=self.fiber,
                channels=self.channels,
                enable_fwm=self.enable_fwm,
                enable_raman=self.enable_raman,
                freq_resolution=self.freq_resolution,
                return_spectrum=False,
            )

        return self._noise_result

    def get_fwm_noise(self) -> np.ndarray:
        """
        获取 FWM 噪声功率

        Returns
        -------
        fwm_noise : np.ndarray
            FWM 噪声功率 [W]，shape: (n_channels,)
        """
        result = self.compute(reset_cache=False)
        return result.fwm_power

    def get_raman_noise(self) -> np.ndarray:
        """
        获取拉曼噪声功率

        Returns
        -------
        raman_noise : np.ndarray
            拉曼噪声功率 [W]，shape: (n_channels,)
        """
        result = self.compute(reset_cache=False)
        return result.raman_power

    def get_snr(self, signal_powers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算各信道的信噪比（SNR）

        Parameters
        ----------
        signal_powers : np.ndarray, optional
            各信道的信号功率 [W]。如果为 None，使用 channels 中的功率。

        Returns
        -------
        snr : np.ndarray
            信噪比（线性倍数，非 dB），shape: (n_channels,)
        """
        if signal_powers is None:
            signal_powers = np.array([ch.power for ch in self.channels])

        result = self.compute(reset_cache=False)
        noise_powers = result.total_power

        # 避免除零
        snr = np.where(
            noise_powers > 1e-20,
            signal_powers / noise_powers,
            np.inf
        )

        return snr

    def get_snr_db(self, signal_powers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算各信道的信噪比（dB）

        Parameters
        ----------
        signal_powers : np.ndarray, optional
            各信道的信号功率 [W]。如果为 None，使用 channels 中的功率。

        Returns
        -------
        snr_db : np.ndarray
            信噪比 [dB]，shape: (n_channels,)
        """
        snr_linear = self.get_snr(signal_powers)
        return 10 * np.log10(snr_linear)

    def report(self) -> str:
        """
        生成噪声计算报告

        Returns
        -------
        report : str
            噪声计算报告（人类可读格式）
        """
        result = self.compute(reset_cache=False)

        lines = [
            "=" * 60,
            "单跨段噪声计算报告",
            "=" * 60,
            f"光纤：{self.fiber}",
            f"信道数：{len(self.channels)}",
            f"噪声模式：{result.mode}",
            f"FWM 噪声：{'启用' if self.enable_fwm else '禁用'}",
            f"拉曼噪声：{'启用' if self.enable_raman else '禁用'}",
            "-" * 60,
        ]

        if len(self.channels) <= 10:
            # 信道数少时，逐信道报告
            lines.append(f"{'信道':<6} {'频率 [THz]':<12} {'信号 [dBm]':<12} {'FWM [dBm]':<12} {'Raman [dBm]':<12} {'总噪声 [dBm]':<14}")
            lines.append("-" * 60)

            for i, ch in enumerate(self.channels):
                freq_thz = ch.center_freq / 1e12
                signal_dbm = 10 * np.log10(ch.power / 1e-3)
                fwm_dbm = 10 * np.log10(max(result.fwm_power[i], 1e-20) / 1e-3)
                raman_dbm = 10 * np.log10(max(result.raman_power[i], 1e-20) / 1e-3)
                total_dbm = 10 * np.log10(max(result.total_power[i], 1e-20) / 1e-3)
                lines.append(f"{i:<6} {freq_thz:<12.3f} {signal_dbm:<12.2f} {fwm_dbm:<12.2f} {raman_dbm:<12.2f} {total_dbm:<14.2f}")
        else:
            # 信道数多时，只报告统计信息
            total_noise = result.total_power
            lines.append(f"FWM 噪声功率范围：{10*np.log10(np.min(result.fwm_power)/1e-3):.2f} ~ {10*np.log10(np.max(result.fwm_power)/1e-3):.2f} dBm")
            lines.append(f"拉曼噪声功率范围：{10*np.log10(np.min(result.raman_power)/1e-3):.2f} ~ {10*np.log10(np.max(result.raman_power)/1e-3):.2f} dBm")
            lines.append(f"总噪声功率范围：{10*np.log10(np.min(total_noise)/1e-3):.2f} ~ {10*np.log10(np.max(total_noise)/1e-3):.2f} dBm")

        lines.append("=" * 60)

        return "\n".join(lines)
