"""
信号表示模块

定义 WDM 信道、信号状态等数据结构。
支持两种输入方式：
1. 直接参数定义（GN-Model 范式：滚降系数 + 波特率）
2. OSA 频谱数据导入
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum
import numpy as np
from scipy.constants import c


class SpectrumType(Enum):
    """
    信号频谱类型枚举

    Attributes
    ----------
    SINGLE_FREQ : str
        单频模型（delta 函数），所有功率集中在中心频率
    RECTANGULAR : str
        矩形谱，在波特率带宽内均匀分布
    RAISED_COSINE : str
        升余弦滚降谱，平坦区 + 余弦滚降过渡带
    OSA_SAMPLED : str
        OSA 采样数据，从实际测量导入
    """
    SINGLE_FREQ = 'single_freq'      # 单频模型
    RECTANGULAR = 'rectangular'       # 矩形谱（波特率带宽内均匀）
    RAISED_COSINE = 'raised_cosine'   # 升余弦滚降谱
    OSA_SAMPLED = 'osa_sampled'       # OSA 采样数据


@dataclass
class WDMChannel:
    """
    WDM 信道数据类

    用于表示单个 WDM 信道的参数，包括中心频率、功率、带宽、调制格式等。

    支持多种频谱建模方式：
    - SINGLE_FREQ: 单频模型（delta 函数）
    - RECTANGULAR: 矩形谱（波特率带宽内均匀分布）
    - RAISED_COSINE: 升余弦滚降谱

    Attributes
    ----------
    center_freq : float
        中心频率 [Hz]
    power : float
        信道功率 [W]（线性单位，非 dBm）
    baudrate : float, optional
        符号率 [Baud]，用于计算带宽
    roll_off : float, optional
        滚降系数 (0~1)，默认 0.1
        - 0: 理想 Nyquist（矩形谱）
        - 0.1~0.5: 升余弦滚降
    spectrum_type : SpectrumType or str, optional
        频谱类型，默认 RAISED_COSINE
        - 'single_freq' 或 SpectrumType.SINGLE_FREQ: 单频模型
        - 'rectangular' 或 SpectrumType.RECTANGULAR: 矩形谱
        - 'raised_cosine' 或 SpectrumType.RAISED_COSINE: 升余弦滚降谱
    modulation : str, optional
        调制格式，如 'QPSK', '16QAM', '64QAM'
    channel_id : int, optional
        ITU 信道编号（可选）
    direction : {'forward', 'backward'}, optional
        传播方向（'forward' 为 +z 方向，'backward' 为 -z 方向）

    Examples
    --------
    >>> # 单频模型
    >>> ch_single = WDMChannel(center_freq=193.4e12, power=1e-3,
    ...                        spectrum_type='single_freq')
    >>>
    >>> # 矩形谱（波特率带宽内均匀）
    >>> ch_rect = WDMChannel(center_freq=193.4e12, power=1e-3,
    ...                      baudrate=32e9, spectrum_type='rectangular')
    >>>
    >>> # 升余弦滚降谱
    >>> ch_rc = WDMChannel(center_freq=193.4e12, power=1e-3,
    ...                    baudrate=32e9, roll_off=0.1,
    ...                    spectrum_type='raised_cosine')
    """

    # 必需参数
    center_freq: float  # 中心频率 [Hz]
    power: float  # 信道功率 [W]

    # 可选参数（带宽相关）
    baudrate: float = 32e9  # 符号率 [Baud]
    roll_off: float = 0.1  # 滚降系数
    spectrum_type: SpectrumType = SpectrumType.RAISED_COSINE  # 频谱类型
    modulation: str = 'QPSK'  # 调制格式
    channel_id: Optional[int] = None  # ITU 信道编号
    direction: Literal['forward', 'backward'] = 'forward'  # 传播方向

    def __post_init__(self):
        """数据类初始化后的验证"""
        if self.center_freq <= 0:
            raise ValueError(f"center_freq must be positive, got {self.center_freq}")
        if self.power < 0:
            raise ValueError(f"power must be non-negative, got {self.power}")
        if not 0 <= self.roll_off <= 1:
            raise ValueError(f"roll_off must be in [0, 1], got {self.roll_off}")
        if self.baudrate <= 0:
            raise ValueError(f"baudrate must be positive, got {self.baudrate}")

        # 处理 spectrum_type 为字符串的情况
        if isinstance(self.spectrum_type, str):
            self.spectrum_type = SpectrumType(self.spectrum_type)

    @property
    def bandwidth(self) -> float:
        """
        信道带宽 [Hz]

        对于不同谱形类型：
        - SINGLE_FREQ: 带宽为 0（delta 函数）
        - RECTANGULAR: 带宽 = baudrate
        - RAISED_COSINE: 带宽 = (1 + roll_off) * baudrate
        """
        if self.spectrum_type == SpectrumType.SINGLE_FREQ:
            return 0.0
        elif self.spectrum_type == SpectrumType.RECTANGULAR:
            return self.baudrate
        else:  # RAISED_COSINE
            return (1 + self.roll_off) * self.baudrate

    @property
    def wavelength(self) -> float:
        """波长 [m]"""
        return c / self.center_freq

    @property
    def power_dbm(self) -> float:
        """功率 [dBm]（只读属性，用于与实验数据对比）"""
        return 10 * np.log10(self.power / 1e-3)

    def get_psd(self, freq_array: np.ndarray, power: Optional[float] = None) -> np.ndarray:
        """
        计算该信道在给定频率点上的功率谱密度 (PSD)

        根据 spectrum_type 选择不同的谱形模型：
        - SINGLE_FREQ: 单频模型 (delta 函数近似)
        - RECTANGULAR: 矩形谱，在波特率带宽内均匀分布
        - RAISED_COSINE: 升余弦滚降谱

        Parameters
        ----------
        freq_array : np.ndarray
            频率数组 [Hz]
        power : float, optional
            临时功率 [W]。如果提供，使用此功率而非 self.power 计算 PSD。
            用于 SignalState 等场景中计算某位置的功率谱而不修改信道对象。

        Returns
        -------
        psd : np.ndarray
            功率谱密度 [W/Hz]，与 freq_array 同形状

        Notes
        -----
        PSD 归一化公式参考 FORMULAS_REVISION.md 公式 (27)-(28)
        """
        # 使用传入的临时功率，或使用对象的 self.power
        p = power if power is not None else self.power

        if self.spectrum_type == SpectrumType.SINGLE_FREQ:
            return self._get_psd_single_freq(freq_array, p)
        elif self.spectrum_type == SpectrumType.RECTANGULAR:
            return self._get_psd_rectangular(freq_array, p)
        else:  # RAISED_COSINE
            return self._get_psd_raised_cosine(freq_array, p)

    def _get_psd_single_freq(self, freq_array: np.ndarray, power: float) -> np.ndarray:
        """
        单频模型（delta 函数近似）

        将所有功率集中在中心频率处。
        数值实现：在最近中心频率的点放置全部功率。

        Parameters
        ----------
        freq_array : np.ndarray
            频率数组 [Hz]
        power : float
            信道功率 [W]

        Returns
        -------
        psd : np.ndarray
            功率谱密度 [W/Hz]
        """
        psd = np.zeros_like(freq_array, dtype=np.float64)

        # 找到最接近中心频率的点
        idx_center = np.argmin(np.abs(freq_array - self.center_freq))

        # 如果频率数组只有一个点
        if len(freq_array) == 1:
            # 无法定义 delta 函数，返回平均 PSD
            return np.array([power])

        # 计算频率分辨率
        df = np.min(np.abs(np.diff(freq_array)))

        # 在中心频率处放置总功率（delta 函数近似）
        psd[idx_center] = power / df

        return psd

    def _get_psd_rectangular(self, freq_array: np.ndarray, power: float) -> np.ndarray:
        """
        矩形谱模型

        在波特率带宽内均匀分布功率。
        - 带宽 = baudrate（注意：不考虑滚降系数）
        - PSD = power / baudrate（在带宽内）

        Parameters
        ----------
        freq_array : np.ndarray
            频率数组 [Hz]
        power : float
            信道功率 [W]

        Returns
        -------
        psd : np.ndarray
            功率谱密度 [W/Hz]
        """
        psd = np.zeros_like(freq_array, dtype=np.float64)

        # 频率偏移（相对于中心频率）
        df = np.abs(freq_array - self.center_freq)

        # 带宽 = baudrate（矩形谱不考虑滚降）
        half_bw = self.baudrate / 2

        # 在带宽内均匀分布
        in_band_mask = df <= half_bw
        psd[in_band_mask] = power / self.baudrate

        return psd

    def _get_psd_raised_cosine(self, freq_array: np.ndarray, power: float) -> np.ndarray:
        """
        升余弦滚降谱模型（解析归一化）

        使用升余弦滚降滤波器模型，采用解析归一化：
        - PSD(f) = (P_total / R_s) * H(f)
        - 参考 FORMULAS_REVISION.md 公式 (26)-(28)

        Parameters
        ----------
        freq_array : np.ndarray
            频率数组 [Hz]
        power : float
            信道功率 [W]

        Returns
        -------
        psd : np.ndarray
            功率谱密度 [W/Hz]
        """
        # 频率偏移（相对于中心频率）
        df = np.abs(freq_array - self.center_freq)

        # 升余弦谱的关键频率点
        f1 = (1 - self.roll_off) * self.baudrate / 2  # 平坦区边界
        f2 = (1 + self.roll_off) * self.baudrate / 2  # 截止频率

        # 计算升余弦谱形状 H(f)
        h_f = np.zeros_like(freq_array, dtype=np.float64)

        # 平坦区
        flat_mask = df <= f1
        h_f[flat_mask] = 1.0

        # 滚降区
        roll_mask = (df > f1) & (df <= f2)
        if self.roll_off > 0:
            arg = np.pi * (df[roll_mask] - f1) / (f2 - f1)
            h_f[roll_mask] = 0.5 * (1 + np.cos(arg))

        # 解析归一化：PSD = (P_total / R_s) * H(f)
        # 参考公式 (28)
        psd = h_f * (power / self.baudrate)

        return psd

    def get_capacity(self, snr: Optional[float] = None) -> float:
        """
        计算信道的理论容量（考虑调制格式和滚降系数）

        参考 FORMULAS_REVISION.md 公式 (31)：
        C = R_s × log₂(M) × (1 - α) [bps]

        Parameters
        ----------
        snr : float, optional
            信噪比 [线性单位]。如果提供，使用香农公式 (29) 计算实际容量限制。

        Returns
        -------
        capacity : float
            信道容量 [bps]

        Notes
        -----
        支持调制格式：
        - QPSK: M=4
        - 16QAM: M=16
        - 64QAM: M=64
        - 256QAM: M=256

        如果提供 snr 参数，返回 min(公式 (31), 公式 (29))
        """
        # 调制阶数映射
        modulation_map = {
            'QPSK': 4,
            '16QAM': 16,
            '64QAM': 64,
            '256QAM': 256,
            '1024QAM': 1024,
        }

        M = modulation_map.get(self.modulation.upper(), 4)  # 默认 QPSK
        spectral_eff = np.log2(M) * (1 - self.roll_off)  # bps/Hz

        # 公式 (31): C = R_s × log₂(M) × (1 - α)
        capacity = self.baudrate * spectral_eff

        # 如果提供 SNR，使用香农公式限制
        if snr is not None and snr > 0:
            shannon_capacity = self.bandwidth * np.log2(1 + snr) if self.bandwidth > 0 else 0
            capacity = min(capacity, shannon_capacity)

        return capacity

    def __repr__(self):
        bw_str = f"B={self.bandwidth/1e9:.2f} GHz" if self.bandwidth > 0 else "B=0 (single-freq)"
        return (f"WDMChannel(f={self.center_freq/1e12:.3f} THz, "
                f"P={self.power_dbm:.2f} dBm, {bw_str}, "
                f"type={self.spectrum_type.value})")


@dataclass
class SignalState:
    """
    信号状态数据类

    用于表示沿光纤传播时的信号状态（多信道、多位置）。

    Attributes
    ----------
    channels : List[WDMChannel]
        WDM 信道列表
    powers : np.ndarray
        各信道在各位置的功率 [W]
        shape: (n_channels, n_positions)
    z_positions : np.ndarray
        光纤位置数组 [m]
        shape: (n_positions,)
    freq_array : np.ndarray, optional
        频率采样数组 [Hz]（用于带宽模型的 PSD 表示）

    Examples
    --------
    >>> channels = [WDMChannel(...), WDMChannel(...)]
    >>> z = np.array([0, 10e3, 20e3, 50e3])  # 4 个位置
    >>> powers = np.random.rand(len(channels), 4) * 1e-3
    >>> state = SignalState(channels, powers, z)
    >>> state.get_power_at_z(50e3)  # 获取末端功率
    """

    channels: List[WDMChannel]
    powers: np.ndarray  # shape: (n_channels, n_positions)
    z_positions: np.ndarray  # shape: (n_positions,)
    freq_array: Optional[np.ndarray] = None  # 频率采样数组

    def __post_init__(self):
        """验证输入形状"""
        if len(self.channels) != self.powers.shape[0]:
            raise ValueError(
                f"channels length ({len(self.channels)}) != powers.shape[0] ({self.powers.shape[0]})"
            )
        if len(self.z_positions) != self.powers.shape[1]:
            raise ValueError(
                f"z_positions length ({len(self.z_positions)}) != powers.shape[1] ({self.powers.shape[1]})"
            )

    @property
    def n_channels(self) -> int:
        """信道数量"""
        return len(self.channels)

    @property
    def n_positions(self) -> int:
        """位置数量"""
        return len(self.z_positions)

    def get_power_at_z(self, z: float) -> np.ndarray:
        """
        获取指定位置 z 处的各信道功率

        Parameters
        ----------
        z : float
            光纤位置 [m]

        Returns
        -------
        powers : np.ndarray
            各信道功率 [W]，shape: (n_channels,)
        """
        # 找到最近的 z 位置
        idx = np.argmin(np.abs(self.z_positions - z))
        return self.powers[:, idx]

    def get_power_spectrum(self, position_idx: int = -1) -> np.ndarray:
        """
        获取指定位置的功率谱（频域表示）

        Parameters
        ----------
        position_idx : int, optional
            z_positions 的索引，-1 表示最后一个位置（光纤末端）

        Returns
        -------
        freq_array : np.ndarray
            频率数组 [Hz]
        psd_total : np.ndarray
            总 PSD [W/Hz]（所有信道叠加）
        """
        if self.freq_array is None:
            # 自动生成频率数组（覆盖所有信道带宽）
            f_min = min(ch.center_freq - ch.bandwidth for ch in self.channels)
            f_max = max(ch.center_freq + ch.bandwidth for ch in self.channels)
            self.freq_array = np.linspace(f_min, f_max, 10000)

        psd_total = np.zeros_like(self.freq_array, dtype=np.float64)
        powers = self.powers[:, position_idx]

        for ch, power in zip(self.channels, powers):
            # 使用临时功率计算 PSD，不修改信道对象
            psd_total += ch.get_psd(self.freq_array, power=power)

        return self.freq_array, psd_total


def load_osa_csv(filepath: str, encoding: str = 'utf-8') -> tuple:
    """
    加载 OSA（光谱仪）导出的 CSV 数据

    通用解析器，支持多种格式：
    - 分隔符：逗号、制表符、分号自动检测
    - 波长单位：nm 或 μm 自动检测
    - 功率单位：dBm 或 W 自动检测

    Parameters
    ----------
    filepath : str
        CSV 文件路径
    encoding : str, optional
        文件编码（默认 'utf-8'，可尝试 'gbk'）

    Returns
    -------
    wavelength_nm : np.ndarray
        波长数组 [nm]
    power_dbm : np.ndarray
        功率数组 [dBm]

    Examples
    --------
    >>> wl, pwr = load_osa_csv('spectrum.csv')
    >>> channels = osa_to_channels(wl, pwr, channel_spacing_GHz=50)
    """
    # 尝试不同的分隔符和编码
    delimiters = [',', '\t', ';']
    data = None

    for delim in delimiters:
        try:
            data = np.genfromtxt(filepath, delimiter=delim, encoding=encoding, skip_header=1)
            if data.ndim == 2 and data.shape[1] >= 2:
                break
        except:
            data = None

    if data is None:
        # 尝试无表头的情况
        for delim in delimiters:
            try:
                data = np.genfromtxt(filepath, delimiter=delim, encoding=encoding)
                if data.ndim == 2 and data.shape[1] >= 2:
                    break
            except:
                data = None

    if data is None or data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Failed to parse CSV file: {filepath}")

    # 提取两列数据
    col1 = data[:, 0]
    col2 = data[:, 1]

    # 判断波长单位（nm 或 μm）
    # 如果数值范围在 1.5-1.6，则是 μm；如果在 1500-1600，则是 nm
    if np.mean(col1) < 10:
        wavelength_nm = col1 * 1000  # μm → nm
    else:
        wavelength_nm = col1

    # 判断功率单位（dBm 或 W）
    # 如果数值范围在 -30 到 10，则是 dBm；如果在 1e-6 到 1e-1，则是 W
    if np.max(col2) < 20 and np.min(col2) > -50:
        power_dbm = col2
    else:
        power_dbm = 10 * np.log10(col2 / 1e-3)  # W → dBm

    return wavelength_nm, power_dbm


def osa_to_channels(
    wavelength_nm: np.ndarray,
    power_dbm: np.ndarray,
    channel_spacing_GHz: float = 50,
    baudrate_GHz: float = 32,
    roll_off: float = 0.1
) -> List[WDMChannel]:
    """
    将 OSA 频谱数据转换为 WDMChannel 列表

    通过寻峰算法识别 WDM 信道，然后创建 WDMChannel 对象。

    Parameters
    ----------
    wavelength_nm : np.ndarray
        波长数组 [nm]
    power_dbm : np.ndarray
        功率数组 [dBm]
    channel_spacing_GHz : float, optional
        信道标称间隔 [GHz]，用于识别相邻峰
    baudrate_GHz : float, optional
        每个信道的符号率 [GHz]
    roll_off : float, optional
        滚降系数

    Returns
    -------
    channels : List[WDMChannel]
        WDM 信道列表

    Examples
    --------
    >>> wl, pwr = load_osa_csv('spectrum.csv')
    >>> channels = osa_to_channels(wl, pwr, channel_spacing_GHz=50)
    """
    from scipy.signal import find_peaks

    # 波长转频率
    freq_THz = c / (wavelength_nm * 1e-9) / 1e12  # Hz → THz

    # 寻峰（功率局部最大值）
    # prominence: 峰的显著度，避免检测到噪声尖峰
    peaks, properties = find_peaks(power_dbm, prominence=3, distance=5)

    channels = []
    for peak_idx in peaks:
        freq = freq_THz[peak_idx] * 1e12  # THz → Hz
        power = 10 ** (power_dbm[peak_idx] / 10) * 1e-3  # dBm → W

        ch = WDMChannel(
            center_freq=freq,
            power=power,
            baudrate=baudrate_GHz * 1e9,
            roll_off=roll_off
        )
        channels.append(ch)

    # 按频率排序
    channels.sort(key=lambda ch: ch.center_freq)

    return channels
