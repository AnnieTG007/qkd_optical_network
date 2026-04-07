"""
噪声功率谱计算与输出脚本

参考 tool.py 的验证方式，调用噪声计算函数计算 FWM 和拉曼噪声功率，
输出噪声功率谱（PSD）数据，支持 CSV 导出和可视化。

使用方法:
--------
python validation/spectrum_computation/compute_noise_spectrum.py

输出:
----
- output/noise_spectra/noise_spectrum_fwm.csv   - FWM 噪声功率谱
- output/noise_spectra/noise_spectrum_raman.csv - 拉曼噪声功率谱
- output/noise_spectra/noise_spectrum_total.csv - 总噪声功率谱
- output/spectrum_models/signal_distribution.png  - 信号频域分布图
- output/spectrum_models/fwm_spectrum.png         - FWM 噪声谱图
- output/spectrum_models/raman_spectrum.png       - 拉曼噪声谱图
"""

import numpy as np
import os
from typing import List, Tuple

# 添加项目根目录到路径（当前文件在 validation/spectrum_computation/下）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics.fiber import Fiber
from physics.signal import WDMChannel
from physics.noise import compute_noise
from constants.fiber_parameters import FiberType


def build_wdm_system(
    f_min: float = 191.0e12,
    f_max: float = 195.0e12,
    spacing: float = 50e9,
    classic_channels: List[float] = None,  # 经典信道频率列表 [Hz]
    classic_power: float = 1e-3,  # 0 dBm
    quantum_power: float = 1e-11  # -80 dBm，量子信号功率
) -> List[WDMChannel]:
    """
    构建 WDM 系统，包含经典信道和量子信道

    Parameters
    ----------
    f_min : float
        最小频率 [Hz]
    f_max : float
        最大频率 [Hz]
    spacing : float
        信道间隔 [Hz]
    classic_channels : List[float]
        经典信道频率列表 [Hz]。如果为 None，则无经典信道
    classic_power : float
        经典信道功率 [W]
    quantum_power : float
        量子信道功率 [W]

    Returns
    -------
    channels : List[WDMChannel]
        WDM 信道列表
    """
    channels = []
    freq_array = np.arange(f_min, f_max, spacing)

    # 经典信道频率集合（用于快速查找）
    classic_set = set(classic_channels) if classic_channels else set()

    for freq in freq_array:
        # 判断是否为经典信道
        if freq in classic_set:
            # 经典信道（高功率，作为泵浦）
            channels.append(WDMChannel(
                center_freq=freq,
                power=classic_power,
                baudrate=32e9,
                modulation='QPSK',
                direction='forward'
            ))
        else:
            # 量子信道（低功率，作为信号）
            channels.append(WDMChannel(
                center_freq=freq,
                power=quantum_power,
                baudrate=32e9,
                modulation='QPSK',
                direction='forward'
            ))

    return channels


def compute_noise_spectrum(
    fiber: Fiber,
    channels: List[WDMChannel]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算噪声功率谱

    使用 compute_noise() 统一接口。

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : List[WDMChannel]
        WDM 信道列表

    Returns
    -------
    freq_array : np.ndarray
        频率数组 [Hz]
    fwm_noise : np.ndarray
        FWM 噪声功率 [W]
    raman_noise : np.ndarray
        拉曼噪声功率 [W]
    total_noise : np.ndarray
        总噪声功率 [W]
    """
    # 使用统一接口计算噪声
    print("计算噪声（离散模型）...")
    result = compute_noise(fiber, channels)

    # 频率数组（离散模式为信道中心频率）
    freq_array = np.array([ch.center_freq for ch in channels])

    return freq_array, result.fwm_power, result.raman_power, result.total_power


def export_to_csv(
    freq_array: np.ndarray,
    fwm_noise: np.ndarray,
    raman_noise: np.ndarray,
    total_noise: np.ndarray,
    output_dir: str = "output/noise_spectra"
):
    """
    导出噪声功率谱到 CSV 文件

    Parameters
    ----------
    freq_array : np.ndarray
        频率数组 [Hz]
    fwm_noise : np.ndarray
        FWM 噪声功率 [W]
    raman_noise : np.ndarray
        拉曼噪声功率 [W]
    total_noise : np.ndarray
        总噪声功率 [W]
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 转换为常用单位
    freq_thz = freq_array / 1e12  # THz
    fwm_dbm = 10 * np.log10(np.maximum(fwm_noise, 1e-30) / 1e-3)  # dBm
    raman_dbm = 10 * np.log10(np.maximum(raman_noise, 1e-30) / 1e-3)  # dBm
    total_dbm = 10 * np.log10(np.maximum(total_noise, 1e-30) / 1e-3)  # dBm

    # 导出 FWM 噪声谱
    fwm_file = os.path.join(output_dir, "noise_spectrum_fwm.csv")
    np.savetxt(
        fwm_file,
        np.column_stack([freq_thz, fwm_dbm]),
        delimiter=',',
        header='Frequency[THz],FWM_Noise[dBm]',
        fmt='%.6f,%.6f'
    )
    print(f"FWM 噪声谱已导出：{fwm_file}")

    # 导出拉曼噪声谱
    raman_file = os.path.join(output_dir, "noise_spectrum_raman.csv")
    np.savetxt(
        raman_file,
        np.column_stack([freq_thz, raman_dbm]),
        delimiter=',',
        header='Frequency[THz],Raman_Noise[dBm]',
        fmt='%.6f,%.6f'
    )
    print(f"拉曼噪声谱已导出：{raman_file}")

    # 导出总噪声谱
    total_file = os.path.join(output_dir, "noise_spectrum_total.csv")
    np.savetxt(
        total_file,
        np.column_stack([freq_thz, total_dbm]),
        delimiter=',',
        header='Frequency[THz],Total_Noise[dBm]',
        fmt='%.6f,%.6f'
    )
    print(f"总噪声谱已导出：{total_file}")


def plot_signal_distribution(
    channels: List[WDMChannel],
    output_dir: str = "output/noise_spectra"
):
    """
    绘制信号频域分布图（经典 + 量子）

    Parameters
    ----------
    channels : List[WDMChannel]
        WDM 信道列表
    output_dir : str
        输出目录
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    freq_array = np.array([ch.center_freq for ch in channels])
    freq_thz = freq_array / 1e12
    power_dbm = np.array([10 * np.log10(ch.power / 1e-3) for ch in channels])

    # 标记经典信道和量子信道
    classic_mask = np.array([ch.power > 1e-6 for ch in channels])
    quantum_mask = ~classic_mask

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制经典信道（用红色竖线）
    if np.any(classic_mask):
        ax.vlines(freq_thz[classic_mask], -90, power_dbm[classic_mask],
                  colors='red', linestyles='solid', linewidth=2, label='Classic Signal (0 dBm)')

    # 绘制量子信道（用蓝色竖线）
    if np.any(quantum_mask):
        ax.vlines(freq_thz[quantum_mask], -90, power_dbm[quantum_mask],
                  colors='blue', linestyles='solid', linewidth=1, label='Quantum Signal (-80 dBm)')

    ax.set_xlabel('Frequency [THz]')
    ax.set_ylabel('Power [dBm]')
    ax.set_title('Signal Distribution in Frequency Domain')
    ax.set_ylim(-90, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 保存图片
    plot_file = os.path.join(output_dir, "signal_distribution.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"信号分布图已保存：{plot_file}")

    plt.close()


def plot_fwm_spectrum(
    freq_array: np.ndarray,
    fwm_noise: np.ndarray,
    channels: List[WDMChannel],
    output_dir: str = "output/noise_spectra"
):
    """
    绘制 FWM 噪声功率谱图

    Parameters
    ----------
    freq_array : np.ndarray
        频率数组 [Hz]
    fwm_noise : np.ndarray
        FWM 噪声功率 [W]
    channels : List[WDMChannel]
        WDM 信道列表
    output_dir : str
        输出目录
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    freq_thz = freq_array / 1e12
    fwm_dbm = 10 * np.log10(np.maximum(fwm_noise, 1e-30) / 1e-3)

    # 标记经典信道位置
    classic_mask = np.array([ch.power > 1e-6 for ch in channels])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1: 线性坐标
    ax1.plot(freq_thz, fwm_dbm, 'g-', linewidth=1.5, label='FWM Noise')
    if np.any(classic_mask):
        ax1.vlines(freq_thz[classic_mask], -120, 5, colors='red',
                   linestyles='dashed', linewidth=1.5, label='Classic Channel')
    ax1.set_xlabel('Frequency [THz]')
    ax1.set_ylabel('Noise Power [dBm]')
    ax1.set_title('FWM Noise Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-120, 5)

    # 子图 2: 对数坐标
    ax2.semilogy(freq_thz, np.maximum(fwm_noise, 1e-30), 'g-', linewidth=1.5, label='FWM Noise')
    if np.any(classic_mask):
        ax2.vlines(freq_thz[classic_mask], 1e-33, 1e-2, colors='red',
                   linestyles='dashed', linewidth=1.5, label='Classic Channel')
    ax2.set_xlabel('Frequency [THz]')
    ax2.set_ylabel('Noise Power [W] (log scale)')
    ax2.set_title('FWM Noise Spectrum (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, "fwm_spectrum.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"FWM 噪声谱图已保存：{plot_file}")

    plt.close()


def plot_raman_spectrum(
    freq_array: np.ndarray,
    raman_noise: np.ndarray,
    channels: List[WDMChannel],
    output_dir: str = "output/noise_spectra"
):
    """
    绘制拉曼噪声功率谱图

    Parameters
    ----------
    freq_array : np.ndarray
        频率数组 [Hz]
    raman_noise : np.ndarray
        拉曼噪声功率 [W]
    channels : List[WDMChannel]
        WDM 信道列表
    output_dir : str
        输出目录
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    freq_thz = freq_array / 1e12
    raman_dbm = 10 * np.log10(np.maximum(raman_noise, 1e-30) / 1e-3)

    # 标记经典信道位置
    classic_mask = np.array([ch.power > 1e-6 for ch in channels])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1: 线性坐标
    ax1.plot(freq_thz, raman_dbm, 'r-', linewidth=1.5, label='Raman Noise')
    if np.any(classic_mask):
        ax1.vlines(freq_thz[classic_mask], -120, 5, colors='blue',
                   linestyles='dashed', linewidth=1.5, label='Classic Channel')
    ax1.set_xlabel('Frequency [THz]')
    ax1.set_ylabel('Noise Power [dBm]')
    ax1.set_title('Raman Noise Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-120, 5)

    # 子图 2: 对数坐标
    ax2.semilogy(freq_thz, np.maximum(raman_noise, 1e-30), 'r-', linewidth=1.5, label='Raman Noise')
    if np.any(classic_mask):
        ax2.vlines(freq_thz[classic_mask], 1e-33, 1e-2, colors='blue',
                   linestyles='dashed', linewidth=1.5, label='Classic Channel')
    ax2.set_xlabel('Frequency [THz]')
    ax2.set_ylabel('Noise Power [W] (log scale)')
    ax2.set_title('Raman Noise Spectrum (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, "raman_spectrum.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"拉曼噪声谱图已保存：{plot_file}")

    plt.close()


def run_test_case(
    case_name: str,
    classic_channels: List[float],
    fiber: Fiber
):
    """
    运行单个测试用例

    Parameters
    ----------
    case_name : str
        测试用例名称
    classic_channels : List[float]
        经典信道频率列表 [Hz]
    fiber : Fiber
        光纤对象
    """
    print("\n" + "=" * 60)
    print(f"测试用例：{case_name}")
    print("=" * 60)

    # 构建 WDM 系统
    channels = build_wdm_system(
        f_min=191.0e12,
        f_max=195.0e12,
        spacing=50e9,
        classic_channels=classic_channels,
        classic_power=1e-3,  # 0 dBm
        quantum_power=1e-11  # -80 dBm
    )

    n_classic = sum(1 for ch in channels if ch.power > 1e-6)
    n_quantum = sum(1 for ch in channels if ch.power <= 1e-6)
    print(f"信道总数：{len(channels)}")
    print(f"经典信道数：{n_classic}")
    print(f"量子信道数：{n_quantum}")

    # 计算噪声
    freq_array, fwm_noise, raman_noise, total_noise = compute_noise_spectrum(fiber, channels)

    # 打印统计信息
    print("\n噪声统计信息:")
    print(f"FWM 噪声功率范围：{10*np.log10(np.max(fwm_noise)/1e-3):.2f} ~ "
          f"{10*np.log10(np.max(fwm_noise)/1e-3):.2f} dBm")
    print(f"拉曼噪声功率范围：{10*np.log10(np.max(raman_noise)/1e-3):.2f} ~ "
          f"{10*np.log10(np.max(raman_noise)/1e-3):.2f} dBm")
    print(f"总噪声功率范围：{10*np.log10(np.max(total_noise)/1e-3):.2f} ~ "
          f"{10*np.log10(np.max(total_noise)/1e-3):.2f} dBm")

    # 创建子目录
    case_dir = os.path.join("output", case_name.replace(" ", "_"))
    os.makedirs(case_dir, exist_ok=True)

    # 导出 CSV
    export_to_csv(freq_array, fwm_noise, raman_noise, total_noise, output_dir=case_dir)

    # 绘制图形
    plot_signal_distribution(channels, output_dir=case_dir)
    plot_fwm_spectrum(freq_array, fwm_noise, channels, output_dir=case_dir)
    plot_raman_spectrum(freq_array, raman_noise, channels, output_dir=case_dir)


def main():
    """
    主函数：运行所有测试用例
    """
    print("=" * 60)
    print("QKD 共纤传输噪声功率谱计算")
    print("=" * 60)

    # ========== 光纤配置 ==========
    fiber = Fiber(
        fiber_type=FiberType.SSMF,
        length=50e3,  # 50 km
        temperature=300.0  # 300 K
    )
    print(f"\n光纤：{fiber}")

    # 中心频率
    f_center = 193.4e12
    f_spacing = 50e9  # 50 GHz

    # ========== 测试用例 1: 1 路经典信号 ==========
    run_test_case(
        case_name="1_classic_channel",
        classic_channels=[f_center],
        fiber=fiber
    )

    # ========== 测试用例 2: 3 路经典信号 ==========
    run_test_case(
        case_name="3_classic_channels",
        classic_channels=[f_center - f_spacing, f_center, f_center + f_spacing],
        fiber=fiber
    )

    # ========== 测试用例 3: 5 路经典信号 ==========
    run_test_case(
        case_name="5_classic_channels",
        classic_channels=[
            f_center - 2*f_spacing,
            f_center - f_spacing,
            f_center,
            f_center + f_spacing,
            f_center + 2*f_spacing
        ],
        fiber=fiber
    )

    print("\n" + "=" * 60)
        # ========== 测试用例 4: 8 路经典信号 ==========
    run_test_case(
        case_name="8_classic_channels",
        classic_channels=[
            f_center - 4*f_spacing,
            f_center - 3*f_spacing,
            f_center - 2*f_spacing,
            f_center - f_spacing,
            f_center,
            f_center + f_spacing,
            f_center + 2*f_spacing,
            f_center + 3*f_spacing
        ],
        fiber=fiber
    )

    print("所有测试完成！")
    print("=" * 60)
    print("输出文件位置：output/[测试用例名称]/")
    print("  - signal_distribution.png  (信号频域分布图)")
    print("  - fwm_spectrum.png         (FWM 噪声谱图)")
    print("  - raman_spectrum.png       (拉曼噪声谱图)")
    print("  - noise_spectrum_*.csv     (CSV 数据文件)")


if __name__ == '__main__':
    main()
