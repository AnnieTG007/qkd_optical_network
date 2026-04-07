"""
信号功率谱对比脚本

对比三种信号建模方式的功率谱形状：
1. 单频模型 (SINGLE_FREQ): 所有功率集中在中心频率
2. 矩形谱 (RECTANGULAR): 在波特率带宽内均匀分布
3. 升余弦滚降谱 (RAISED_COSINE): 平坦区 + 余弦滚降过渡带

输出:
- output/spectrum_models/spectrum_comparison.png: 功率谱形状对比图
- output/spectrum_models/spectrum_comparison.csv: PSD 和功率数据

使用方法:
    python validation/model_comparison/compare_spectrum_models.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics.signal import WDMChannel, SpectrumType


def create_channels_with_different_spectra(
    center_freq: float = 193.4e12,
    power: float = 1e-3,
    baudrate: float = 32e9,
    roll_off: float = 0.1
):
    """
    创建使用不同谱形类型的信道

    Parameters
    ----------
    center_freq : float
        中心频率 [Hz]
    power : float
        功率 [W]
    baudrate : float
        波特率 [Baud]
    roll_off : float
        滚降系数

    Returns
    -------
    dict
        信道字典，键为谱形类型名称
    """
    channels = {}

    # 单频模型
    channels['single_freq'] = WDMChannel(
        center_freq=center_freq,
        power=power,
        baudrate=baudrate,
        roll_off=roll_off,
        spectrum_type=SpectrumType.SINGLE_FREQ
    )

    # 矩形谱（波特率带宽内均匀分布，不考虑滚降）
    channels['rectangular'] = WDMChannel(
        center_freq=center_freq,
        power=power,
        baudrate=baudrate,
        roll_off=roll_off,
        spectrum_type=SpectrumType.RECTANGULAR
    )

    # 升余弦滚降谱
    channels['raised_cosine'] = WDMChannel(
        center_freq=center_freq,
        power=power,
        baudrate=baudrate,
        roll_off=roll_off,
        spectrum_type=SpectrumType.RAISED_COSINE
    )

    return channels


def compute_psd_comparison(channels: dict, freq_array: np.ndarray):
    """
    计算各信道的 PSD

    Parameters
    ----------
    channels : dict
        信道字典
    freq_array : np.ndarray
        频率数组 [Hz]

    Returns
    -------
    dict
        PSD 字典，键与 channels 相同
    """
    psd_dict = {}
    for name, ch in channels.items():
        psd_dict[name] = ch.get_psd(freq_array)
    return psd_dict


def plot_spectrum_comparison(
    freq_array: np.ndarray,
    power_dict: dict,
    channels: dict,
    df: float,
    output_dir: str = "output/spectrum_models"
):
    """
    绘制功率谱对比图（功率谱 = PSD × df，模拟 OSA 测量结果）

    Parameters
    ----------
    freq_array : np.ndarray
        频率数组 [Hz]
    power_dict : dict
        功率谱字典（PSD × df）[W]
    channels : dict
        信道字典
    df : float
        频率分辨率（RBW）[Hz]
    output_dir : str
        输出目录
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 转换单位
    freq_thz = freq_array / 1e12

    # 创建图形 (1 行 2 列)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 颜色和标签
    colors = {
        'single_freq': 'red',
        'rectangular': 'blue',
        'raised_cosine': 'green'
    }
    labels = {
        'single_freq': 'Single-Freq (Delta)',
        'rectangular': f'Rectangular (B={channels["rectangular"].baudrate/1e9:.0f} GHz)',
        'raised_cosine': f'Raised-Cosine (roll-off={channels["raised_cosine"].roll_off})'
    }

    # 计算动态 y 轴范围
    all_powers = np.concatenate([power_dict[n] for n in power_dict.keys()])
    power_min = np.min(all_powers[all_powers > 0])
    power_max = np.max(all_powers)

    # ========== 子图 1: 功率谱 [W] - 对数坐标 ==========
    ax1 = axes[0]
    ax1.set_yscale('log')
    for name, power in power_dict.items():
        power_plot = np.maximum(power, 1e-30)  # 避免 log(0)
        ax1.plot(freq_thz, power_plot, color=colors[name], linewidth=1.5,
                label=labels[name], alpha=0.8)
    ax1.set_xlabel('Frequency [THz]')
    ax1.set_ylabel(f'Power Spectrum [W] (RBW = {df/1e6:.1f} MHz)')
    ax1.set_title('Power Spectrum (Linear - Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(power_min * 0.5, power_max * 2)

    # ========== 子图 2: 功率谱 [dBm] ==========
    ax2 = axes[1]
    for name, power in power_dict.items():
        power_dbm = 10 * np.log10(np.maximum(power, 1e-30) / 1e-3)
        ax2.plot(freq_thz, power_dbm, color=colors[name], linewidth=1.5,
                label=labels[name], alpha=0.8)
    ax2.set_xlabel('Frequency [THz]')
    ax2.set_ylabel(f'Power Spectrum [dBm] (RBW = {df/1e6:.1f} MHz)')
    ax2.set_title('Power Spectrum (dBm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # 动态调整 y 轴
    power_dbm_min = 10 * np.log10(power_min / 1e-3)
    power_dbm_max = 10 * np.log10(power_max / 1e-3)
    ax2.set_ylim(power_dbm_min - 5, power_dbm_max + 5)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, "spectrum_comparison.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Spectrum comparison plot saved: {plot_file}")

    plt.close()


def export_comparison_data(
    freq_array: np.ndarray,
    psd_dict: dict,
    power_dict: dict,
    channels: dict,
    output_dir: str = "output/spectrum_models"
):
    """
    导出对比数据到 CSV（同时包含 PSD 和功率值）

    Parameters
    ----------
    freq_array : np.ndarray
        频率数组
    psd_dict : dict
        PSD 字典
    power_dict : dict
        功率谱字典（PSD × df）
    channels : dict
        信道字典
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    freq_thz = freq_array / 1e12

    # 添加各模型的 PSD (dBm/Hz) 和功率 (dBm)
    data = np.column_stack([
        freq_thz,
        # PSD [dBm/Hz]
        10 * np.log10(np.maximum(psd_dict['single_freq'], 1e-30) / 1e-3),
        10 * np.log10(np.maximum(psd_dict['rectangular'], 1e-30) / 1e-3),
        10 * np.log10(np.maximum(psd_dict['raised_cosine'], 1e-30) / 1e-3),
        # Power [dBm] (PSD × df)
        10 * np.log10(np.maximum(power_dict['single_freq'], 1e-30) / 1e-3),
        10 * np.log10(np.maximum(power_dict['rectangular'], 1e-30) / 1e-3),
        10 * np.log10(np.maximum(power_dict['raised_cosine'], 1e-30) / 1e-3)
    ])

    header = 'Frequency[THz],SingleFreq_PSD[dBm/Hz],Rectangular_PSD[dBm/Hz],RaisedCosine_PSD[dBm/Hz],SingleFreq_Power[dBm],Rectangular_Power[dBm],RaisedCosine_Power[dBm]'

    output_file = os.path.join(output_dir, "spectrum_comparison.csv")
    np.savetxt(output_file, data, delimiter=',', header=header, fmt='%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f')
    print(f"CSV data saved: {output_file}")


def print_comparison_stats(channels: dict, psd_dict: dict, power_dict: dict, freq_array: np.ndarray):
    """
    打印对比统计信息

    Parameters
    ----------
    channels : dict
        信道字典
    psd_dict : dict
        PSD 字典
    power_dict : dict
        功率谱字典
    freq_array : np.ndarray
        频率数组
    """
    print("\n" + "=" * 70)
    print("Signal Spectrum Comparison Statistics")
    print("=" * 70)

    df = freq_array[1] - freq_array[0]

    for name, ch in channels.items():
        psd = psd_dict[name]
        power = power_dict[name]
        integrated_power = np.sum(psd) * df
        peak_psd = np.max(psd)
        peak_power = np.max(power)

        print(f"\n[{name.upper()}]")
        print(f"  Spectrum type: {ch.spectrum_type.value}")
        print(f"  Center frequency: {ch.center_freq/1e12:.4f} THz")
        print(f"  Total power (target): {ch.power:.2e} W ({ch.power_dbm:.2f} dBm)")
        print(f"  Bandwidth: {ch.bandwidth/1e9:.2f} GHz")

        if ch.bandwidth > 0:
            print(f"  PSD (in-band avg): {ch.power/ch.bandwidth:.4e} W/Hz")

        print(f"  PSD (peak): {peak_psd:.4e} W/Hz ({10*np.log10(peak_psd/1e-3):.2f} dBm/Hz)")
        print(f"  Peak power (RBW): {peak_power:.4e} W ({10*np.log10(peak_power/1e-3):.2f} dBm)")
        print(f"  Integrated power: {integrated_power:.4e} W")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("1. SINGLE_FREQ: Peak power depends on frequency resolution (RBW)")
    print("   - Higher resolution -> Higher peak power in RBW (delta function)")
    print("   - Integrated power always equals target power")
    print()
    print("2. RECTANGULAR: Uniform power distribution within baudrate bandwidth")
    print(f"   - Power per RBW = P × (df/baudrate) = {channels['rectangular'].power * df/channels['rectangular'].baudrate:.4e} W")
    print(f"   - Bandwidth = {channels['rectangular'].baudrate/1e9:.0f} GHz (no roll-off)")
    print()
    print("3. RAISED_COSINE: Same flat region, additional roll-off edges")
    print(f"   - Flat region: {channels['raised_cosine'].baudrate/1e9:.0f} GHz")
    print(f"   - Roll-off region: {channels['raised_cosine'].roll_off*channels['raised_cosine'].baudrate/1e9:.1f} GHz")
    print(f"   - Total bandwidth: {channels['raised_cosine'].bandwidth/1e9:.1f} GHz")
    print("-" * 70)


def main():
    """
    主函数：运行功率谱对比
    """
    print("=" * 70)
    print("Signal Spectrum Model Comparison")
    print("=" * 70)
    print("\nComparing three spectrum models:")
    print("1. Single-Frequency (Delta function)")
    print("2. Rectangular (Uniform within baudrate bandwidth)")
    print("3. Raised-Cosine (Flat + roll-off transition)")
    print()

    # 参数配置
    center_freq = 193.4e12  # 193.4 THz
    power = 1e-3            # 0 dBm
    baudrate = 32e9         # 32 GBaud
    roll_off = 0.5          # 50% roll-off

    print(f"Configuration:")
    print(f"  Center frequency: {center_freq/1e12:.2f} THz")
    print(f"  Power: {power:.2e} W ({10*np.log10(power/1e-3):.2f} dBm)")
    print(f"  Baudrate: {baudrate/1e9:.0f} GBaud")
    print(f"  Roll-off: {roll_off}")
    print()

    # 创建信道
    channels = create_channels_with_different_spectra(
        center_freq=center_freq,
        power=power,
        baudrate=baudrate,
        roll_off=roll_off
    )

    # 生成频率数组（覆盖所有信道带宽）
    # 目标粒度：10 MHz（模拟 OSA 分辨率）
    max_bandwidth = channels['raised_cosine'].bandwidth
    total_range = 4 * max_bandwidth  # ±2×带宽
    desired_resolution = 10e6  # 10 MHz
    n_points = int(total_range / desired_resolution)
    freq_array = np.linspace(
        center_freq - 2 * max_bandwidth,
        center_freq + 2 * max_bandwidth,
        n_points
    )
    df = freq_array[1] - freq_array[0]  # 实际频率粒度（用作 RBW）

    print(f"Frequency grid: {n_points} points, df = {df/1e6:.2f} MHz (RBW)")

    # 计算 PSD
    print("Computing PSD for each spectrum type...")
    psd_dict = compute_psd_comparison(channels, freq_array)

    # 计算功率谱（PSD × df），模拟 OSA 测量结果
    power_dict = {name: psd * df for name, psd in psd_dict.items()}

    # 打印统计信息
    print_comparison_stats(channels, psd_dict, power_dict, freq_array)

    # 绘制对比图（使用功率谱）
    print("\nGenerating comparison plots...")
    plot_spectrum_comparison(freq_array, power_dict, channels, df)

    # 导出数据（同时保留 PSD 和功率）
    export_comparison_data(freq_array, psd_dict, power_dict, channels)

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    print("Output files:")
    print("  - output/spectrum_models/spectrum_comparison.png (plot)")
    print("  - output/spectrum_models/spectrum_comparison.csv (data)")


if __name__ == '__main__':
    main()
