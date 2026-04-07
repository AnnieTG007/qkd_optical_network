"""
噪声模型对比脚本

对比单频离散模型与连续带宽模型的 FWM 和拉曼噪声计算结果。

输出:
- output/noise_spectra/noise_spectra_W.png: W 单位谱图 (4×2 布局)
- output/noise_spectra/noise_spectra_dBm.png: dBm 单位谱图 (4×2 布局)
- output/noise_spectra/noise_model_statistics.csv: 5 个统计表数据（包含 PSD 和功率）

使用方法:
    python validation/model_comparison/compare_noise_models.py
"""

import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径（当前文件在 validation/model_comparison/下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics.fiber import Fiber
from physics.signal import WDMChannel, SpectrumType
from constants.fiber_parameters import FiberType


def build_test_channels(
    center_freq: float = 193.4e12,
    n_classic: int = 3,
    spacing: float = 50e9,
    classic_power: float = 1e-3,
    quantum_power: float = 1e-11,
    baudrate: float = 32e9,
    roll_off: float = 0.5,
    spectrum_type: SpectrumType = SpectrumType.RAISED_COSINE
) -> list:
    """
    构建测试信道（经典信道 + 量子信道）

    Parameters
    ----------
    center_freq : float
        中心频率 [Hz]
    n_classic : int
        经典信道数量
    spacing : float
        信道间隔 [Hz]
    classic_power : float
        经典信道功率 [W]
    quantum_power : float
        量子信道功率 [W]
    baudrate : float
        波特率 [Baud]
    roll_off : float
        滚降系数
    spectrum_type : SpectrumType
        频谱类型

    Returns
    -------
    channels : list
        WDMChannel 列表（按频率排序）
    """
    channels = []

    # 经典信道（围绕中心频率）
    for i in range(n_classic):
        offset = (i - n_classic // 2) * spacing
        freq = center_freq + offset
        channels.append(WDMChannel(
            center_freq=freq,
            power=classic_power,
            baudrate=baudrate,
            roll_off=roll_off,
            spectrum_type=spectrum_type,
            modulation='QPSK',
            direction='forward',
            channel_id=i
        ))

    # 量子信道（在边缘）
    channels.append(WDMChannel(
        center_freq=center_freq + (n_classic // 2 + 1) * spacing,
        power=quantum_power,
        baudrate=baudrate,
        roll_off=roll_off,
        spectrum_type=spectrum_type,
        modulation='QPSK',
        direction='forward',
        channel_id=n_classic
    ))

    # 按频率排序
    channels.sort(key=lambda ch: ch.center_freq)

    return channels


def run_comparison(
    fiber: Fiber,
    channels: list,
    freq_resolution: float = 100e6
):
    """
    运行噪声对比计算

    使用 compute_noise() 统一接口，分别获取离散和连续模型结果。

    Parameters
    ----------
    fiber : Fiber
        光纤对象
    channels : list
        信道列表（应使用连续类型，如 RAISED_COSINE）
    freq_resolution : float
        频率分辨率 [Hz]

    Returns
    -------
    dict
        包含离散和连续模型结果的字典，包括：
        - fwm_discrete, raman_discrete: 离散模型总噪声功率
        - freq_grid: 频率网格
        - fwm_spectrum, raman_spectrum: 各信道的噪声频谱 (n_channels, n_freq)
    """
    from physics.noise import compute_noise

    results = {}

    # ========== 离散模型（单频）==========
    # 构造单频信道列表（复用原始信道的频率和功率）
    print("\n[1/4] Computing noise (discrete model)...")
    channels_single = []
    for ch in channels:
        single_ch = WDMChannel(
            center_freq=ch.center_freq,
            power=ch.power,
            baudrate=ch.baudrate,
            roll_off=ch.roll_off,
            spectrum_type=SpectrumType.SINGLE_FREQ,
            modulation=ch.modulation,
            direction=ch.direction,
        )
        channels_single.append(single_ch)

    result_discrete = compute_noise(fiber, channels_single)
    results['fwm_discrete'] = result_discrete.fwm_power
    results['raman_discrete'] = result_discrete.raman_power

    # ========== 连续模型（使用原信道，已是连续类型）============
    print("[2/4] Computing noise (continuous model)...")
    result_continuous = compute_noise(
        fiber, channels,
        freq_resolution=freq_resolution,
        return_spectrum=True
    )

    results['freq_grid'] = result_continuous.freq_array
    results['df'] = result_continuous.df
    results['fwm_spectrum'] = result_continuous.fwm_spectrum
    results['raman_spectrum'] = result_continuous.raman_spectrum
    results['fwm_powers'] = result_continuous.fwm_power
    results['raman_powers'] = result_continuous.raman_power

    return results


def plot_spectra(
    channels: list,
    results: dict,
    output_dir: str = "output/noise_spectra"
):
    """
    绘制频谱图 - 2 张大图，每张 4 个子图（功率谱 = PSD × df）
    使用向量化连续模型计算的噪声频谱
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    os.makedirs(output_dir, exist_ok=True)
    freq_grid = results.get('freq_grid')

    if freq_grid is None or len(freq_grid) == 0:
        print("No frequency grid available, using fallback method")
        _plot_simple_spectra(channels, results, output_dir)
        return

    n_freq = len(freq_grid)
    df = results.get('df', 1e9)

    # ========== 从向量化连续模型获取噪声频谱 ==========
    fwm_spectrum = results.get('fwm_spectrum', None)  # (n_channels, n_coarse_freq)
    raman_spectrum = results.get('raman_spectrum', None)  # (n_channels, n_coarse_freq)

    if fwm_spectrum is not None and raman_spectrum is not None:
        # Check dimensions - could be 1D (powers) or 2D(spectrum)
        if np.ndim(fwm_spectrum) == 1:
            # Degraded case: fwm_spectrum is (n_channels,), raman_spectrum is also (n_channels,)
            print(f"  Note: Received 1D arrays, using fallback approximation")
            fwm_power = np.zeros(n_freq)
            raman_power = np.zeros(n_freq)
            for i, ch in enumerate(channels):
                mask = (freq_grid >= ch.center_freq - ch.bandwidth/2) & \
                       (freq_grid <= ch.center_freq + ch.bandwidth/2)
                bandwidth = np.sum(mask) * df
                if bandwidth > 0:
                    fwm_power[mask] += fwm_spectrum[i] / bandwidth * df
                    raman_power[mask] += raman_spectrum[i] / bandwidth * df
        else:
            # FWM + Raman noise spectrum (aggregate all channel contributions)
            # The coarse grid might not match the target freq_grid
            n_ch, n_pts = fwm_spectrum.shape

            # Interpolate coarse spectrum to target resolution
            freq_coarse = np.linspace(freq_grid[0], freq_grid[-1], n_pts)

            fwm_power = np.interp(freq_grid, freq_coarse, np.sum(fwm_spectrum, axis=0))
            raman_power = np.interp(freq_grid, freq_coarse, np.sum(raman_spectrum, axis=0))
            print(f"  Using vectorized spectrum: {n_ch} channels, {n_pts} coarse points -> interpolated to {n_freq}")
    else:
        # Fallback: uniform distribution approximation
        print("  No spectrum data, using uniform distribution approximation")
        fwm_power = np.zeros(n_freq)
        raman_power = np.zeros(n_freq)

        for i, ch in enumerate(channels):
            mask = (freq_grid >= ch.center_freq - ch.bandwidth/2) & \
                   (freq_grid <= ch.center_freq + ch.bandwidth/2)
            bandwidth = np.sum(mask) * df

            if bandwidth > 0:
                fwm_power[mask] += results['fwm_discrete'][i] / bandwidth * df
                raman_power[mask] += results['raman_discrete'][i] / bandwidth * df

    # ========== 信号 PSD ==========
    signal_psd = np.zeros_like(freq_grid)
    for ch in channels:
        signal_psd += ch.get_psd(freq_grid, power=ch.power)

    total_noise_power = fwm_power + raman_power
    total_power = signal_psd * df + total_noise_power

    # 转换为 dBm
    def to_dbm(power):
        return 10 * np.log10(np.maximum(power, 1e-30) / 1e-3)

    subplot_titles = [
        'FWM Noise Power',
        'SpRS (Raman) Noise Power',
        'Total Noise (FWM+SpRS)',
        'Total Power (Signal+Noise)'
    ]

    # W 图（功率谱）
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, title in zip(axes.flat, subplot_titles):
        if 'FWM' in title:
            data = fwm_power
        elif 'SpRS' in title:
            data = raman_power
        elif 'Total Noise' in title:
            data = total_noise_power
        else:
            data = total_power

        ax.plot(freq_grid / 1e12, np.maximum(data, 1e-30), 'b-', linewidth=1.2)
        ax.set_xlabel('Frequency [THz]', fontsize=11)
        ax.set_ylabel(f'Power Spectrum [W] (RBW = {df/1e6:.1f} MHz)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.semilogy()
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_spectra_W.png"), dpi=150, bbox_inches='tight')
    print(f"\nW spectra saved: {output_dir}/noise_spectra_W.png")
    plt.close()

    # dBm 图（功率谱）
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, title in zip(axes.flat, subplot_titles):
        if 'FWM' in title:
            data = to_dbm(fwm_power)
        elif 'SpRS' in title:
            data = to_dbm(raman_power)
        elif 'Total Noise' in title:
            data = to_dbm(total_noise_power)
        else:
            data = to_dbm(total_power)

        ax.plot(freq_grid / 1e12, data, 'b-', linewidth=1.2)
        ax.set_xlabel('Frequency [THz]', fontsize=11)
        ax.set_ylabel(f'Power Spectrum [dBm] (RBW = {df/1e6:.1f} MHz)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_spectra_dBm.png"), dpi=150, bbox_inches='tight')
    print(f"dBm spectra saved: {output_dir}/noise_spectra_dBm.png")
    plt.close()


def _plot_simple_spectra(channels, results, output_dir):
    """简单的回退绘图方法"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 生成频率网格
    freq_min = min(ch.center_freq - ch.bandwidth for ch in channels)
    freq_max = max(ch.center_freq + ch.bandwidth for ch in channels)
    freq_grid = np.linspace(freq_min, freq_max, 10000)
    df = freq_grid[1] - freq_grid[0]

    # 信号功率谱
    signal_power = np.zeros_like(freq_grid)
    for ch in channels:
        signal_power += ch.get_psd(freq_grid, power=ch.power) * df

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, title in zip(axes.flat, ['FWM', 'SpRS', 'Total Noise', 'Total Power']):
        if title == 'FWM':
            noise_power = results.get('fwm_discrete', np.zeros(len(channels)))
        elif title == 'SpRS':
            noise_power = results.get('raman_discrete', np.zeros(len(channels)))
        else:
            continue
        # 简单分布噪声功率
        noise_power_spectrum = np.zeros_like(freq_grid)
        for i, ch in enumerate(channels):
            mask = (freq_grid >= ch.center_freq - ch.bandwidth/2) & \
                   (freq_grid <= ch.center_freq + ch.bandwidth/2)
            bandwidth = np.sum(mask) * df
            if bandwidth > 0:
                noise_power_spectrum[mask] += noise_power[i] / bandwidth * df

        if title == 'Total Noise':
            noise_power_spectrum = results.get('fwm_discrete', np.zeros(len(channels))) + results.get('raman_discrete', np.zeros(len(channels)))
            # 简化处理
        elif title == 'Total Power':
            pass  # handled separately

        ax.semilogy(freq_grid / 1e12, np.maximum(signal_power if title == 'Total Power' else noise_power_spectrum, 1e-30), 'b-')
        ax.set_xlabel('Freq [THz]')
        ax.set_ylabel(f'{title} [W] (RBW={df/1e6:.1f} MHz)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_spectra_W.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'noise_spectra_dBm.png'), dpi=150)
    plt.close()


def print_statistics(channels: list, results: dict):
    """
    打印统计信息（5 个表格）

    Parameters
    ----------
    channels : list
        信道列表
    results : dict
        结果字典
    """
    print("\n" + "=" * 70)
    print("Noise Model Comparison Statistics")
    print("=" * 70)

    n_channels = len(channels)

    print(f"\nConfiguration:")
    print(f"  Number of channels: {n_channels}")
    print(f"  Baudrate: {channels[0].baudrate/1e9:.0f} GBaud")
    print(f"  Roll-off: {channels[0].roll_off}")
    print(f"  Bandwidth: {channels[0].bandwidth/1e9:.1f} GHz")

    # ========== 表格 1: FWM 噪声 ==========
    print("\n" + "-" * 70)
    print("Table 1: FWM Noise Power (Discrete Models)")
    print("-" * 70)
    print(f"  {'Channel':<12} {'Power[dBm]':<12} {'FWM[dBm]':<12} {'Diff[dB]'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for i, ch in enumerate(channels):
        disc = 10 * np.log10(max(results['fwm_discrete'][i], 1e-30) / 1e-3)
        power_dbm = ch.power_dbm
        diff = disc - power_dbm
        print(f"  [{i}] {ch.center_freq/1e12:.3f} THz   {power_dbm:>8.2f}      {disc:>8.2f}     {diff:>6.2f}")

    # ========== 表格 2: SpRS 噪声 ==========
    print("\n" + "-" * 70)
    print("Table 2: SpRS (Raman) Noise Power (Discrete Models)")
    print("-" * 70)
    print(f"  {'Channel':<12} {'Power[dBm]':<12} {'SpRS[dBm]':<12} {'Diff[dB]'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for i, ch in enumerate(channels):
        disc = 10 * np.log10(max(results['raman_discrete'][i], 1e-30) / 1e-3)
        power_dbm = ch.power_dbm
        diff = disc - power_dbm
        print(f"  [{i}] {ch.center_freq/1e12:.3f} THz   {power_dbm:>8.2f}      {disc:>8.2f}     {diff:>6.2f}")

    # ========== 表格 3: 总噪声 ==========
    print("\n" + "-" * 70)
    print("Table 3: Total Noise (FWM+SpRS)")
    print("-" * 70)
    print(f"  {'Channel':<12} {'FWM[dBm]':<12} {'SpRS[dBm]':<12} {'Total[dBm]'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for i, ch in enumerate(channels):
        fwm_dbm = 10 * np.log10(max(results['fwm_discrete'][i], 1e-30) / 1e-3)
        raman_dbm = 10 * np.log10(max(results['raman_discrete'][i], 1e-30) / 1e-3)
        total_lin = results['fwm_discrete'][i] + results['raman_discrete'][i]
        total_dbm = 10 * np.log10(max(total_lin, 1e-30) / 1e-3)
        print(f"  [{i}] {fwm_dbm:>8.2f}      {raman_dbm:>8.2f}      {total_dbm:>8.2f}")

    # ========== 表格 4: 信号功率 ==========
    print("\n" + "-" * 70)
    print("Table 4: Signal Power by Channel")
    print("-" * 70)
    print(f"  {'Channel ID':<12} {'Frequency[THz]':<16} {'Power[W]':<14} {'Power[dBm]'}")
    print(f"  {'-'*12} {'-'*16} {'-'*14} {'-'*10}")

    for i, ch in enumerate(channels):
        print(f"  [{ch.channel_id}]      {ch.center_freq/1e12:.3f}       {ch.power:.4e}     {ch.power_dbm:>6.2f}")

    # ========== 表格 5: 总功率谱（向量化连续模型） ==========
    print("\n" + "-" * 70)
    print("Table 5: Total Power Spectrum (Vectorized Continuous Model)")
    print("-" * 70)
    print(f"  {'Spectrum Type':<30} {'Max [W]':<14} {'Max [dBm]'}")
    print(f"  {'-'*30} {'-'*14} {'-'*12}")

    if results.get('freq_grid') is not None and len(results['freq_grid']) > 0:
        freq_grid = results['freq_grid']
        df = results.get('df', freq_grid[1] - freq_grid[0]) if len(freq_grid) > 1 else 1e9

        signal_power = np.zeros_like(freq_grid)
        for ch in channels:
            signal_power += ch.get_psd(freq_grid, power=ch.power) * df

        # 从向量化连续模型获取噪声频谱
        fwm_spectrum = results.get('fwm_spectrum', None)
        raman_spectrum = results.get('raman_spectrum', None)

        if fwm_spectrum is not None and raman_spectrum is not None:
            # 使用真实频谱数据
            total_fwm = np.sum(fwm_spectrum, axis=0)
            total_raman = np.sum(raman_spectrum, axis=0)
        else:
            # Fallback: 均匀分布近似
            print("  Warning: No spectrum data available, using approximation")
            total_fwm = np.zeros(len(freq_grid))
            total_raman = np.zeros(len(freq_grid))
            for i, ch in enumerate(channels):
                mask = (freq_grid >= ch.center_freq - ch.bandwidth/2) & \
                       (freq_grid <= ch.center_freq + ch.bandwidth/2)
                bandwidth = np.sum(mask) * df
                if bandwidth > 0:
                    total_fwm[mask] += results['fwm_discrete'][i] / bandwidth * df
                    total_raman[mask] += results['raman_discrete'][i] / bandwidth * df

        total_noise_power = total_fwm + total_raman
        total_power = signal_power + total_noise_power

        print(f"  {'Signal Power':<30} {np.max(signal_power):>14.4e} {10*np.log10(max(np.max(signal_power),1e-30)/1e-3):>12.2f}")
        print(f"  {'FWM Noise (continuous)':<30} {np.max(total_fwm):>14.4e} {10*np.log10(max(np.max(total_fwm),1e-30)/1e-3):>12.2f}")
        print(f"  {'Raman Noise (continuous)':<30} {np.max(total_raman):>14.4e} {10*np.log10(max(np.max(total_raman),1e-30)/1e-3):>12.2f}")
        print(f"  {'Total Noise':<30} {np.max(total_noise_power):>14.4e} {10*np.log10(max(np.max(total_noise_power),1e-30)/1e-3):>12.2f}")
        print(f"  {'Total Power':<30} {np.max(total_power):>14.4e} {10*np.log10(max(np.max(total_power),1e-30)/1e-3):>12.2f}")

    print("\n" + "=" * 70)


def export_data(channels: list, results: dict, output_dir: str = "output/noise_spectra"):
    """
    导出统计数据

    Parameters
    ----------
    channels : list
        信道列表
    results : dict
        结果字典
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 导出数据表格
    data_rows = []
    header_lines = []

    # Table 1: FWM 噪声
    header_lines.append("# Table 1: FWM Noise")
    header_lines.append("# Channel_ID, Frequency_THz, Signal_Power_W, FWM_Noise_W, FWM_Noise_dBm, Diff_FWM_Signal_dB")
    for i, ch in enumerate(channels):
        fwm_dbm = 10 * np.log10(max(results['fwm_discrete'][i], 1e-30) / 1e-3)
        diff = fwm_dbm - ch.power_dbm
        data_rows.append([
            ch.channel_id,
            ch.center_freq / 1e12,
            ch.power,
            results['fwm_discrete'][i],
            fwm_dbm,
            diff
        ])

    # Table 2: SpRS 噪声
    header_lines.append("\n# Table 2: SpRS Noise")
    header_lines.append("# Channel_ID, Frequency_THz, Signal_Power_W, SpRS_Noise_W, SpRS_Noise_dBm, Diff_Raman_Signal_dB")
    for i, ch in enumerate(channels):
        raman_dbm = 10 * np.log10(max(results['raman_discrete'][i], 1e-30) / 1e-3)
        diff = raman_dbm - ch.power_dbm
        data_rows.extend([[
            ch.channel_id,
            ch.center_freq / 1e12,
            ch.power,
            results['raman_discrete'][i],
            raman_dbm,
            diff
        ]])

    output_file = os.path.join(output_dir, "noise_model_statistics.csv")
    with open(output_file, 'w') as f:
        f.write('\n'.join(header_lines) + '\n\n')
        for row in data_rows:
            f.write(','.join(f'{v}' for v in row) + '\n')

    print(f"Data exported: {output_file}")


def main():
    """
    主函数
    """
    print("=" * 70)
    print("Noise Model Comparison: Discrete vs Continuous")
    print("=" * 70)

    # 光纤配置
    fiber = Fiber(
        fiber_type=FiberType.SSMF,
        length=50e3,
        temperature=300.0
    )
    print(f"\nFiber: {fiber}")

    # 信道配置
    print("\nBuilding test channels...")
    channels = build_test_channels(
        center_freq=193.4e12,
        n_classic=8,
        spacing=50e9,
        classic_power=1e-3,
        baudrate=32e9,
        roll_off=0.5,
        spectrum_type=SpectrumType.RAISED_COSINE
    )

    print(f"  Number of channels: {len(channels)}")
    for ch in channels:
        print(f"  [{ch.channel_id}] {ch}")

    # 运行对比
    print("\nRunning noise calculations...")
    results = run_comparison(fiber, channels, freq_resolution=100e6)  # 100 MHz

    # 打印统计
    print_statistics(channels, results)

    # 绘图
    print("\nGenerating spectral plots...")
    plot_spectra(channels, results)

    # 导出数据
    export_data(channels, results)

    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)
    print("Output files:")
    print("  - output/noise_spectra/noise_spectra_W.png (W)")
    print("  - output/noise_spectra/noise_spectra_dBm.png (dBm)")
    print("  - output/noise_spectra/noise_model_statistics.csv")


if __name__ == '__main__':
    main()
