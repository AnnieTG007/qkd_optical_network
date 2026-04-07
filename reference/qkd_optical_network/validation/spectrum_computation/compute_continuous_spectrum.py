"""
连续噪声功率谱与链路信号功率谱计算脚本

使用连续带宽噪声模型（向量化版）计算 FWM 和 SpRS 噪声功率谱，
同时精确计算链路信号功率谱（每个采样点 PSD × df）。

每个采样点的功率 = PSD [W/Hz] × df [Hz]，纵轴单位 W 或 dBm。

使用方法:
--------
python validation/spectrum_computation/compute_continuous_spectrum.py

输出（每个场景各一份，位于 output/{场景名}/）:
----
  link_signal_spectrum_W.png       - 信号功率谱 [W, 对数纵轴]
  link_signal_spectrum_dBm.png     - 信号功率谱 [dBm]
  noise_fwm_W.png                  - FWM 噪声功率谱 [W, 对数纵轴]
  noise_sprs_W.png                 - SpRS 噪声功率谱 [W, 对数纵轴]
  noise_fwm_sprs_W.png             - FWM+SpRS 总噪声 [W, 对数纵轴]
  noise_combined_W.png             - 信号+FWM+SpRS 叠加 [W, 对数纵轴]
  noise_fwm_dBm.png                - FWM 噪声功率谱 [dBm]
  noise_sprs_dBm.png               - SpRS 噪声功率谱 [dBm]
  noise_fwm_sprs_dBm.png           - FWM+SpRS 总噪声 [dBm]
  noise_combined_dBm.png           - 信号+FWM+SpRS 叠加 [dBm]
  continuous_spectrum_data.csv     - 完整数据表

场景:
----
  1_classic_channel    - 1 路经典 + 1 路量子
  3_classic_channels   - 3 路经典 + 1 路量子
  5_classic_channels   - 5 路经典 + 1 路量子
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from physics.fiber import Fiber
from physics.signal import WDMChannel, SpectrumType
from physics.noise import compute_noise
from constants.fiber_parameters import FiberType


# ─────────────────────────────────────────────────────────────────
#  单位转换
# ─────────────────────────────────────────────────────────────────

_FLOOR_W = 1e-40  # 防止 log(0) 的功率下限 [W]


def _to_dbm(power_w: np.ndarray) -> np.ndarray:
    """W → dBm，用 _FLOOR_W 截断防止 log(0)"""
    return 10.0 * np.log10(np.maximum(power_w, _FLOOR_W) / 1e-3)


# ─────────────────────────────────────────────────────────────────
#  信道构建
# ─────────────────────────────────────────────────────────────────

def build_channels(
    classic_freqs: List[float],
    f_spacing: float = 50e9,
    classic_power: float = 1e-3,
    quantum_power: float = 1e-11,
    baudrate: float = 32e9,
    roll_off: float = 0.1
) -> Tuple[List[WDMChannel], np.ndarray]:
    """
    构建 WDM 信道列表：经典信道（升余弦 PSD）+ 1 个量子信道

    量子信道置于经典信道组低频侧，间隔 1 个 f_spacing。

    Parameters
    ----------
    classic_freqs : List[float]
        经典信道中心频率列表 [Hz]
    f_spacing : float
        信道间隔 [Hz]
    classic_power : float
        经典信道功率 [W]
    quantum_power : float
        量子信道功率 [W]
    baudrate : float
        波特率 [Baud]
    roll_off : float
        升余弦滚降系数

    Returns
    -------
    channels : List[WDMChannel]
        信道列表（经典信道在前，量子信道在末尾）
    classic_mask : np.ndarray[bool]
        True = 经典信道，False = 量子信道
    """
    channels: List[WDMChannel] = []
    classic_set = set(classic_freqs)

    for freq in sorted(classic_freqs):
        channels.append(WDMChannel(
            center_freq=freq,
            power=classic_power,
            baudrate=baudrate,
            roll_off=roll_off,
            spectrum_type=SpectrumType.RAISED_COSINE
        ))

    # 量子信道：经典信道组低频侧相邻
    q_freq = min(classic_freqs) - f_spacing
    channels.append(WDMChannel(
        center_freq=q_freq,
        power=quantum_power,
        baudrate=baudrate,
        roll_off=roll_off,
        spectrum_type=SpectrumType.RAISED_COSINE
    ))

    classic_mask = np.array([ch.center_freq in classic_set for ch in channels])
    return channels, classic_mask


# ─────────────────────────────────────────────────────────────────
#  频谱计算
# ─────────────────────────────────────────────────────────────────

def compute_spectra(
    fiber: Fiber,
    channels: List[WDMChannel],
    compute_resolution: float = 1e9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    在统一频率网格上计算信号与噪声功率谱

    使用 compute_noise() 统一接口，返回连续模式频谱数据。

    信号：对每个频率采样点精确求所有信道的 PSD 之和，乘以 df
    噪声：调用连续带宽模型，返回每采样点的噪声功率，对信道维度聚合

    Parameters
    ----------
    fiber : Fiber
    channels : List[WDMChannel]
    compute_resolution : float
        频率网格分辨率 df [Hz]，默认 1 GHz

    Returns
    -------
    freq_hz : np.ndarray
        频率网格 [Hz]
    signal_bin : np.ndarray
        信号功率/bin [W]，signal_psd × df
    fwm_bin : np.ndarray
        FWM 噪声功率/bin [W]
    raman_bin : np.ndarray
        SpRS 噪声功率/bin [W]
    df : float
        实际频率分辨率 [Hz]
    """
    # ── 统一噪声计算接口 ────────────────────────────────────────────
    print("  计算噪声谱（连续带宽模型）...")
    result = compute_noise(
        fiber, channels,
        freq_resolution=compute_resolution,
        return_spectrum=True
    )

    # ── 频率网格 ──────────────────────────────────────────────────
    if result.freq_array is None:
        raise RuntimeError("连续模式未返回有效频率网格，请检查信道配置")

    freq_hz = result.freq_array
    df = result.df if result.df is not None else compute_resolution

    # ── FWM 噪声/bin（对信道维度求和） ────────────────────────────
    if result.fwm_spectrum is not None:
        fwm_total = np.sum(result.fwm_spectrum, axis=0)
        fwm_bin = fwm_total
    else:
        fwm_bin = np.zeros(len(freq_hz))
        print("  FWM 噪声为零（信道数 < 3 或 FWM 未启用）")

    # ── Raman 噪声/bin（对信道维度求和） ──────────────────────────
    if result.raman_spectrum is not None:
        raman_total = np.sum(result.raman_spectrum, axis=0)
        raman_bin = raman_total
    else:
        raman_bin = np.zeros(len(freq_hz))

    # ── 信号功率/bin：每采样点精确计算 PSD，乘以 df ───────────────
    signal_psd = np.zeros(len(freq_hz), dtype=np.float64)
    for ch in channels:
        signal_psd += ch.get_psd(freq_hz)
    signal_bin = signal_psd * df

    return freq_hz, signal_bin, fwm_bin, raman_bin, df


# ─────────────────────────────────────────────────────────────────
#  绘图辅助
# ─────────────────────────────────────────────────────────────────

def _add_vlines(ax, channels: List[WDMChannel], classic_mask: np.ndarray):
    """在坐标轴上标注信道中心频率（灰色虚线=经典，青色点线=量子）"""
    for i, ch in enumerate(channels):
        f_thz = ch.center_freq / 1e12
        if classic_mask[i]:
            ax.axvline(f_thz, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        else:
            ax.axvline(f_thz, color='cyan', linestyle=':', linewidth=0.8, alpha=0.6)


def _decorate(ax, ylabel: str, title: str, legend: bool = True):
    """统一装饰坐标轴"""
    ax.set_xlabel('Frequency [THz]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=8)


def _savefig(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────
#  1. 链路信号功率谱（2 张）
# ─────────────────────────────────────────────────────────────────

def plot_signal_spectra(
    freq_hz: np.ndarray,
    channels: List[WDMChannel],
    classic_mask: np.ndarray,
    output_dir: str,
    df: float
):
    """
    绘制链路信号功率谱

    经典/量子信道分色显示，纵轴分别为 W（对数）和 dBm。
    纵轴值 = PSD × df（每频率 bin 的功率）。

    输出：
      link_signal_spectrum_W.png
      link_signal_spectrum_dBm.png
    """
    freq_thz = freq_hz / 1e12

    # 按信道类型分别计算功率/bin
    classic_bin = np.zeros(len(freq_hz), dtype=np.float64)
    quantum_bin = np.zeros(len(freq_hz), dtype=np.float64)
    for i, ch in enumerate(channels):
        ch_bin = ch.get_psd(freq_hz) * df
        if classic_mask[i]:
            classic_bin += ch_bin
        else:
            quantum_bin += ch_bin

    ylabel_suffix = f'  (df = {df/1e9:.1f} GHz)'

    # ── 图1: W 对数纵轴 ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freq_thz, np.maximum(classic_bin, _FLOOR_W),
                color='red', linewidth=1.2, label='Classic channel')
    ax.semilogy(freq_thz, np.maximum(quantum_bin, _FLOOR_W),
                color='blue', linewidth=1.2, label='Quantum channel')
    _add_vlines(ax, channels, classic_mask)
    _decorate(ax,
              ylabel='Power [W]' + ylabel_suffix,
              title='Link Signal Power Spectrum [W, log scale]')
    _savefig(fig, os.path.join(output_dir, 'link_signal_spectrum_W.png'))

    # ── 图2: dBm ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq_thz, _to_dbm(classic_bin),
            color='red', linewidth=1.2, label='Classic channel')
    ax.plot(freq_thz, _to_dbm(quantum_bin),
            color='blue', linewidth=1.2, label='Quantum channel')
    _add_vlines(ax, channels, classic_mask)
    _decorate(ax,
              ylabel='Power [dBm]' + ylabel_suffix,
              title='Link Signal Power Spectrum [dBm]')
    _savefig(fig, os.path.join(output_dir, 'link_signal_spectrum_dBm.png'))


# ─────────────────────────────────────────────────────────────────
#  2. 连续噪声功率谱（8 张 = 2 组 × 4 张）
# ─────────────────────────────────────────────────────────────────

def plot_noise_spectra(
    freq_hz: np.ndarray,
    signal_bin: np.ndarray,
    fwm_bin: np.ndarray,
    raman_bin: np.ndarray,
    channels: List[WDMChannel],
    classic_mask: np.ndarray,
    output_dir: str,
    df: float
):
    """
    绘制 8 张连续噪声功率谱图（2 组 × 4 类型）

    组 W（对数纵轴）：
      noise_fwm_W.png / noise_sprs_W.png / noise_fwm_sprs_W.png / noise_combined_W.png
    组 dBm：
      noise_fwm_dBm.png / noise_sprs_dBm.png / noise_fwm_sprs_dBm.png / noise_combined_dBm.png
    """
    freq_thz = freq_hz / 1e12
    total_noise = fwm_bin + raman_bin
    ylabel_suffix = f'  (df = {df/1e9:.1f} GHz)'

    # 前 3 类单谱线图的配置
    single_specs = [
        ('noise_fwm',      fwm_bin,      'FWM Noise',         'green'),
        ('noise_sprs',     raman_bin,    'SpRS Noise',         'orange'),
        ('noise_fwm_sprs', total_noise,  'FWM + SpRS Noise',   'purple'),
    ]

    for fname, data, label, color in single_specs:

        # ── W 对数 ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(freq_thz, np.maximum(data, _FLOOR_W),
                    color=color, linewidth=1.2, label=label)
        _add_vlines(ax, channels, classic_mask)
        _decorate(ax,
                  ylabel='Power [W]' + ylabel_suffix,
                  title=f'{label} [W, log scale]')
        _savefig(fig, os.path.join(output_dir, f'{fname}_W.png'))

        # ── dBm ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freq_thz, _to_dbm(data),
                color=color, linewidth=1.2, label=label)
        _add_vlines(ax, channels, classic_mask)
        _decorate(ax,
                  ylabel='Power [dBm]' + ylabel_suffix,
                  title=f'{label} [dBm]')
        _savefig(fig, os.path.join(output_dir, f'{fname}_dBm.png'))

    # ── 叠加图（信号 + FWM + SpRS + 总噪声）× 2 ──────────────────
    combined_lines = [
        (signal_bin,  'Signal',         'red',    '-',  1.4),
        (fwm_bin,     'FWM Noise',      'green',  '--', 1.0),
        (raman_bin,   'SpRS Noise',     'orange', '--', 1.0),
        (total_noise, 'Total Noise',    'purple', '-',  1.2),
    ]

    # W 对数
    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label, color, ls, lw in combined_lines:
        ax.semilogy(freq_thz, np.maximum(data, _FLOOR_W),
                    color=color, linestyle=ls, linewidth=lw, label=label)
    _add_vlines(ax, channels, classic_mask)
    _decorate(ax,
              ylabel='Power [W]' + ylabel_suffix,
              title='Signal + Noise Power Spectrum [W, log scale]')
    _savefig(fig, os.path.join(output_dir, 'noise_combined_W.png'))

    # dBm
    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label, color, ls, lw in combined_lines:
        ax.plot(freq_thz, _to_dbm(data),
                color=color, linestyle=ls, linewidth=lw, label=label)
    _add_vlines(ax, channels, classic_mask)
    _decorate(ax,
              ylabel='Power [dBm]' + ylabel_suffix,
              title='Signal + Noise Power Spectrum [dBm]')
    _savefig(fig, os.path.join(output_dir, 'noise_combined_dBm.png'))


# ─────────────────────────────────────────────────────────────────
#  CSV 导出
# ─────────────────────────────────────────────────────────────────

def export_csv(
    freq_hz: np.ndarray,
    signal_bin: np.ndarray,
    fwm_bin: np.ndarray,
    raman_bin: np.ndarray,
    output_dir: str
):
    """
    导出完整频谱数据到 CSV

    列：freq_THz | signal_W | fwm_W | raman_W | total_noise_W |
        signal_dBm | fwm_dBm | raman_dBm | total_noise_dBm

    所有功率列均为每频率 bin 的功率（PSD × df）。
    """
    total_noise = fwm_bin + raman_bin
    data = np.column_stack([
        freq_hz / 1e12,
        signal_bin,
        fwm_bin,
        raman_bin,
        total_noise,
        _to_dbm(signal_bin),
        _to_dbm(fwm_bin),
        _to_dbm(raman_bin),
        _to_dbm(total_noise)
    ])
    header = (
        'freq_THz,signal_W,fwm_W,raman_W,total_noise_W,'
        'signal_dBm,fwm_dBm,raman_dBm,total_noise_dBm'
    )
    path = os.path.join(output_dir, 'continuous_spectrum_data.csv')
    np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.8e')
    print(f"    continuous_spectrum_data.csv")


# ─────────────────────────────────────────────────────────────────
#  场景运行入口
# ─────────────────────────────────────────────────────────────────

def run_scenario(
    scenario_name: str,
    classic_freqs: List[float],
    fiber: Fiber,
    f_spacing: float = 50e9,
    classic_power: float = 1e-3,
    quantum_power: float = 1e-11
):
    """
    运行单个场景，输出 10 张 PNG + 1 个 CSV

    Parameters
    ----------
    scenario_name : str
        场景名称（用作输出子目录名）
    classic_freqs : List[float]
        经典信道中心频率列表 [Hz]
    fiber : Fiber
    f_spacing : float
        信道间隔 [Hz]
    classic_power : float
        经典信道功率 [W]
    quantum_power : float
        量子信道功率 [W]
    """
    print(f"\n{'='*60}")
    print(f"场景：{scenario_name}")
    print(f"{'='*60}")

    channels, classic_mask = build_channels(
        classic_freqs, f_spacing, classic_power, quantum_power
    )
    n_classic = int(classic_mask.sum())
    n_quantum = len(channels) - n_classic
    print(f"信道数：{len(channels)}（经典 {n_classic}，量子 {n_quantum}）")
    for ch, is_cl in zip(channels, classic_mask):
        tag = 'Classic' if is_cl else 'Quantum'
        print(f"  {tag:8s}  {ch.center_freq/1e12:.3f} THz  "
              f"{10*np.log10(ch.power/1e-3):.1f} dBm  bw={ch.bandwidth/1e9:.1f} GHz")

    # 计算频谱
    freq_hz, signal_bin, fwm_bin, raman_bin, df = compute_spectra(fiber, channels)
    print(f"  频率网格：{len(freq_hz)} 点，df = {df/1e9:.2f} GHz")

    # 输出目录
    out_dir = os.path.join('output', scenario_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  输出目录：{out_dir}")

    # 绘图
    print("  生成链路信号功率谱（2 张）：")
    plot_signal_spectra(freq_hz, channels, classic_mask, out_dir, df)

    print("  生成连续噪声功率谱（8 张）：")
    plot_noise_spectra(freq_hz, signal_bin, fwm_bin, raman_bin,
                       channels, classic_mask, out_dir, df)

    # CSV
    print("  导出 CSV：")
    export_csv(freq_hz, signal_bin, fwm_bin, raman_bin, out_dir)

    print(f"  场景 {scenario_name} 完成，共 10 张 PNG + 1 个 CSV")


# ─────────────────────────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('连续噪声功率谱与链路信号功率谱计算')
    print('=' * 60)

    # 光纤配置（与 compute_noise_spectrum.py 保持一致）
    fiber = Fiber(
        fiber_type=FiberType.SSMF,
        length=50e3,       # 50 km
        temperature=300.0  # 300 K
    )
    print(f"\n光纤：{fiber}")

    f_center = 193.4e12   # 中心频率 [Hz]
    f_spacing = 50e9      # 信道间隔 [Hz]

    # ── 场景 1：1 路经典信号 ──────────────────────────────────────
    run_scenario(
        '1_classic_channel',
        classic_freqs=[f_center],
        fiber=fiber,
        f_spacing=f_spacing
    )

    # ── 场景 2：3 路经典信号 ──────────────────────────────────────
    run_scenario(
        '3_classic_channels',
        classic_freqs=[
            f_center - f_spacing,
            f_center,
            f_center + f_spacing
        ],
        fiber=fiber,
        f_spacing=f_spacing
    )

    # ── 场景 3：5 路经典信号 ──────────────────────────────────────
    run_scenario(
        '5_classic_channels',
        classic_freqs=[
            f_center - 2 * f_spacing,
            f_center - f_spacing,
            f_center,
            f_center + f_spacing,
            f_center + 2 * f_spacing
        ],
        fiber=fiber,
        f_spacing=f_spacing
    )

    print(f"\n{'='*60}")
    print("所有场景完成！")
    print("输出文件（每个场景 10 张 PNG + 1 个 CSV）：")
    for case in ['1_classic_channel', '3_classic_channels', '5_classic_channels']:
        print(f"  output/{case}/")
        print(f"    link_signal_spectrum_W.png / _dBm.png")
        print(f"    noise_fwm_W.png / _dBm.png")
        print(f"    noise_sprs_W.png / _dBm.png")
        print(f"    noise_fwm_sprs_W.png / _dBm.png")
        print(f"    noise_combined_W.png / _dBm.png")
        print(f"    continuous_spectrum_data.csv")
    print('=' * 60)


if __name__ == '__main__':
    main()
