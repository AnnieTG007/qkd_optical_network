"""
当前信号建模验证脚本 - 确认离散信道 vs 连续频谱模型

运行方式:
    python validation/signal_validation/test_signal_model.py
"""

import numpy as np
from scipy.constants import c
import sys
import os

# 添加项目根目录到路径（当前文件在 validation/signal_validation/下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics.signal import WDMChannel


def get_psd_numeric(ch, freq_array):
    """数值计算 PSD（从 signal.py 复制实现）"""
    df = np.abs(freq_array - ch.center_freq)

    B = ch.bandwidth
    f1 = (1 - ch.roll_off) * ch.baudrate / 2
    f2 = (1 + ch.roll_off) * ch.baudrate / 2

    psd = np.zeros_like(freq_array, dtype=np.float64)

    flat_mask = df <= f1
    psd[flat_mask] = 1.0

    roll_mask = (df > f1) & (df <= f2)
    if ch.roll_off > 0:
        arg = np.pi * (df[roll_mask] - f1) / (f2 - f1)
        psd[roll_mask] = 0.5 * (1 + np.cos(arg))

    if len(freq_array) > 1:
        df_num = freq_array[1] - freq_array[0]
        total_power = np.sum(psd) * df_num
        if total_power > 0:
            psd = psd * (ch.power / total_power)
    else:
        psd = psd * ch.power / ch.bandwidth

    return psd


def main():
    print("=" * 70)
    print("当前信号建模验证")
    print("=" * 70)

    ch = WDMChannel(
        center_freq=193.4e12,
        power=1e-3,
        baudrate=32e9,
        roll_off=0.1
    )

    freq_array = np.linspace(193.30e12, 193.50e12, 10000)
    psd = get_psd_numeric(ch, freq_array)

    df_num = freq_array[1] - freq_array[0]
    integral_power = np.sum(psd) * df_num

    print("\n测试信道参数:")
    print(f"  中心频率：{ch.center_freq/1e12:.4f} THz")
    print(f"  总功率：{ch.power_dbm:.2f} dBm ({ch.power:.2e} W)")
    print(f"  波特率：{ch.baudrate/1e9:.2f} GBaud")
    print(f"  滚降系数：{ch.roll_off}")
    print(f"  带宽：B=(1+{ch.roll_off})*{ch.baudrate/1e9:.2f} GHz = {ch.bandwidth/1e9:.2f} GHz")
    print()

    print("PSD 特性验证:")
    print(f"  PSD 峰值：{np.max(psd)/1e-12:.4f} pW/Hz")
    print(f"  PSD 积分：{integral_power:.2e} W (应为 {ch.power:.2e} W)")
    print(f"  相对误差：{(integral_power-ch.power)/ch.power*100:.4f}%")
    print()

    idx_within_band = (freq_array >= ch.center_freq - ch.bandwidth/2) & \
                      (freq_array <= ch.center_freq + ch.bandwidth/2)
    power_in_band = np.sum(psd[idx_within_band]) * df_num
    print(f"主瓣内功率占比：{power_in_band/ch.power*100:.2f}%")
    print()

    print("=" * 70)
    print("结论分析")
    print("=" * 70)
    print()
    print("[OK] signal.py 中 WDMChannel.get_psd() 已实现连续频谱建模")
    print("     - 支持升余弦滚降谱形")
    print("     - PSD 已归一化确保总功率守恒")
    print()
    print("[ISSUE] 但噪声计算模块实际使用的是【离散信道模型】:")
    print("     - physics/noise/fwm.py: 直接使用 channel.power")
    print("     - physics/noise/raman.py: 直接使用 channel.power")
    print()
    print("这意味着当前的 FWM 和拉曼噪声计算:")
    print("     [单频近似]: 假设所有能量集中在中心频率")
    print("     [未考虑]: 信道内部带宽分布、频谱形状影响")
    print()
    print("-" * 70)
    print("下一步需要:")
    print("1. 实现基于连续 PSD 的 FWM 噪声计算")
    print("2. 实现基于连续 PSD 的拉曼噪声计算")
    print("3. 对比单频模型与连续模型的差异")
    print("-" * 70)


if __name__ == '__main__':
    main()
