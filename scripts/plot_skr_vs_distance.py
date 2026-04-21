"""绘制诱骗态 BB84 SKR 随光纤距离的变化曲线。

用法：
    python scripts/plot_skr_vs_distance.py
    python scripts/plot_skr_vs_distance.py --noise-prob 1e-6
    python scripts/plot_skr_vs_distance.py --fiber-config path/to/fiber.yaml --skr-config path/to/bb84.yaml

输出：三条曲线（无限长 / 近似有限长 / 严格有限长），X 轴为距离 [km]，Y 轴为 SKR [bit/s]。
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 默认配置文件路径（相对于项目根目录）
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FIBER_YAML = _ROOT / "src" / "qkd_sim" / "config" / "defaults" / "fiber_para" / "fiber_smf.yaml"
_DEFAULT_SKR_YAML   = _ROOT / "src" / "qkd_sim" / "config" / "defaults" / "skr_para" / "bb84_default.yaml"

# 噪声光子计数概率（可通过命令行覆盖，或由 SpRS 噪声模型计算后传入）
DEFAULT_NOISE_PROB = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot BB84 SKR vs. fiber distance")
    p.add_argument("--fiber-config", type=Path, default=_DEFAULT_FIBER_YAML,
                   help="FiberConfig YAML 路径")
    p.add_argument("--skr-config", type=Path, default=_DEFAULT_SKR_YAML,
                   help="SKRConfig YAML 路径")
    p.add_argument("--noise-prob", type=float, default=DEFAULT_NOISE_PROB,
                   help="噪声光子计数概率/脉冲（SpRS 等），默认 0")
    p.add_argument("--d-max-km", type=float, default=350.0,
                   help="最大距离 [km]，默认 350")
    p.add_argument("--n-points", type=int, default=200,
                   help="距离采样点数，默认 200")
    p.add_argument("--output", type=Path, default=None,
                   help="保存图片路径（不指定则弹窗显示）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- 加载配置 ---
    from qkd_sim.config.schema import load_fiber_config, load_skr_config
    from qkd_sim.physical.skr.skr_decoy_bb84 import (
        infinite_key_rate,
        approx_finite_key_rate,
        strict_finite_key_rate,
    )

    fiber_cfg = load_fiber_config(args.fiber_config)
    skr_cfg   = load_skr_config(args.skr_config)
    p_noise   = args.noise_prob

    # --- 距离采样 ---
    distances_km = np.linspace(0.1, args.d_max_km, args.n_points)
    distances_m  = distances_km * 1e3

    # --- 计算三种 SKR ---
    skr_inf    = np.zeros(len(distances_m))
    skr_approx = np.zeros(len(distances_m))
    skr_strict = np.zeros(len(distances_m))

    for i, d in enumerate(distances_m):
        skr_inf[i],    _, _ = infinite_key_rate(d, fiber_cfg, skr_cfg, p_noise)
        skr_approx[i], _, _ = approx_finite_key_rate(d, fiber_cfg, skr_cfg, p_noise)
        skr_strict[i], _, _ = strict_finite_key_rate(d, fiber_cfg, skr_cfg, p_noise)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(distances_km, np.where(skr_inf    > 0, skr_inf,    np.nan), label="无限长密钥")
    ax.semilogy(distances_km, np.where(skr_approx > 0, skr_approx, np.nan), label="近似有限长")
    ax.semilogy(distances_km, np.where(skr_strict  > 0, skr_strict,  np.nan), label="严格有限长",
                linestyle="--")

    ax.set_xlabel("光纤距离 [km]", fontsize=12)
    ax.set_ylabel("安全码率 [bit/s]", fontsize=12)
    ax.set_xlim(0, args.d_max_km)

    # 标题显示噪声参数
    noise_str = f"p_noise = {p_noise:.2e}" if p_noise != 0.0 else "p_noise = 0（无额外噪声）"
    ax.set_title(f"诱骗态 BB84 安全码率 vs. 距离\n{noise_str}", fontsize=13)

    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"图片已保存至：{args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()