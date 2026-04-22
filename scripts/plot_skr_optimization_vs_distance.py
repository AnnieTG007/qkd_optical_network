"""绘制诱骗态 BB84 SKR 随光纤距离的变化曲线（含参数优化对比）。

用法：
    python scripts/plot_skr_optimization_vs_distance.py
    python scripts/plot_skr_optimization_vs_distance.py --noise-prob 1e-6
    python scripts/plot_skr_optimization_vs_distance.py --output skr_comparison.png

输出：
    Panel 1 — SKR vs 距离（semilogy）：无限长 / 近似有限长 / 严格有限长（固定） / 严格有限长（优化）
    Panel 2 — 最优参数随距离变化：mu_signal, mu_decoy, p_signal, P_X_alice
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FIBER_YAML = _ROOT / "src" / "qkd_sim" / "config" / "defaults" / "fiber_para" / "fiber_smf.yaml"
_DEFAULT_SKR_YAML   = _ROOT / "src" / "qkd_sim" / "config" / "defaults" / "skr_para" / "bb84_config.yaml"

DEFAULT_NOISE_PROB = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot BB84 SKR vs. fiber distance with parameter optimization")
    p.add_argument("--fiber-config", type=Path, default=_DEFAULT_FIBER_YAML,
                   help="FiberConfig YAML 路径")
    p.add_argument("--skr-config", type=Path, default=_DEFAULT_SKR_YAML,
                   help="SKRConfig YAML 路径")
    p.add_argument("--profile", choices=["custom", "reference"], default="custom",
                   help="SKR profile: 'custom' (默认，实际系统) 或 'reference' (文献参考值)")
    p.add_argument("--noise-prob", type=float, default=DEFAULT_NOISE_PROB,
                   help="噪声光子计数概率/脉冲（SpRS 等），默认 0")
    p.add_argument("--d-max-km", type=float, default=200.0,
                   help="最大距离 [km]，默认 200（优化较慢，不宜过大）")
    p.add_argument("--n-points", type=int, default=80,
                   help="距离采样点数，默认 80")
    p.add_argument("--output", type=Path, default=None,
                   help="保存图片路径（不指定则弹窗显示）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from dataclasses import replace

    from qkd_sim.config.schema import load_fiber_config, load_skr_config
    from qkd_sim.physical.skr.skr_decoy_bb84 import (
        infinite_key_rate,
        approx_finite_key_rate,
        strict_finite_key_rate,
    )
    from qkd_sim.physical.skr.skr_optimizer import DEFAULT_X0, SKROptimizer

    fiber_cfg = load_fiber_config(args.fiber_config)
    skr_cfg   = load_skr_config(args.skr_config, profile=args.profile)
    p_noise   = args.noise_prob

    # --- 距离采样 ---
    distances_km = np.linspace(0.1, args.d_max_km, args.n_points)
    distances_m  = distances_km * 1e3

    # --- 固定参数曲线 ---
    skr_inf    = np.zeros(len(distances_m))
    skr_approx = np.zeros(len(distances_m))
    skr_strict = np.zeros(len(distances_m))

    for i, d in enumerate(distances_m):
        skr_inf[i],    _, _ = infinite_key_rate(d, fiber_cfg, skr_cfg, p_noise)
        skr_approx[i], _, _ = approx_finite_key_rate(d, fiber_cfg, skr_cfg, p_noise)
        skr_strict[i], _, _ = strict_finite_key_rate(d, fiber_cfg, skr_cfg, p_noise)

    print("固定参数曲线计算完成。")

    # --- 优化参数曲线 ---
    optimizer = SKROptimizer(fiber_cfg, skr_cfg, p_noise)
    opt_results = optimizer.optimize_over_distances(list(distances_m), x0_initial=list(DEFAULT_X0))

    skr_opt = np.array([r.optimal_skr_bps for r in opt_results])
    qber_opt = np.array([r.qber for r in opt_results])

    # 提取最优参数
    p_mu_opt    = np.array([r.optimal_params["mu_signal"] for r in opt_results])
    p_nu_opt    = np.array([r.optimal_params["mu_decoy"] for r in opt_results])
    p_sig_opt   = np.array([r.optimal_params["p_signal"] for r in opt_results])
    p_xa_opt    = np.array([r.optimal_params["P_X_alice"] for r in opt_results])

    print("优化参数曲线计算完成。")

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                   gridspec_kw={"height_ratios": [1, 1]})

    # Panel 1: SKR vs distance
    ax1.semilogy(distances_km, np.where(skr_inf    > 0, skr_inf,    np.nan), label="无限长密钥")
    ax1.semilogy(distances_km, np.where(skr_approx > 0, skr_approx, np.nan), label="近似有限长")
    ax1.semilogy(distances_km, np.where(skr_strict  > 0, skr_strict,  np.nan),
                 label="严格有限长（固定参数）", linestyle="--")
    ax1.semilogy(distances_km, np.where(skr_opt    > 0, skr_opt,    np.nan),
                 label="严格有限长（优化参数）", linewidth=2)

    ax1.set_xlabel("光纤距离 [km]", fontsize=12)
    ax1.set_ylabel("安全码率 [bit/s]", fontsize=12)
    ax1.set_xlim(0, args.d_max_km)

    noise_str = f"p_noise = {p_noise:.2e}" if p_noise != 0.0 else "p_noise = 0（无额外噪声）"
    ax1.set_title(f"诱骗态 BB84 安全码率 vs. 距离\n{noise_str}", fontsize=13)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    # Panel 2: Optimal parameters vs distance
    colors_param = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax2.plot(distances_km, p_mu_opt,  "o-", color=colors_param[0], label=r"$\mu_{\mathrm{signal}}$", markersize=3)
    ax2.plot(distances_km, p_nu_opt,  "s-", color=colors_param[1], label=r"$\mu_{\mathrm{decoy}}$", markersize=3)
    ax2.plot(distances_km, p_sig_opt, "^-", color=colors_param[2], label=r"$p_{\mathrm{signal}}$", markersize=3)
    ax2.plot(distances_km, p_xa_opt,  "d-", color=colors_param[3], label=r"$P_{X,\mathrm{Alice}}$", markersize=3)

    ax2.set_xlabel("光纤距离 [km]", fontsize=12)
    ax2.set_ylabel("最优参数值", fontsize=12)
    ax2.set_xlim(0, args.d_max_km)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("优化参数随距离变化", fontsize=13)
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"图片已保存至：{args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
