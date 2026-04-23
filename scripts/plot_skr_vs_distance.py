"""Plot decoy-state BB84 SKR vs. fiber distance.

Usage:
    python scripts/plot_skr_vs_distance.py
    python scripts/plot_skr_vs_distance.py --noise-prob 1e-6
    python scripts/plot_skr_vs_distance.py --fiber-config path/to/fiber.yaml --skr-config path/to/bb84.yaml

Output: Three curves (infinite-key / approx finite-key / strict finite-key),
        X-axis: distance [km], Y-axis: SKR [bit/s].
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# Default config paths (relative to project root)
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FIBER_YAML = _ROOT / "src" / "qkd_sim" / "config" / "defaults" / "fiber_para" / "fiber_smf.yaml"
_DEFAULT_SKR_YAML   = _ROOT / "src" / "qkd_sim" / "config" / "defaults" / "skr_para" / "bb84_config.yaml"

# Noise photon count probability per pulse (overridable via CLI or computed from SpRS model)
DEFAULT_NOISE_PROB = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot BB84 SKR vs. fiber distance")
    p.add_argument("--fiber-config", type=Path, default=_DEFAULT_FIBER_YAML,
                   help="FiberConfig YAML path")
    p.add_argument("--skr-config", type=Path, default=_DEFAULT_SKR_YAML,
                   help="SKRConfig YAML path")
    p.add_argument("--profile", choices=["custom", "reference"], default="custom",
                   help="SKR profile within bb84_config.yaml: 'custom' (default, 实际系统) 或 'reference' (文献参考值)")
    p.add_argument("--noise-prob", type=float, default=DEFAULT_NOISE_PROB,
                   help="Noise photon count probability per pulse (SpRS, etc.), default 0")
    p.add_argument("--d-max-km", type=float, default=350.0,
                   help="Maximum distance [km], default 350")
    p.add_argument("--n-points", type=int, default=200,
                   help="Number of distance sampling points, default 200")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip Nelder-Mead parameter optimization (fast mode)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output image path (shows interactive window if not specified)")
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
    skr_cfg   = load_skr_config(args.skr_config, profile=args.profile)
    p_noise   = args.noise_prob

    # --- 距离采样 ---
    distances_km = np.linspace(0.1, args.d_max_km, args.n_points)
    distances_m  = distances_km * 1e3

    # --- 计算三种 SKR ---
    skr_inf    = np.zeros(len(distances_m))
    skr_approx = np.zeros(len(distances_m))
    skr_strict = np.zeros(len(distances_m))

    for i, d in enumerate(distances_m):
        if i % 20 == 0:
            print(f"[{i}/{len(distances_m)}] Computing distance {distances_km[i]:.1f} km...")
        skr_inf[i],    _, _ = infinite_key_rate(d, fiber_cfg, skr_cfg, p_noise)
        skr_approx[i], _, _ = approx_finite_key_rate(d, fiber_cfg, skr_cfg, p_noise)
        skr_strict[i], _, _ = strict_finite_key_rate(d, fiber_cfg, skr_cfg, p_noise,
                                                      optimize_params=not args.no_optimize)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(distances_km, np.where(skr_inf    > 0, skr_inf,    np.nan), label="Infinite-key")
    ax.semilogy(distances_km, np.where(skr_approx > 0, skr_approx, np.nan), label="Approx. finite-key")
    ax.semilogy(distances_km, np.where(skr_strict  > 0, skr_strict,  np.nan), label="Strict finite-key",
                linestyle="--")

    ax.set_xlabel("Fiber distance [km]", fontsize=12)
    ax.set_ylabel("Secure key rate [bit/s]", fontsize=12)
    ax.set_xlim(0, args.d_max_km)

    # Title includes noise parameter if non-zero
    noise_str = f"p_noise = {p_noise:.2e}" if p_noise != 0.0 else "p_noise = 0 (no extra noise)"
    ax.set_title(f"Decoy-state BB84 Secure Key Rate vs. Distance\n{noise_str}", fontsize=13)

    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"Figure saved to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()