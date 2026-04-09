"""信号发射功率谱对比绘图脚本。

比较以下信号模型的发射功率（离散 vs 连续）：
  - Discrete: delta 近似（stem）
  - Raised Cosine β=0 (≡矩形)
  - Raised Cosine β=0.01
  - Raised Cosine β=0.1
  - Raised Cosine β=0.5
  - OSA

输出到: outputs/phase4_N80_C3/Signal_TX/

物理说明（见 spectrum.make_signal_psd_comparison_figure 图注）：
  离散模型的 stem 高度 = 信道功率 P [W]；
  连续模型的曲线高度 = PSD × Δf [W]（bin 功率）；量纲统一为 [W]。
"""

from pathlib import Path

import numpy as np

# ---- 项目路径 setup ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))   # 让 scripts/ 可作为包导入
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.schema import WDMConfig
from qkd_sim.physical.signal import build_wdm_grid, SpectrumType
from qkd_sim.physical.spectrum import (
    make_signal_psd_comparison_figure,
    SignalPSDResult,
    get_model_color,
)

# ============================================================================
# 配置参数（与 plot_noise_spectrum.py 保持一致）
# ============================================================================

WDM_PARAMS = dict(
    f_center=193.4e12,
    N_ch=80,
    channel_spacing=50e9,
    B_s=32e9,
    P0=1e-3,
    beta_rolloff=0.2,
)

CLASSICAL_INDICES = [39, 40, 41]

FREQ_GRID_RESOLUTION_HZ = 0.1e9
FREQ_GRID_PADDING_FACTOR = 1.5

OSA_RBW_HZ = 1.0e9
OSA_CSV_PATH = _PROJECT_ROOT / "data" / "osa"

# ---- 信号 PSD 对比专用模型列表（与 plot_noise_spectrum.py 保持一致） ----
SIGNAL_TX_SPECS = {
    "discrete": {
        "label": "Discrete",
        "spectrum_type": SpectrumType.SINGLE_FREQ,
        "continuous": False,
        "beta_rolloff": None,
    },
    "rc_beta0": {
        "label": "RC (β=0, ≡Rect)",
        "spectrum_type": SpectrumType.RAISED_COSINE,
        "continuous": True,
        "beta_rolloff": 0.0,
    },
    "rc_beta001": {
        "label": "RC (β=0.01)",
        "spectrum_type": SpectrumType.RAISED_COSINE,
        "continuous": True,
        "beta_rolloff": 0.01,
    },
    "rc_beta01": {
        "label": "RC (β=0.1)",
        "spectrum_type": SpectrumType.RAISED_COSINE,
        "continuous": True,
        "beta_rolloff": 0.1,
    },
    "rc_beta05": {
        "label": "RC (β=0.5)",
        "spectrum_type": SpectrumType.RAISED_COSINE,
        "continuous": True,
        "beta_rolloff": 0.5,
    },
    "osa": {
        "label": "OSA",
        "spectrum_type": SpectrumType.OSA_SAMPLED,
        "continuous": True,
        "beta_rolloff": None,
    },
}


def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV files found in {OSA_CSV_PATH}")
    return csv_files[0]


def _build_frequency_grid(config: WDMConfig) -> np.ndarray:
    half_span = (config.N_ch - 1) / 2 * config.channel_spacing
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = config.f_center - half_span - padding
    f_max = config.f_center + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / FREQ_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_discrete_signal_psd(
    wdm_grid, f_grid: np.ndarray
) -> np.ndarray:
    """离散模型信号功率谱：每个格点存储信道功率 P [W]。

    离散信道没有 PSD 概念，只有信道功率 P。
    这里用 P（而非 P/df）作为"谱"值，使与连续模型的 PSD×Δf 量纲一致。
    """
    psd = np.zeros_like(f_grid, dtype=np.float64)
    for ch in wdm_grid.get_classical_channels():
        idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
        psd[idx] += ch.power
    return psd


def _integrate_psd(f_grid: np.ndarray, psd: np.ndarray, is_discrete: bool = False) -> float:
    """均匀网格黎曼和积分。

    Parameters
    ----------
    f_grid : ndarray
        频率网格 [Hz]
    psd : ndarray
        PSD [W/Hz] 或离散功率 [W]
    is_discrete : bool
        若为 True，psd 存储的是信道功率 P [W]（不含 df），
        积分即为 sum(psd)。
        若为 False，psd 存储的是 PSD [W/Hz]，积分用 sum(psd * df)。
    """
    if is_discrete:
        return float(np.sum(psd))
    df = float(np.mean(np.diff(f_grid)))
    return float(np.sum(psd * df))


def _compute_signal_tx_results(
    wdm_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
) -> list[SignalPSDResult]:
    """计算所有信号模型的 PSD 结果。

    Parameters
    ----------
    wdm_config : WDMConfig
        基础 WDM 配置
    f_grid : ndarray
        频率网格
    osa_csv_path : Path
        OSA CSV 文件路径

    Returns
    -------
    list[SignalPSDResult]
    """
    from scripts.plot_noise_spectrum import _build_signal_tx_grid  # noqa: E402

    results: list[SignalPSDResult] = []

    for model_key, spec in SIGNAL_TX_SPECS.items():
        grid = _build_signal_tx_grid(model_key, wdm_config, f_grid, osa_csv_path)

    results: list[SignalPSDResult] = []

    for model_key, spec in SIGNAL_TX_SPECS.items():
        grid = _build_signal_tx_grid(model_key, wdm_config, f_grid, osa_csv_path)

        if model_key == "discrete":
            psd = _build_discrete_signal_psd(grid, f_grid)
            integrated_power = _integrate_psd(f_grid, psd, is_discrete=True)
        else:
            psd = grid.get_total_psd()
            integrated_power = _integrate_psd(f_grid, psd, is_discrete=False)

        results.append(
            SignalPSDResult(
                key=model_key,
                label=spec["label"],
                color=get_model_color(model_key),
                f_hz=f_grid,
                psd_W_per_Hz=psd,
                integrated_power_W=integrated_power,
            )
        )

    return results


def _export_csv(results: list[SignalPSDResult], out_dir: Path) -> None:
    """导出信号 PSD 数据到 CSV（W + dBm）。"""
    header = ["f_THz"]
    rows: list[list[float]] = []

    # 对齐频率网格
    ref_f = results[0].f_hz
    for r in results:
        assert np.allclose(r.f_hz, ref_f), f"Frequency mismatch for {r.key}"

    for r in results:
        header.append(f"{r.key}_W_per_Hz")
        header.append(f"{r.key}_dBm_per_Hz")
        rows.append([])

    csv_lines = [",".join(header)]
    f_THz = ref_f / 1e12

    def _to_dBm(v: float) -> float:
        return 10 * np.log10(max(v, 1e-30)) + 30

    for i in range(len(ref_f)):
        row = [f"{f_THz[i]:.6f}"]
        for r in results:
            row.append(f"{r.psd_W_per_Hz[i]:.6e}")
            row.append(f"{_to_dBm(r.psd_W_per_Hz[i]):.3f}")
        csv_lines.append(",".join(row))

    # 积分功率行
    int_row = ["# integrated_power_W"]
    for r in results:
        int_row.append(f"{r.key}:{r.integrated_power_W:.6e}")
    csv_lines.append("")
    csv_lines.append(",".join(int_row))

    (out_dir / "signal_tx.csv").write_text("\n".join(csv_lines), encoding="utf-8")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    output_dir = _PROJECT_ROOT / "outputs" / "phase4_N80_C3" / "Signal_TX"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    wdm_config = WDMConfig(**WDM_PARAMS, quantum_channel_indices=[1, 2])
    f_grid = _build_frequency_grid(wdm_config)
    osa_csv_path = _resolve_osa_csv()

    print("Computing signal PSD for all models...")
    results = _compute_signal_tx_results(wdm_config, f_grid, osa_csv_path)

    # 打印积分功率验证
    print("\nIntegrated powers (should all be ~P0 = 1e-3 W):")
    for r in results:
        print(f"  {r.key:15s}: {r.integrated_power_W:.6e} W")

    print("\nGenerating figures...")
    fig_W = make_signal_psd_comparison_figure(results, unit="W")
    fig_dBm = make_signal_psd_comparison_figure(results, unit="dBm")

    fig_W.savefig(output_dir / "signal_tx_linear.png", dpi=150, bbox_inches="tight")
    fig_dBm.savefig(output_dir / "signal_tx_log.png", dpi=150, bbox_inches="tight")

    print(f"  Saved: {output_dir / 'signal_tx_linear.png'}")
    print(f"  Saved: {output_dir / 'signal_tx_log.png'}")

    _export_csv(results, output_dir)
    print(f"  Saved: {output_dir / 'signal_tx.csv'}")

    import matplotlib.pyplot as plt
    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
