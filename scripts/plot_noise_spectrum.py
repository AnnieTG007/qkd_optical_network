"""Phase 4 噪声功率谱绘图脚本。

生成以下对比图：
  1. C 波段 4 种信号模型（Discrete / Rectangular / Raised Cosine / OSA）
     的 FWM 噪声谱、SpRS 噪声谱、总噪声谱、信号谱 — 2×2 布局，W/dBm 各一张
  2. 固定量子信道频率处，噪声功率随光纤长度（1–100 km）变化 — 1×3 布局

运行方式：
    python scripts/plot_noise_spectrum.py

输出（outputs/phase4_N80_C3/ 目录）：
  phase4_model_comparison_W.png
  phase4_model_comparison_dBm.png
  phase4_noise_vs_length.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.noise import DiscreteFWMSolver, DiscreteSPRSSolver, compute_noise
from qkd_sim.physical.signal import (
    SpectrumType, WDMGrid, build_frequency_grid, build_wdm_grid,
)
from qkd_sim.physical.spectrum import (
    ModelLengthSweepResult,
    ModelSpectrumResult,
    get_model_color,
    make_model_comparison_figure,
    make_noise_vs_length_figure,
)

# ===========================================================================
# CONFIG
# ===========================================================================

FIBER_PARAMS = dict(
    alpha_dB_per_km=0.2,
    gamma_per_W_km=1.3,
    D_ps_nm_km=17.0,
    D_slope_ps_nm2_km=0.056,
    L_km=50.0,
    A_eff=80e-12,
    rayleigh_coeff=4.8e-8,
    T_kelvin=300.0,
)

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

OSA_RBW_HZ = 1.0e9  # OSA 分辨率带宽 [Hz]（仅影响中间 PSD 标定，归一化后无影响）

FIXED_QUANTUM_TARGET_FREQ_HZ = 193.4e12  # 固定量子信道目标频率
LENGTH_SWEEP_KM = np.linspace(1.0, 100.0, 100)

OSA_CSV_PATH = _PROJECT_ROOT / "data" / "osa"

MODEL_SPECS = {
    "discrete": {
        "label": "Discrete",
        "spectrum_type": SpectrumType.SINGLE_FREQ,
        "continuous": False,
    },
    "rectangular": {
        "label": "Rectangular",
        "spectrum_type": SpectrumType.RECTANGULAR,
        "continuous": True,
    },
    "raised_cosine": {
        "label": "Raised Cosine",
        "spectrum_type": SpectrumType.RAISED_COSINE,
        "continuous": True,
    },
    "osa": {
        "label": "OSA",
        "spectrum_type": SpectrumType.OSA_SAMPLED,
        "continuous": True,
    },
}


# ===========================================================================
# Helper functions
# ===========================================================================

def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV files found in {OSA_CSV_PATH}")
    return csv_files[0]


def _channel_center_frequencies(config: WDMConfig) -> np.ndarray:
    indices = np.arange(-(config.N_ch - 1) / 2, (config.N_ch + 1) / 2)
    return config.f_center + indices * config.channel_spacing


def _complement_indices(n_ch: int, classical_indices: list[int]) -> list[int]:
    classical_set = set(classical_indices)
    return [i for i in range(n_ch) if i not in classical_set]


def _build_wdm_config(quantum_indices: list[int]) -> WDMConfig:
    return WDMConfig(**WDM_PARAMS, quantum_channel_indices=list(quantum_indices))


def _build_model_grid(
    model_key: str,
    config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
) -> WDMGrid:
    spec = MODEL_SPECS[model_key]
    if spec["spectrum_type"] == SpectrumType.OSA_SAMPLED:
        return build_wdm_grid(
            config=config,
            spectrum_type=spec["spectrum_type"],
            f_grid=f_grid,
            osa_csv_path=osa_csv_path,
            osa_rbw=OSA_RBW_HZ,
            classical_channel_indices=CLASSICAL_INDICES,
        )
    return build_wdm_grid(
        config=config,
        spectrum_type=spec["spectrum_type"],
        f_grid=f_grid,
        classical_channel_indices=CLASSICAL_INDICES,
    )


def _build_discrete_signal_psd(wdm_grid: WDMGrid, f_grid: np.ndarray) -> np.ndarray:
    """离散模型信号 PSD：delta 近似，G = P/df 放在最近格点。"""
    df = float(np.mean(np.diff(f_grid)))
    psd = np.zeros_like(f_grid, dtype=np.float64)
    for ch in wdm_grid.get_classical_channels():
        idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
        psd[idx] += ch.power / df
    return psd


def _get_signal_psd(wdm_grid: WDMGrid, f_grid: np.ndarray, model_key: str) -> np.ndarray:
    if model_key == "discrete":
        return _build_discrete_signal_psd(wdm_grid, f_grid)
    return wdm_grid.get_total_psd()


def _select_reference_quantum_slot(
    config: WDMConfig,
    classical_indices: list[int],
    target_freq_hz: float,
) -> tuple[int, float]:
    """选择 C 波段中心最近的可用量子信道。"""
    all_freqs = _channel_center_frequencies(config)
    quantum_indices = _complement_indices(config.N_ch, classical_indices)
    q_freqs = all_freqs[quantum_indices]
    nearest_local = int(np.argmin(np.abs(q_freqs - target_freq_hz)))
    channel_index = quantum_indices[nearest_local]
    return channel_index, float(all_freqs[channel_index])


# ===========================================================================
# Scenario 1: C-band model comparison
# ===========================================================================

def _compute_spectrum_comparison_results(
    fiber: Fiber,
    config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
) -> list[ModelSpectrumResult]:
    results: list[ModelSpectrumResult] = []
    for model_key, spec in MODEL_SPECS.items():
        grid = _build_model_grid(model_key, config, f_grid, osa_csv_path)
        noise = compute_noise(
            "all",
            fiber,
            grid,
            sprs_solver=DiscreteSPRSSolver(),
            fwm_solver=DiscreteFWMSolver(),
            continuous=spec["continuous"],
        )
        q_freqs = np.array(
            [ch.f_center for ch in grid.get_quantum_channels()], dtype=np.float64
        )
        signal_psd = _get_signal_psd(grid, f_grid, model_key)
        results.append(
            ModelSpectrumResult(
                key=model_key,
                label=spec["label"],
                color=get_model_color(model_key),
                f_signal_hz=f_grid,
                signal_psd_W_per_Hz=signal_psd,
                f_quantum_hz=q_freqs,
                fwm_W=np.asarray(noise["fwm_fwd"], dtype=np.float64),
                sprs_W=np.asarray(noise["sprs_fwd"], dtype=np.float64),
            )
        )
    return results


# ===========================================================================
# Scenario 2: Noise vs fiber length
# ===========================================================================

def _compute_length_sweep_results(
    quantum_index: int,
    f_grid: np.ndarray,
    osa_csv_path: Path,
) -> list[ModelLengthSweepResult]:
    config = _build_wdm_config([quantum_index])
    results: list[ModelLengthSweepResult] = []
    for model_key, spec in MODEL_SPECS.items():
        grid = _build_model_grid(model_key, config, f_grid, osa_csv_path)
        fwm_vals: list[float] = []
        sprs_vals: list[float] = []
        for length_km in LENGTH_SWEEP_KM:
            fp = dict(FIBER_PARAMS)
            fp["L_km"] = float(length_km)
            fiber = Fiber(FiberConfig(**fp))
            noise = compute_noise(
                "all",
                fiber,
                grid,
                sprs_solver=DiscreteSPRSSolver(),
                fwm_solver=DiscreteFWMSolver(),
                continuous=spec["continuous"],
            )
            fwm_vals.append(float(noise["fwm_fwd"][0]))
            sprs_vals.append(float(noise["sprs_fwd"][0]))
        results.append(
            ModelLengthSweepResult(
                key=model_key,
                label=spec["label"],
                color=get_model_color(model_key),
                length_km=np.asarray(LENGTH_SWEEP_KM, dtype=np.float64),
                fwm_W=np.asarray(fwm_vals, dtype=np.float64),
                sprs_W=np.asarray(sprs_vals, dtype=np.float64),
            )
        )
    return results


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 72)
    print("Phase 4 — 多信号模型噪声功率谱对比")
    print("=" * 72)

    osa_csv_path = _resolve_osa_csv()
    print(f"OSA CSV: {osa_csv_path.name}")

    base_quantum_indices = _complement_indices(WDM_PARAMS["N_ch"], CLASSICAL_INDICES)
    wdm_config = _build_wdm_config(base_quantum_indices)
    fiber = Fiber(FiberConfig(**FIBER_PARAMS))

    f_grid = build_frequency_grid(
        wdm_config,
        resolution=FREQ_GRID_RESOLUTION_HZ,
        padding_factor=FREQ_GRID_PADDING_FACTOR,
    )

    n_classical = len(CLASSICAL_INDICES)
    n_quantum = WDM_PARAMS["N_ch"] - n_classical
    run_tag = f"phase4_N{WDM_PARAMS['N_ch']}_C{n_classical}"
    output_dir = _PROJECT_ROOT / "outputs" / run_tag
    print(f"Output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Scenario 1 ----
    print("\nScenario 1: C-band model comparison")
    spectrum_results = _compute_spectrum_comparison_results(
        fiber=fiber, config=wdm_config, f_grid=f_grid, osa_csv_path=osa_csv_path
    )

    fig_W = make_model_comparison_figure(spectrum_results, unit="W")
    fig_dBm = make_model_comparison_figure(spectrum_results, unit="dBm")
    fig_W.savefig(output_dir / "phase4_model_comparison_W.png", dpi=150, bbox_inches="tight")
    fig_dBm.savefig(output_dir / "phase4_model_comparison_dBm.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"  Saved: phase4_model_comparison_W.png")
    print(f"  Saved: phase4_model_comparison_dBm.png")

    # ---- Scenario 2 ----
    print("\nScenario 2: Noise vs fiber length")
    q_index, q_freq_hz = _select_reference_quantum_slot(
        wdm_config, CLASSICAL_INDICES, FIXED_QUANTUM_TARGET_FREQ_HZ
    )
    print(
        f"  Target {FIXED_QUANTUM_TARGET_FREQ_HZ/1e12:.3f} THz "
        f"→ quantum slot {q_index} @ {q_freq_hz/1e12:.3f} THz"
    )

    length_results = _compute_length_sweep_results(
        quantum_index=q_index, f_grid=f_grid, osa_csv_path=osa_csv_path
    )
    fig_len = make_noise_vs_length_figure(length_results)
    fig_len.savefig(output_dir / "phase4_noise_vs_length.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: phase4_noise_vs_length.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
