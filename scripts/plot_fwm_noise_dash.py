"""FWM 噪声交互式 Dash 应用（支持波长滑条）。

运行方式：
    python scripts/plot_fwm_noise_dash.py
    然后在浏览器打开 http://127.0.0.1:8050

滑条可以动态选择量子信道频率，选择后重新计算 FWM 噪声。
依赖：pip install dash plotly pandas numpy
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- 项目路径 setup ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.plot_config import get_color, load_model_specs
from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import build_wdm_grid
from qkd_sim.physical.noise import DiscreteFWMSolver

# ============================================================================
# 参数配置
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
NOISE_GRID_RESOLUTION_HZ = 5e9
FREQ_GRID_PADDING_FACTOR = 1.5
LENGTHS_KM = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
OSA_RBW_HZ = 1.0e9
OSA_CSV_PATH = _PROJECT_ROOT / "data" / "osa"

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


def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV found in {OSA_CSV_PATH}")
    return csv_files[0]


def _build_noise_frequency_grid(config: WDMConfig) -> np.ndarray:
    half_span = (config.N_ch - 1) / 2 * config.channel_spacing
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = config.f_center - half_span - padding
    f_max = config.f_center + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / NOISE_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_model_grid(
    model_key: str,
    spec: dict,
    base_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
) -> "WDMGrid":
    """构建单模型 WDMGrid。"""
    from qkd_sim.physical.signal import SpectrumType

    if spec["beta_rolloff"] is not None:
        model_config = WDMConfig(
            f_center=base_config.f_center,
            N_ch=base_config.N_ch,
            channel_spacing=base_config.channel_spacing,
            B_s=base_config.B_s,
            P0=base_config.P0,
            beta_rolloff=spec["beta_rolloff"],
            quantum_channel_indices=base_config.quantum_channel_indices,
        )
    else:
        model_config = base_config

    if spec["spectrum_type"] == SpectrumType.OSA_SAMPLED:
        return build_wdm_grid(
            config=model_config,
            spectrum_type=spec["spectrum_type"],
            f_grid=f_grid,
            osa_csv_path=osa_csv_path,
            osa_rbw=OSA_RBW_HZ,
            classical_channel_indices=CLASSICAL_INDICES,
        )
    return build_wdm_grid(
        config=model_config,
        spectrum_type=spec["spectrum_type"],
        f_grid=f_grid,
        classical_channel_indices=CLASSICAL_INDICES,
    )


def _build_wdm_config(quantum_indices: list[int]) -> WDMConfig:
    return WDMConfig(**WDM_PARAMS, quantum_channel_indices=list(quantum_indices))


# ============================================================================
# 预计算（启动时一次性完成）
# ============================================================================

osa_csv_path = _resolve_osa_csv()
base_quantum_indices = [i for i in range(WDM_PARAMS["N_ch"]) if i not in CLASSICAL_INDICES]
base_config = _build_wdm_config(base_quantum_indices)
noise_f_grid = _build_noise_frequency_grid(base_config)
df = float(np.mean(np.diff(noise_f_grid)))

# 构建默认长度的 Fiber
fp = dict(FIBER_PARAMS)
default_fiber = Fiber(FiberConfig(**fp))
fwm_solver = DiscreteFWMSolver()

specs = load_model_specs("fwm_noise")

# 所有量子信道频率列表（用于滑条标注）
all_quantum_freqs = np.array([
    base_config.f_center + (i - (WDM_PARAMS["N_ch"] - 1) / 2) * WDM_PARAMS["channel_spacing"]
    for i in base_quantum_indices
])  # shape (N_q,)


def _compute_psd_for_quantum_channel(q_index_in_list: int) -> dict[str, np.ndarray]:
    """对指定量子信道索引（在 all_quantum_freqs 中的位置）计算各模型 FWM PSD。

    原理：构造只含单个量子信道的 wdm_grid，重新计算。
    """
    q_global_idx = base_quantum_indices[q_index_in_list]
    single_q_config = _build_wdm_config([q_global_idx])
    results = {}
    for model_key, spec in specs.items():
        grid = _build_model_grid(model_key, spec, single_q_config, noise_f_grid, osa_csv_path)
        psd = fwm_solver.compute_fwm_spectrum_conti(default_fiber, grid, noise_f_grid, direction="forward")
        results[model_key] = psd
    return results


def _compute_length_sweep_for_quantum_channel(
    q_index_in_list: int,
) -> dict[str, dict[str, np.ndarray]]:
    """对指定量子信道计算前向/后向噪声 vs 光纤长度。"""
    q_global_idx = base_quantum_indices[q_index_in_list]
    single_q_config = _build_wdm_config([q_global_idx])

    results = {}
    for model_key, spec in specs.items():
        grid = _build_model_grid(model_key, spec, single_q_config, noise_f_grid, osa_csv_path)
        fwd_arr = np.zeros_like(LENGTHS_KM, dtype=np.float64)
        bwd_arr = np.zeros_like(LENGTHS_KM, dtype=np.float64)
        for i, L_km in enumerate(LENGTHS_KM):
            fp_l = dict(FIBER_PARAMS)
            fp_l["L_km"] = float(L_km)
            fiber = Fiber(FiberConfig(**fp_l))
            if spec["continuous"]:
                fwd = fwm_solver.compute_forward_conti(fiber, grid, noise_f_grid)
                bwd = fwm_solver.compute_backward_conti(fiber, grid, noise_f_grid)
            else:
                fwd = fwm_solver.compute_forward(fiber, grid)
                bwd = fwm_solver.compute_backward(fiber, grid)
            fwd_arr[i] = float(fwd[0])
            bwd_arr[i] = float(bwd[0])
        results[model_key] = dict(fwd=fwd_arr, bwd=bwd_arr)
    return results


# ============================================================================
# Dash 应用
# ============================================================================

app = Dash(__name__)

app.layout = html.Div([
    html.H2("FWM Noise Interactive — Quantum Channel Wavelength Slider"),
    html.P(
        "滑条选择量子信道频率，选择后自动重新计算 FWM 噪声。",
        style=dict(color="gray"),
    ),

    # ---- 滑条：选择量子信道 ----
    html.Div([
        html.Label("量子信道索引 (Quantum Channel Index)"),
        dcc.Slider(
            id="q-channel-slider",
            min=0,
            max=len(all_quantum_freqs) - 1,
            step=1,
            value=len(all_quantum_freqs) // 2,  # 默认居中
            marks={
                i: {
                    "label": f"Ch{i} ({all_quantum_freqs[i]/1e12:.4f} THz)",
                    "style": {"font-size": "8px", "white-space": "nowrap"},
                }
                for i in range(0, len(all_quantum_freqs), max(1, len(all_quantum_freqs) // 10))
            },
        ),
    ], style=dict(width="90%", padding="10px")),

    # 当前选择信息
    html.Div(id="q-channel-display", style=dict(padding="5px 10px", fontFamily="monospace")),

    # ---- FWM PSD 对比图 ----
    html.H3("FWM Noise PSD Comparison"),
    dcc.Graph(id="fwm-psd-graph"),

    # ---- 长度扫描图（前向 + 后向）----
    html.H3("FWM Noise vs Fiber Length"),
    dcc.Graph(id="fwm-length-graph"),
], style=dict(fontFamily="Arial", padding="20px"))


@callback(
    Output("q-channel-display", "children"),
    Input("q-channel-slider", "value"),
)
def update_q_display(q_idx: int) -> str:
    freq_thz = all_quantum_freqs[q_idx] / 1e12
    ch_idx = base_quantum_indices[q_idx]
    return f"Selected: Ch {ch_idx} ({freq_thz:.4f} THz, wavelength ≈ {299792458/freq_thz/1e12*1e9:.2f} nm)"


def _to_dBm(v: np.ndarray | float) -> np.ndarray | float:
    v_arr = np.asarray(v, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(v_arr, 1e-30)) + 30.0


@callback(
    Output("fwm-psd-graph", "figure"),
    Input("q-channel-slider", "value"),
)
def update_psd_graph(q_idx: int) -> go.Figure:
    psd_results = _compute_psd_for_quantum_channel(q_idx)
    freq_THz = noise_f_grid / 1e12

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "FWM Noise Power per Bin (W, Log Scale)",
            "FWM Noise Power per Bin (dBm, Linear Scale)",
        ),
    )

    for model_key, spec in specs.items():
        power_bin_W = psd_results[model_key] * df
        mask = power_bin_W > 0
        y_W = power_bin_W[mask]
        y_dBm = _to_dBm(y_W)
        f_THz = freq_THz[mask]

        _ht_W = "f=%{x:.4f} THz<br>P=%{y:.3e} W<extra>" + spec["label"] + "</extra>"
        _ht_dBm = "f=%{x:.4f} THz<br>P=%{y:.2f} dBm<extra>" + spec["label"] + "</extra>"

        if spec["continuous"]:
            mode = "lines"
        else:
            mode = "markers"

        fig.add_trace(go.Scatter(
            x=f_THz, y=y_W, mode=mode,
            line=dict(color=spec["color"], width=2.0) if spec["continuous"] else None,
            marker=dict(size=6, color=spec["color"], symbol="circle"),
            name=spec["label"],
            legendgroup=model_key,
            showlegend=True,
            hovertemplate=_ht_W,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=f_THz, y=y_dBm, mode=mode,
            line=dict(color=spec["color"], width=2.0) if spec["continuous"] else None,
            marker=dict(size=6, color=spec["color"], symbol="circle"),
            name=spec["label"],
            legendgroup=model_key,
            showlegend=False,
            hovertemplate=_ht_dBm,
        ), row=1, col=2)

    # 动态范围
    all_p = [psd_results[k] * df for k in psd_results]
    nonzero = [p[p > 0] for p in all_p if np.any(p > 0)]
    if nonzero:
        y_bot = min(np.concatenate(nonzero)) / 1e6
        y_top = max(np.concatenate(nonzero)) * 10.0
        y_bot_dBm = _to_dBm(y_bot)
        y_top_dBm = _to_dBm(y_top)
    else:
        y_bot, y_top = 1e-15, 1e-8
        y_bot_dBm, y_top_dBm = -120.0, -60.0

    f_min, f_max = freq_THz.min(), freq_THz.max()
    fig.update_xaxes(title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=1)
    fig.update_xaxes(title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=2)
    fig.update_yaxes(title_text="Power per Bin [W]", range=[y_bot, y_top], type="log",
                    tickformat="%.0e", exponentformat="none", row=1, col=1)
    fig.update_yaxes(title_text="Power per Bin [dBm]", range=[y_bot_dBm, y_top_dBm],
                    tickformat="%.1f", exponentformat="none", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(
        title="FWM Noise PSD — W (left) / dBm (right)",
        template="plotly_white",
        height=450,
        legend=dict(groupclick="toggleitem"),
    )
    return fig


@callback(
    Output("fwm-length-graph", "figure"),
    Input("q-channel-slider", "value"),
)
def update_length_graph(q_idx: int) -> go.Figure:
    sweep_results = _compute_length_sweep_for_quantum_channel(q_idx)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Forward FWM Noise vs Length (W, Log Scale)",
            "Backward FWM Noise vs Length (W, Log Scale)",
        ),
    )

    for model_key, spec in specs.items():
        d = sweep_results[model_key]
        mask_fwd = d["fwd"] > 0
        mask_bwd = d["bwd"] > 0

        _ht = "L=%{x:.1f} km<br>P=%{y:.3e} W<extra>" + spec["label"] + "</extra>"

        fig.add_trace(go.Scatter(
            x=LENGTHS_KM[mask_fwd], y=d["fwd"][mask_fwd],
            mode="lines+markers",
            line=dict(color=spec["color"], width=2.0),
            marker=dict(size=6, color=spec["color"]),
            name=spec["label"],
            legendgroup=model_key,
            showlegend=True,
            hovertemplate=_ht,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=LENGTHS_KM[mask_bwd], y=d["bwd"][mask_bwd],
            mode="lines+markers",
            line=dict(color=spec["color"], width=2.0),
            marker=dict(size=6, color=spec["color"]),
            name=spec["label"],
            legendgroup=model_key,
            showlegend=False,
            hovertemplate=_ht,
        ), row=1, col=2)

    # 动态范围
    all_f = np.concatenate([sweep_results[k]["fwd"][sweep_results[k]["fwd"] > 0] for k in sweep_results])
    all_b = np.concatenate([sweep_results[k]["bwd"][sweep_results[k]["bwd"] > 0] for k in sweep_results])
    if all_f.size:
        y_bot_f = min(all_f) / 10.0
        y_top_f = max(all_f) * 10.0
        y_bot_b = min(all_b) / 10.0
        y_top_b = max(all_b) * 10.0
    else:
        y_bot_f, y_top_f = 1e-15, 1e-5
        y_bot_b, y_top_b = 1e-15, 1e-5

    fig.update_xaxes(title_text="Fiber Length [km]", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Fiber Length [km]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Noise Power [W]", type="log", range=[y_bot_f, y_top_f],
                    tickformat="%.0e", exponentformat="none", row=1, col=1)
    fig.update_yaxes(title_text="Noise Power [W]", type="log", range=[y_bot_b, y_top_b],
                    tickformat="%.0e", exponentformat="none", row=1, col=2)

    fig.update_layout(
        title="FWM Noise vs Fiber Length — Forward (left) / Backward (right)",
        template="plotly_white",
        height=450,
        legend=dict(groupclick="toggleitem"),
    )
    return fig


if __name__ == "__main__":
    print("Starting FWM Noise Dash app...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)
