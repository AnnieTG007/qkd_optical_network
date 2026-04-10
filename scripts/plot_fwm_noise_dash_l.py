"""Dash App 2 - FWM Noise PSD Comparison with Fiber Length Slider.

Run:
    python scripts/plot_fwm_noise_dash_l.py
    Open http://127.0.0.1:8051

Content: choose fiber length with the slider and update both subplots together.
  - Left: FWM noise power per frequency bin (W, log scale)
  - Right: FWM noise power per frequency bin (dBm, linear scale)

All data is precomputed at startup. The slider only selects an index.
Requires: pip install dash plotly numpy pandas
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# ---- project path setup ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import dash
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qkd_sim.config.plot_config import load_model_specs
from qkd_sim.config.schema import FiberConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.noise import DiscreteFWMSolver
from scripts.dash_utils import (
    CLASSICAL_INDICES,
    FIBER_PARAMS,
    LENGTHS_KM,
    WDM_PARAMS,
    _LEGEND_SYNC_JS,
    _build_model_grid,
    _build_noise_frequency_grid,
    _build_wdm_config,
    _display_channel_label,
    _nice_log_tickvals,
    _resolve_osa_csv,
)


def _psd_to_json(data: dict) -> dict:
    """Serialize ALL_PSD by converting numpy arrays to lists."""
    return {str(Li): {mk: v[mk].tolist() for mk in v.keys()} for Li, v in data.items()}


def _build_discrete_fwm_psd(
    f_grid: np.ndarray,
    quantum_freqs_hz: np.ndarray,
    channel_power_w: np.ndarray,
) -> np.ndarray:
    fwm_psd = np.zeros_like(f_grid, dtype=np.float64)
    df = float(np.mean(np.diff(f_grid)))
    for power_w, f_q_hz in zip(channel_power_w, quantum_freqs_hz):
        if power_w <= 0.0:
            continue
        idx = int(np.argmin(np.abs(f_grid - f_q_hz)))
        fwm_psd[idx] += power_w / df
    return fwm_psd


def _nice_linear_tickvals(
    y_bot: float, y_top: float, n: int = 7
) -> tuple[list[float], list[str]]:
    """Generate explicit evenly spaced tick values/text for a linear axis."""
    if n <= 1:
        vals = [float(y_bot)]
    else:
        step = (y_top - y_bot) / (n - 1)
        vals = [float(y_bot + i * step) for i in range(n)]
    texts = [f"{v:.1f}" for v in vals]
    return vals, texts


print("=" * 60)
print("FWM PSD vs Length Dash - precomputing all fiber lengths x all model PSDs")
print("=" * 60)
t0 = time.time()

osa_csv_path = _resolve_osa_csv()
base_quantum_indices = [
    i
    for i in range(int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 1))
    if i not in CLASSICAL_INDICES
]
base_config = _build_wdm_config(base_quantum_indices)
noise_f_grid = _build_noise_frequency_grid(base_config)
df = float(np.mean(np.diff(noise_f_grid)))
freq_THz = noise_f_grid / 1e12

specs = load_model_specs("fwm_noise")
model_keys = list(specs.keys())
fwm_solver = DiscreteFWMSolver()
N_L = len(LENGTHS_KM)
N_f = len(noise_f_grid)

print(f"  N_length={N_L}, N_freq={N_f}, N_models={len(model_keys)}")
print(f"  Total work: {N_L} x {len(model_keys)} = {N_L * len(model_keys)} PSD solves")

# ALL_PSD[Li][model_key] = np.array(N_f,) for one length index and model.
ALL_PSD: dict[int, dict[str, np.ndarray]] = {}
for Li in range(N_L):
    ALL_PSD[Li] = {mk: np.zeros(N_f) for mk in model_keys}

for Li, L_km in enumerate(LENGTHS_KM):
    fp = dict(FIBER_PARAMS)
    fp["L_km"] = float(L_km)
    fiber = Fiber(FiberConfig(**fp))

    for model_key, spec in specs.items():
        grid = _build_model_grid(model_key, spec, base_config, noise_f_grid, osa_csv_path)
        if spec["continuous"]:
            psd = fwm_solver.compute_fwm_spectrum_conti(
                fiber, grid, noise_f_grid, direction="forward"
            )
        else:
            q_chs = grid.get_quantum_channels()
            f_q_hz = np.array([ch.f_center for ch in q_chs], dtype=np.float64)
            fwm_channel_power = fwm_solver.compute_forward(fiber, grid)
            psd = _build_discrete_fwm_psd(noise_f_grid, f_q_hz, fwm_channel_power)
        ALL_PSD[Li][model_key] = psd.astype(np.float64)

    elapsed = time.time() - t0
    rate = (Li + 1) / elapsed if elapsed > 0 else 0
    remaining = (N_L - Li - 1) / rate if rate > 0 else 0
    print(
        f"  Progress: {Li+1}/{N_L} ({100*(Li+1)/N_L:.0f}%) | "
        f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s"
    )

ALL_PSD_JSON = _psd_to_json(ALL_PSD)

elapsed_total = time.time() - t0
print(f"\nPrecomputation complete. Total elapsed: {elapsed_total:.1f}s")
print("Starting Dash...")
print("=" * 60)

app = Dash(__name__)

_default_index_string = app.index_string
app.index_string = _default_index_string.replace(
    "</body>",
    "<script>" + _LEGEND_SYNC_JS + "</script></body>",
)


def _build_param_text() -> str:
    n_classical = len(CLASSICAL_INDICES)
    ch_lines = []
    for i in CLASSICAL_INDICES:
        f_hz = WDM_PARAMS["start_freq"] + i * WDM_PARAMS["channel_spacing"]
        ch_lines.append(f"{_display_channel_label(i)}: {f_hz / 1e12:.4f} THz")
    lines = [
        (
            f"Sim Parameters  |  L = controlled by slider  |  "
            f"N_classical = {n_classical}  |  "
            f"Spacing = {WDM_PARAMS['channel_spacing'] / 1e9:.0f} GHz  |  "
            f"P0 = {WDM_PARAMS['P0'] * 1e3:.0f} mW"
        ),
        "  |  ".join(ch_lines),
    ]
    return "\n".join(lines)


def _to_dBm(v: np.ndarray | float) -> np.ndarray | float:
    v_arr = np.asarray(v, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(v_arr, 1e-30)) + 30.0


_n_marks = max(1, N_L // 5)
slider_marks = {
    i: {"label": f"{LENGTHS_KM[i]:.0f} km", "style": {"font-size": "10px"}}
    for i in range(0, N_L, _n_marks)
}
if (N_L - 1) not in slider_marks:
    slider_marks[N_L - 1] = {"label": f"{LENGTHS_KM[N_L - 1]:.0f} km"}

app.layout = html.Div(
    [
        html.H2("FWM Noise PSD - Fiber Length Slider (Pre-computed)"),
        html.P(
            "Choose fiber length with the slider. PSD curves update from precomputed data.",
            style=dict(color="gray"),
        ),
        html.Div(
            [
                html.Label("Fiber Length [km]"),
                dcc.Slider(
                    id="l-slider",
                    min=0,
                    max=N_L - 1,
                    step=1,
                    value=list(LENGTHS_KM).index(50.0),
                    marks=slider_marks,
                ),
            ],
            style=dict(width="90%", padding="10px"),
        ),
        html.Div(
            id="l-display",
            style=dict(padding="5px 10px", fontFamily="Courier New", fontSize="13px"),
        ),
        dcc.Store(id="psd-store", data=ALL_PSD_JSON),
        dcc.Graph(id="psd-graph"),
        html.Div(
            id="param-display",
            style=dict(
                padding="10px",
                marginTop="8px",
                backgroundColor="#f9f9f9",
                border="1px solid #cccccc",
                fontFamily="Courier New",
                fontSize="12px",
                whiteSpace="pre-wrap",
                maxWidth="1500px",
                overflowX="auto",
            ),
        ),
    ],
    style=dict(fontFamily="Arial", padding="20px"),
)


@app.callback(
    Output("l-display", "children"),
    Input("l-slider", "value"),
    Input("psd-store", "data"),
)
def update_display(Li: int, store_data: dict) -> str:
    _ = store_data
    L_km = float(LENGTHS_KM[Li])
    return f"Selected Fiber Length: L = {L_km:.0f} km"


@app.callback(
    Output("param-display", "children"),
    Input("l-slider", "value"),
)
def update_param_display(Li: int) -> str:
    _ = Li
    return _build_param_text()


@app.callback(
    Output("psd-graph", "figure"),
    Input("l-slider", "value"),
    Input("psd-store", "data"),
)
def update_graph(Li: int, store_data: dict) -> go.Figure:
    psd_dict = store_data.get(str(Li), {})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "FWM Noise Power per Bin (W, Log Scale)",
            "FWM Noise Power per Bin (dBm, Linear Scale)",
        ),
    )

    for model_key, spec in specs.items():
        raw = psd_dict.get(model_key)
        if raw is None:
            continue

        psd = np.array(raw, dtype=np.float64)
        if psd.size == 0:
            continue

        power_bin_W = psd * df
        if not np.any(power_bin_W > 0):
            continue

        power_bin_dBm = _to_dBm(power_bin_W)

        mask = power_bin_W > 0
        f_THz = freq_THz[mask]
        y_W = power_bin_W[mask]
        y_dBm = power_bin_dBm[mask]

        if f_THz.size == 0 or y_W.size == 0:
            continue

        _ht_W = f"f=%{{x:.4f}} THz<br>P=%{{y:.3e}} W<extra>{spec['label']}</extra>"
        _ht_dBm = f"f=%{{x:.4f}} THz<br>P=%{{y:.2f}} dBm<extra>{spec['label']}</extra>"

        mode = "markers" if not spec["continuous"] else "lines"
        line_dict = dict(color=spec["color"], width=2.0) if spec["continuous"] else None
        marker_dict = dict(size=6, color=spec["color"], symbol="circle")

        fig.add_trace(
            go.Scatter(
                x=f_THz,
                y=y_W,
                mode=mode,
                line=line_dict,
                marker=marker_dict if not spec["continuous"] else None,
                name=spec["label"],
                legendgroup=model_key,
                showlegend=True,
                hovertemplate=_ht_W,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=f_THz,
                y=y_dBm,
                mode=mode,
                line=line_dict,
                marker=marker_dict if not spec["continuous"] else None,
                name=spec["label"],
                legendgroup=model_key,
                showlegend=False,
                hovertemplate=_ht_dBm,
            ),
            row=1,
            col=2,
        )

    all_positive: list[float] = []
    for Li_str in store_data.keys():
        psd_at_length = store_data.get(Li_str, {})
        for mk in model_keys:
            raw = psd_at_length.get(mk)
            if raw is None:
                continue
            arr = np.array(raw, dtype=np.float64)
            if arr.size == 0:
                continue
            power_arr = arr * df
            positive = power_arr[power_arr > 0]
            if positive.size > 0:
                all_positive.extend(positive.tolist())

    if all_positive:
        y_bot_log = round(np.log10(min(all_positive)) - 0.3, 1)
        y_top_log = round(np.log10(max(all_positive)) + 0.3, 1)
        y_bot_dBm = float(_to_dBm(min(all_positive))) - 5.0
        y_top_dBm = float(_to_dBm(max(all_positive))) + 5.0
    else:
        y_bot_log, y_top_log = -15.5, -7.5
        y_bot_dBm, y_top_dBm = -120.0, -60.0

    log_tick_vals, log_tick_texts = _nice_log_tickvals(y_bot_log, y_top_log)
    lin_tick_vals, lin_tick_texts = _nice_linear_tickvals(y_bot_dBm, y_top_dBm)

    f_min, f_max = float(freq_THz.min()), float(freq_THz.max())

    for col in [1, 2]:
        fig.update_xaxes(
            title_text="Frequency [THz]",
            range=[f_min, f_max],
            row=1,
            col=col,
        )

    fig.update_yaxes(
        title_text="Power per Bin [W]",
        type="log",
        range=[y_bot_log, y_top_log],
        tickmode="array",
        tickvals=log_tick_vals,
        ticktext=log_tick_texts,
        tickformat=None,
        exponentformat="none",
        showgrid=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Power per Bin [dBm]",
        range=[y_bot_dBm, y_top_dBm],
        tickmode="array",
        tickvals=lin_tick_vals,
        ticktext=lin_tick_texts,
        tickformat=None,
        exponentformat="none",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="FWM Noise PSD - W (left) / dBm (right)",
        template="plotly_white",
        width=1500,
        height=500,
        legend=dict(groupclick="toggleitem"),
        uirevision="fixed",
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8051")
    app.run(debug=False, port=8051)
