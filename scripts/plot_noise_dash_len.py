"""
Dash app: Noise vs Fiber Length (quantum channel slider).
Usage: python scripts/plot_noise_dash_len.py --type=fwm|sprs|both|only_signal|with_signal
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import dash
from dash import Dash, Input, Output, State, dcc, html
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.plot_config import load_model_specs
from scripts.dash_utils import (
    CLASSICAL_INDICES,
    FIBER_PARAMS,
    LENGTHS_KM,
    NOISE_FLOOR_W,
    WDM_PARAMS,
    _LEGEND_SYNC_JS,
    _build_noise_frequency_grid,
    _build_wdm_config,
    _display_channel_label,
    _resolve_osa_csv,
    adaptive_linear_ticks,
    adaptive_log_ticks,
    get_noise_model_keys,
    precompute_by_length,
    set_power_override,
)


def _to_dbm(values_w: np.ndarray) -> np.ndarray:
    out = np.full_like(values_w, np.nan, dtype=np.float64)
    mask = values_w > 0
    out[mask] = 10.0 * np.log10(values_w[mask] / 1e-3)
    return out


def _global_ranges(all_data: dict, model_keys: list[str]) -> tuple[tuple[float, float], tuple[float, float]]:
    positives: list[float] = []
    for curve_data in all_data.values():
        for model_key in model_keys:
            positives.extend(curve_data[model_key]["fwd"][curve_data[model_key]["fwd"] >= NOISE_FLOOR_W].tolist())
            positives.extend(curve_data[model_key]["bwd"][curve_data[model_key]["bwd"] >= NOISE_FLOOR_W].tolist())
    if not positives:
        return (-18.0, -3.0), (-150.0, -30.0)

    positives_arr = np.asarray(positives, dtype=np.float64)
    y_log = (float(np.log10(positives_arr.min()) - 0.3), float(np.log10(positives_arr.max()) + 0.3))
    dbm = _to_dbm(positives_arr)
    y_lin = (float(np.nanmin(dbm) - 3.0), float(np.nanmax(dbm) + 3.0))
    return y_log, y_lin


parser = argparse.ArgumentParser()
parser.add_argument("--type", default="fwm", choices=["fwm", "sprs", "both", "only_signal", "with_signal"])
ARGS = parser.parse_args()
NOISE_TYPE = ARGS.type

print("=" * 60)
print(f"Precomputing length sweep for type={NOISE_TYPE}")
t0 = time.time()

osa_csv_path = _resolve_osa_csv()
base_quantum_indices_list = [
    i
    for i in range(int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 1))
    if i not in CLASSICAL_INDICES
]
base_config = _build_wdm_config(base_quantum_indices_list)
noise_f_grid = _build_noise_frequency_grid(base_config)
specs = load_model_specs("fwm_noise")
model_keys = get_noise_model_keys(NOISE_TYPE)

# Initial precompute at default power (0 dBm)
set_power_override(0.0)
ALL_BY_LEN, VALID_INDICES = precompute_by_length(
    noise_type=NOISE_TYPE,
    specs=specs,
    LENGTHS_KM=LENGTHS_KM,
    base_config=base_config,
    noise_f_grid=noise_f_grid,
    osa_csv_path=osa_csv_path,
    fiber_params=FIBER_PARAMS,
)
if not VALID_INDICES:
    raise RuntimeError(f"No valid channels found for noise type {NOISE_TYPE!r}")

Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(ALL_BY_LEN, model_keys)
elapsed = time.time() - t0
print(f"Precompute done in {elapsed:.1f}s. Valid selections: {len(VALID_INDICES)}")

app = Dash(__name__)
app.index_string = app.index_string.replace("</body>", "<script>" + _LEGEND_SYNC_JS + "</script></body>")

step = max(1, len(VALID_INDICES) // 10)
slider_marks = {
    i: {"label": _display_channel_label(base_quantum_indices_list[VALID_INDICES[i]]), "style": {"font-size": "9px"}}
    for i in range(0, len(VALID_INDICES), step)
}
if (len(VALID_INDICES) - 1) not in slider_marks:
    last = len(VALID_INDICES) - 1
    slider_marks[last] = {"label": _display_channel_label(base_quantum_indices_list[VALID_INDICES[last]]), "style": {"font-size": "9px"}}

app.layout = html.Div(
    [
        html.H2(f"Noise vs Fiber Length [{NOISE_TYPE}]"),
        html.Div(
            [
                html.Label("Channel"),
                dcc.Slider(
                    id="channel-slider",
                    min=0,
                    max=len(VALID_INDICES) - 1,
                    step=1,
                    value=min(len(VALID_INDICES) // 2, len(VALID_INDICES) - 1),
                    marks=slider_marks,
                ),
            ],
            style=dict(width="92%", padding="10px 0"),
        ),
        html.Div(id="channel-display", style=dict(fontFamily="Courier New", fontSize="13px", padding="4px 0 10px 0")),
        html.Div(
            [
                html.Label("Classical Channel Power [dBm]"),
                dcc.Slider(
                    id="power-slider",
                    min=-15,
                    max=15,
                    step=0.5,
                    value=0,
                    marks={-15: "-15", -10: "-10", -5: "-5", 0: "0", 5: "5", 10: "10", 15: "15"},
                ),
            ],
            style=dict(width="92%", padding="10px 0"),
        ),
        html.Div(id="power-display", style=dict(fontFamily="Courier New", fontSize="13px", padding="4px 0 10px 0")),
        dcc.Graph(id="length-graph"),
    ],
    style=dict(fontFamily="Arial", padding="20px"),
)


@app.callback(
    Output("channel-display", "children"),
    Input("channel-slider", "value"),
)
def update_display(selection_idx: int) -> str:
    q_local = VALID_INDICES[selection_idx]
    ch_idx = base_quantum_indices_list[q_local]  # 全局信道索引用于显示
    freq_hz = WDM_PARAMS["start_freq"] + ch_idx * WDM_PARAMS["channel_spacing"]
    wl_nm = 299792458.0 / freq_hz * 1e9
    return f"Selected: {_display_channel_label(ch_idx)} | f = {freq_hz / 1e12:.4f} THz | lambda ~ {wl_nm:.2f} nm"


@app.callback(
    Output("power-display", "children"),
    Output("length-graph", "figure"),
    Input("power-slider", "value"),
    State("channel-slider", "value"),
    prevent_initial_call=False,
)
def update_power_and_graph(power_dbm: float, channel_selection_idx: int) -> tuple[str, go.Figure]:
    # Apply power override and recompute
    set_power_override(power_dbm)
    all_by_len, valid_indices = precompute_by_length(
        noise_type=NOISE_TYPE,
        specs=specs,
        LENGTHS_KM=LENGTHS_KM,
        base_config=base_config,
        noise_f_grid=noise_f_grid,
        osa_csv_path=osa_csv_path,
        fiber_params=FIBER_PARAMS,
    )
    global Y_LOG_RANGE, Y_DBM_RANGE
    Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(all_by_len, model_keys)
    ch_idx = valid_indices[channel_selection_idx]
    fig = _make_figure(all_by_len[ch_idx], ch_idx)
    return f"Classical power: {power_dbm:.1f} dBm (recomputed)", fig


@app.callback(
    Output("length-graph", "figure", allow_duplicate=True),
    Input("channel-slider", "value"),
    State("power-slider", "value"),
    prevent_initial_call=True,
)
def update_graph_on_channel(channel_selection_idx: int, power_dbm: float) -> go.Figure:
    ch_idx = VALID_INDICES[channel_selection_idx]
    fig = _make_figure(ALL_BY_LEN[ch_idx], ch_idx)
    return fig


def _make_figure(sweep: dict, ch_idx: int) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Noise Power [W]", "Noise Power [dBm]"),
    )

    for model_key in model_keys:
        spec = specs[model_key]
        for direction, dash_style in (("fwd", "solid"), ("bwd", "dot")):
            arr_w = np.asarray(sweep[model_key][direction], dtype=np.float64)
            mask = arr_w >= NOISE_FLOOR_W
            if not np.any(mask):
                continue

            name = f"{spec['label']} ({direction})"
            legendgroup = f"{model_key}-{direction}"
            hover_w = "L=%{x:.1f} km<br>P=%{y:.3e} W<extra>" + name + "</extra>"
            hover_dbm = "L=%{x:.1f} km<br>P=%{y:.2f} dBm<extra>" + name + "</extra>"

            fig.add_trace(
                go.Scatter(
                    x=LENGTHS_KM[mask],
                    y=arr_w[mask],
                    mode="lines+markers",
                    line=dict(color=spec["color"], width=2.0, dash=dash_style),
                    marker=dict(size=6, color=spec["color"]),
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=True,
                    hovertemplate=hover_w,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=LENGTHS_KM[mask],
                    y=_to_dbm(arr_w[mask]),
                    mode="lines+markers",
                    line=dict(color=spec["color"], width=2.0, dash=dash_style),
                    marker=dict(size=6, color=spec["color"]),
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=hover_dbm,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="Fiber Length [km]", row=1, col=1)
    fig.update_xaxes(title_text="Fiber Length [km]", row=1, col=2)
    fig.update_yaxes(
        title_text="Power [W]",
        type="log",
        range=list(Y_LOG_RANGE),
        showgrid=True,
        row=1,
        col=1,
        **adaptive_log_ticks(*Y_LOG_RANGE),
    )
    fig.update_yaxes(
        title_text="Power [dBm]",
        type="linear",
        range=list(Y_DBM_RANGE),
        showgrid=True,
        row=1,
        col=2,
        **adaptive_linear_ticks(*Y_DBM_RANGE),
    )
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=520,
        legend=dict(groupclick="toggleitem"),
        title=f"Noise vs Fiber Length [{NOISE_TYPE}]",
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
