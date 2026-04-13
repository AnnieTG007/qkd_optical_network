"""
Dash app: Noise vs Quantum Channel (fiber length slider).
Usage: python scripts/plot_noise_dash_ch.py --type=fwm|sprs|both|only_signal|with_signal
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

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
    precompute_by_channel,
    set_power_override,
)


def _to_dbm(values_w: np.ndarray) -> np.ndarray:
    out = np.full_like(values_w, np.nan, dtype=np.float64)
    mask = values_w > 0
    out[mask] = 10.0 * np.log10(values_w[mask] / 1e-3)
    return out


def _global_ranges(all_data: dict, model_keys: list[str]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute y-axis ranges for the plot.

    Returns (y_log, y_dbm). All data is now in [W] (power per bin or channel power).
    """
    positives: list[float] = []
    for curve_data in all_data.values():
        for model_key in model_keys:
            entry = curve_data.get(model_key, {})
            fwd = np.asarray(entry.get("fwd", []), dtype=np.float64)
            bwd = np.asarray(entry.get("bwd", []), dtype=np.float64)
            positives.extend(fwd[fwd >= NOISE_FLOOR_W].tolist())
            positives.extend(bwd[bwd >= NOISE_FLOOR_W].tolist())
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
print(f"Precomputing channel sweep for type={NOISE_TYPE}")
t0 = time.time()

osa_csv_path = _resolve_osa_csv()
base_quantum_indices = [
    i
    for i in range(int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 1))
    if i not in CLASSICAL_INDICES
]
base_config = _build_wdm_config(base_quantum_indices)
noise_f_grid = _build_noise_frequency_grid(base_config)
specs = load_model_specs("fwm_noise")
model_keys = get_noise_model_keys(NOISE_TYPE)

# Initial precompute at default power (0 dBm)
set_power_override(0.0)
ALL_BY_CH, VALID_L_INDICES = precompute_by_channel(
    noise_type=NOISE_TYPE,
    specs=specs,
    LENGTHS_KM=LENGTHS_KM,
    base_config=base_config,
    noise_f_grid=noise_f_grid,
    osa_csv_path=osa_csv_path,
    fiber_params=FIBER_PARAMS,
)
if not VALID_L_INDICES:
    raise RuntimeError(f"No valid lengths found for noise type {NOISE_TYPE!r}")

Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(ALL_BY_CH, model_keys)
elapsed = time.time() - t0
print(f"Precompute done in {elapsed:.1f}s. Valid selections: {len(VALID_L_INDICES)}")

# Diagnostic: verify continuous vs discrete model data shapes
l_diag = VALID_L_INDICES[0]
print(f"\n--- Diagnostic: noise_f_grid shape = {noise_f_grid.shape} ---")
for mk in model_keys:
    entry = ALL_BY_CH[l_diag].get(mk, {})
    x = np.asarray(entry.get("x", []), dtype=np.float64)
    fwd = np.asarray(entry.get("fwd", []), dtype=np.float64)
    x_kind = entry.get("x_kind", "N/A")
    y_kind = entry.get("y_kind", "N/A")
    print(f"  {mk}: x.shape={x.shape}, fwd.shape={fwd.shape}, x_kind={x_kind}, y_kind={y_kind}")

app = Dash(__name__)
app.index_string = app.index_string.replace("</body>", "<script>" + _LEGEND_SYNC_JS + "</script></body>")

step = max(1, len(VALID_L_INDICES) // 10)
slider_marks = {
    i: {"label": f"{LENGTHS_KM[VALID_L_INDICES[i]]:.0f}", "style": {"font-size": "9px"}}
    for i in range(0, len(VALID_L_INDICES), step)
}
if (len(VALID_L_INDICES) - 1) not in slider_marks:
    last = len(VALID_L_INDICES) - 1
    slider_marks[last] = {"label": f"{LENGTHS_KM[VALID_L_INDICES[last]]:.0f}", "style": {"font-size": "9px"}}

app.layout = html.Div(
    [
        html.H2(f"Noise vs Channel Frequency [{NOISE_TYPE}]"),
        html.Div(
            [
                html.Label("Fiber Length [km]"),
                dcc.Slider(
                    id="length-slider",
                    min=0,
                    max=len(VALID_L_INDICES) - 1,
                    step=1,
                    value=min(len(VALID_L_INDICES) // 2, len(VALID_L_INDICES) - 1),
                    marks=slider_marks,
                ),
            ],
            style=dict(width="92%", padding="10px 0"),
        ),
        html.Div(id="length-display", style=dict(fontFamily="Courier New", fontSize="13px", padding="4px 0 10px 0")),
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
        dcc.Graph(id="channel-graph"),
    ],
    style=dict(fontFamily="Arial", padding="20px"),
)


@app.callback(
    Output("length-display", "children"),
    Input("length-slider", "value"),
)
def update_display(selection_idx: int) -> str:
    l_idx = VALID_L_INDICES[selection_idx]
    return f"Selected fiber length: {LENGTHS_KM[l_idx]:.1f} km"


@app.callback(
    Output("power-display", "children"),
    Output("channel-graph", "figure"),
    Input("power-slider", "value"),
    State("length-slider", "value"),
    prevent_initial_call=False,
)
def update_power_and_graph(power_dbm: float, length_selection_idx: int) -> tuple[str, go.Figure]:
    # Apply power override and recompute
    set_power_override(power_dbm)
    all_by_ch, valid_l_indices = precompute_by_channel(
        noise_type=NOISE_TYPE,
        specs=specs,
        LENGTHS_KM=LENGTHS_KM,
        base_config=base_config,
        noise_f_grid=noise_f_grid,
        osa_csv_path=osa_csv_path,
        fiber_params=FIBER_PARAMS,
    )
    global Y_LOG_RANGE, Y_DBM_RANGE
    Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(all_by_ch, model_keys)
    l_idx = valid_l_indices[length_selection_idx]
    fig = _make_figure(all_by_ch[l_idx], l_idx)
    return f"Classical power: {power_dbm:.1f} dBm (recomputed)", fig


@app.callback(
    Output("channel-graph", "figure", allow_duplicate=True),
    Input("length-slider", "value"),
    State("power-slider", "value"),
    prevent_initial_call=True,
)
def update_graph_on_length(length_selection_idx: int, power_dbm: float) -> go.Figure:
    l_idx = VALID_L_INDICES[length_selection_idx]
    fig = _make_figure(ALL_BY_CH[l_idx], l_idx)
    return fig


def _make_figure(sweep: dict, l_idx: int) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Noise Power [W]", "Noise Power [dBm]"),
    )

    for model_key in model_keys:
        spec = specs[model_key]
        entry = sweep.get(model_key, {})
        if not entry or entry.get("fwd") is None or len(entry.get("fwd", [])) == 0:
            continue

        x_data = np.asarray(entry.get("x", []), dtype=np.float64)
        x_thz = x_data / 1e12

        for direction, dash_style in (("fwd", "solid"), ("bwd", "dot")):
            arr_w = np.asarray(entry.get(direction, []), dtype=np.float64)

            name = f"{spec['label']} ({direction})"
            legendgroup = f"{model_key}-{direction}"

            if spec["continuous"] or NOISE_TYPE == "with_signal":
                # Continuous model or with_signal: lines on full f_grid
                mode = "lines"
                line_cfg = dict(color=spec["color"], width=2.0, dash=dash_style)
                marker_cfg = None
                x_plot = x_thz
                y_plot = np.where(arr_w >= NOISE_FLOOR_W, arr_w, np.nan)
                y_db = _to_dbm(y_plot)
                hover_w = "f=%{x:.4f} THz<br>P=%{y:.3e} W<extra>" + name + "</extra>"
                hover_db = "f=%{x:.4f} THz<br>P=%{y:.2f} dBm<extra>" + name + "</extra>"
            else:
                # Discrete model: markers only at channel centers
                mask = arr_w >= NOISE_FLOOR_W
                if not np.any(mask):
                    continue
                mode = "markers"
                line_cfg = None
                marker_cfg = dict(size=6, color=spec["color"])
                x_plot = x_thz[mask]
                y_plot = arr_w[mask]
                y_db = _to_dbm(arr_w[mask])
                hover_w = "f=%{x:.4f} THz<br>P=%{y:.3e} W<extra>" + name + "</extra>"
                hover_db = "f=%{x:.4f} THz<br>P=%{y:.2f} dBm<extra>" + name + "</extra>"

            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode=mode,
                    line=line_cfg,
                    marker=marker_cfg,
                    connectgaps=False,
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
                    x=x_plot,
                    y=y_db,
                    mode=mode,
                    line=line_cfg,
                    marker=marker_cfg,
                    connectgaps=False,
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=hover_db,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="Frequency [THz]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [THz]", row=1, col=2)

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
        title=f"Noise vs Channel Frequency [{NOISE_TYPE}] | L = {LENGTHS_KM[l_idx]:.1f} km",
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8051")
    app.run(debug=False, port=8051)
