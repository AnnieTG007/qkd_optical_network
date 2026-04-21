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
    PRECOMPUTE_POWER_LEVELS,
    WDM_PARAMS,
    _LEGEND_SYNC_JS,
    _build_caption,
    _build_noise_frequency_grid,
    _build_wdm_config,
    _display_channel_label,
    _resolve_osa_csv,
    adaptive_linear_ticks,
    adaptive_log_ticks,
    add_strategy_cli_args,
    export_noise_vs_length_xlsx,
    export_simulation_report,
    get_noise_model_keys,
    override_strategy_from_cli,
    precompute_by_length_all_powers,
    print_compute_device,
    set_power_override,
    _POWER_CACHE,
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
parser.add_argument("--modulation", default="16qam", choices=["ook", "16qam"])
parser.add_argument(
    "--export-excel",
    action="store_true",
    help="After precomputation, export Excel files and exit (do not start Dash server).",
)
parser.add_argument(
    "--export-only",
    action="store_true",
    help="Shorthand for --export-excel.",
)
add_strategy_cli_args(parser)
ARGS = parser.parse_args()
NOISE_TYPE = ARGS.type
import scripts.dash_utils as _du
_du.MODULATION_FORMAT = ARGS.modulation

# 策略参数覆盖：在预计算之前用 CLI 参数替换 CLASSICAL_INDICES
if ARGS.strategy_name or ARGS.num_classical or ARGS.reference_channel:
    _du.CLASSICAL_INDICES = override_strategy_from_cli(
        ARGS.strategy_name, ARGS.num_classical, ARGS.reference_channel
    )
    print(
        f"Strategy override: CLASSICAL_INDICES = {_du.CLASSICAL_INDICES} "
        f"(name={ARGS.strategy_name}, N={ARGS.num_classical}, ref={ARGS.reference_channel})"
    )

print("=" * 60)
print_compute_device()
print(f"Precomputing ALL power levels for type={NOISE_TYPE}")
t0 = time.time()

osa_csv_path = _resolve_osa_csv()
# base_quantum_indices_list: ITU G.694.1 channel numbers (1-based), e.g. [1, 2, ..., 61]
base_quantum_indices_list = [
    itn
    for itn in range(1, int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 2))
    if itn not in CLASSICAL_INDICES
]
base_config = _build_wdm_config(base_quantum_indices_list)
noise_f_grid = _build_noise_frequency_grid(base_config)
specs = load_model_specs(f"fwm_noise_{ARGS.modulation}")
model_keys = get_noise_model_keys(NOISE_TYPE)

# Precompute ALL power levels at startup
ALL_BY_LEN, VALID_INDICES = precompute_by_length_all_powers(
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

if ARGS.export_only or ARGS.export_excel:
    out_dir = _PROJECT_ROOT / "data" / "precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)
    # quantum_center_freqs: same as used for base_config
    quantum_center_freqs = np.array(
        [
            WDM_PARAMS["start_freq"] + (itn - WDM_PARAMS["start_channel"]) * WDM_PARAMS["channel_spacing"]
            for itn in base_quantum_indices_list
        ],
        dtype=np.float64,
    )
    xlsx_path = out_dir / "noise_vs_length.xlsx"
    export_noise_vs_length_xlsx(
        all_by_len=_POWER_CACHE.get(("len", 0.0), {}),
        model_keys=model_keys,
        LENGTHS_KM=LENGTHS_KM,
        quantum_center_freqs=quantum_center_freqs,
        output_path=xlsx_path,
    )
    report_path = out_dir / "simulation_report.txt"
    export_simulation_report(
        fiber_params=FIBER_PARAMS,
        noise_f_grid=noise_f_grid,
        model_keys=model_keys,
        noise_type=NOISE_TYPE,
        modulation_format=ARGS.modulation,
        output_path=report_path,
    )
    sys.exit(0)

app = Dash(__name__)
app.index_string = app.index_string.replace("</body>", "<script>" + _LEGEND_SYNC_JS + "</script></body>")

step = max(1, len(VALID_INDICES) // 10)
if NOISE_TYPE == "with_signal":
    # VALID_INDICES contains global channel indices directly
    _get_ch_idx = lambda vi: vi
else:
    # VALID_INDICES contains quantum-local indices; map via base_quantum_indices_list
    _get_ch_idx = lambda vi: base_quantum_indices_list[vi]
slider_marks = {
    i: {"label": _display_channel_label(_get_ch_idx(VALID_INDICES[i])), "style": {"font-size": "9px"}}
    for i in range(0, len(VALID_INDICES), step)
}
if (len(VALID_INDICES) - 1) not in slider_marks:
    last = len(VALID_INDICES) - 1
    slider_marks[last] = {"label": _display_channel_label(_get_ch_idx(VALID_INDICES[last])), "style": {"font-size": "9px"}}

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
                    min=0,
                    max=len(PRECOMPUTE_POWER_LEVELS) - 1,
                    step=1,
                    value=3,  # 0 dBm is index 3
                    marks={i: f"{int(p)}" for i, p in enumerate(PRECOMPUTE_POWER_LEVELS)},
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
    if NOISE_TYPE == "with_signal":
        ch_idx = VALID_INDICES[selection_idx]  # VALID_INDICES contains global channel indices
    else:
        q_local = VALID_INDICES[selection_idx]
        ch_idx = base_quantum_indices_list[q_local]  # 全局信道索引用于显示
    freq_hz = WDM_PARAMS["start_freq"] + (ch_idx - WDM_PARAMS["start_channel"]) * WDM_PARAMS["channel_spacing"]
    wl_nm = 299792458.0 / freq_hz * 1e9
    return f"Selected: {_display_channel_label(ch_idx)} | f = {freq_hz / 1e12:.4f} THz | lambda ~ {wl_nm:.2f} nm"


@app.callback(
    Output("power-display", "children"),
    Output("length-graph", "figure"),
    Input("power-slider", "value"),
    State("channel-slider", "value"),
    prevent_initial_call=False,
)
def update_power_and_graph(power_idx: int, channel_selection_idx: int) -> tuple[str, go.Figure]:
    power_dbm = PRECOMPUTE_POWER_LEVELS[power_idx]
    set_power_override(power_dbm)
    cache_key = ("len", power_dbm)
    all_by_len = _POWER_CACHE.get(cache_key, _POWER_CACHE.get(("len", 0.0), {}))
    label = f"Classical power: {power_dbm:.1f} dBm (precomputed)"
    global Y_LOG_RANGE, Y_DBM_RANGE
    Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(all_by_len, model_keys)
    ch_idx = VALID_INDICES[channel_selection_idx]
    fig = _make_figure(all_by_len[ch_idx], ch_idx)
    return label, fig


@app.callback(
    Output("length-graph", "figure", allow_duplicate=True),
    Input("channel-slider", "value"),
    State("power-slider", "value"),
    prevent_initial_call=True,
)
def update_graph_on_channel(channel_selection_idx: int, power_idx: int) -> go.Figure:
    if NOISE_TYPE == "with_signal":
        ch_idx = VALID_INDICES[channel_selection_idx]
    else:
        q_local = VALID_INDICES[channel_selection_idx]
        ch_idx = base_quantum_indices_list[q_local]
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
        height=560,
        legend=dict(groupclick="toggleitem"),
        title=f"Noise vs Fiber Length [{NOISE_TYPE}]",
        annotations=[
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text=_build_caption(),
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="center",
                xanchor="center",
                yanchor="top",
            )
        ],
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
