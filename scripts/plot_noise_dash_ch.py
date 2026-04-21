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
    export_noise_vs_frequency_xlsx,
    export_simulation_report,
    get_noise_model_keys,
    override_strategy_from_cli,
    precompute_by_channel_all_powers,
    print_compute_device,
    profile_scope,
    set_csv_cache_enabled,
    set_power_override,
    _POWER_CACHE,
)


def _to_dbm(values_w: np.ndarray) -> np.ndarray:
    out = np.full_like(values_w, np.nan, dtype=np.float64)
    mask = values_w > 0
    out[mask] = 10.0 * np.log10(values_w[mask] / 1e-3)
    return out


def _global_ranges(all_data: dict, model_keys: list[str]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute y-axis ranges for the plot.

    Returns (y_log, y_dbm). All data is in [W] (power per bin or channel power).
    For power_per_bin entries, integrates over OSA_RBW_HZ (1 GHz) before dBm conversion.
    """
    OSA_RBW_HZ = 1e9  # OSA reference bandwidth, Hz (same as dash_utils.OSA_RBW_HZ)
    positives: list[float] = []
    for curve_data in all_data.values():
        for model_key in model_keys:
            entry = curve_data.get(model_key, {})
            fwd = np.asarray(entry.get("fwd", []), dtype=np.float64)
            bwd = np.asarray(entry.get("bwd", []), dtype=np.float64)
            y_kind = entry.get("y_kind", "")
            if y_kind == "power_per_bin" and NOISE_TYPE != "only_signal":
                # Integrate PSD over reference bandwidth to get equivalent power [W]
                fwd = fwd * OSA_RBW_HZ
                bwd = bwd * OSA_RBW_HZ
            positives.extend(fwd[fwd >= NOISE_FLOOR_W].tolist())
            positives.extend(bwd[bwd >= NOISE_FLOOR_W].tolist())
    if not positives:
        return (-18.0, -3.0), (-150.0, -30.0)

    positives_arr = np.asarray(positives, dtype=np.float64)
    y_log = (float(np.log10(positives_arr.min()) - 0.3), float(np.log10(positives_arr.max()) + 0.3))
    dbm = _to_dbm(positives_arr)
    y_lin = (float(np.nanmin(dbm) - 3.0), float(np.nanmax(dbm) + 3.0))

    return y_log, y_lin


def _select_profile_lengths(lengths_km: np.ndarray, count: int | None) -> np.ndarray:
    """Return evenly spaced fiber lengths for bounded profiling runs."""
    if count is None or count >= len(lengths_km):
        return np.asarray(lengths_km, dtype=np.float64)
    if count <= 1:
        return np.asarray([float(lengths_km[len(lengths_km) // 2])], dtype=np.float64)
    indices = np.linspace(0, len(lengths_km) - 1, num=count, dtype=int)
    indices = np.unique(indices)
    return np.asarray([float(lengths_km[i]) for i in indices], dtype=np.float64)


parser = argparse.ArgumentParser()
parser.add_argument("--type", default="fwm", choices=["fwm", "sprs", "both", "only_signal", "with_signal"])
parser.add_argument("--modulation", default="16qam", choices=["ook", "16qam"])
parser.add_argument(
    "--profile-only",
    action="store_true",
    help="Run a bounded 0 dBm precompute profile and exit before starting Dash.",
)
parser.add_argument(
    "--profile-length-count",
    type=int,
    default=3,
    help="Number of evenly spaced fiber lengths used with --profile-only.",
)
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

ACTIVE_LENGTHS_KM = _select_profile_lengths(
    LENGTHS_KM,
    ARGS.profile_length_count if ARGS.profile_only else None,
)
if ARGS.profile_only:
    PRECOMPUTE_POWER_LEVELS[:] = [0.0]
    set_csv_cache_enabled(False)

print("=" * 60)
print_compute_device()
if ARGS.profile_only:
    print(
        f"Profiling type={NOISE_TYPE} at 0 dBm with "
        f"{len(ACTIVE_LENGTHS_KM)} fiber length samples: {ACTIVE_LENGTHS_KM.tolist()}"
    )
else:
    print(f"Precomputing ALL power levels for type={NOISE_TYPE}")
t0 = time.perf_counter()

with profile_scope("startup: build config, frequency grid, model specs"):
    osa_csv_path = _resolve_osa_csv()
    # base_quantum_indices: ITU G.694.1 channel numbers (1-based)
    base_quantum_indices = [
        itn
        for itn in range(1, int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 2))
        if itn not in CLASSICAL_INDICES
    ]
    # Quantum channel center frequencies for discrete model x-axis
    quantum_center_freqs = np.array(
        [
            WDM_PARAMS["start_freq"] + (itn - WDM_PARAMS["start_channel"]) * WDM_PARAMS["channel_spacing"]
            for itn in base_quantum_indices
        ],
        dtype=np.float64,
    )
    base_config = _build_wdm_config(base_quantum_indices)
    noise_f_grid = _build_noise_frequency_grid(base_config)
    specs = load_model_specs(f"fwm_noise_{ARGS.modulation}")
    model_keys = get_noise_model_keys(NOISE_TYPE)

# Precompute ALL power levels at startup
ALL_BY_CH, VALID_L_INDICES = precompute_by_channel_all_powers(
    noise_type=NOISE_TYPE,
    specs=specs,
    LENGTHS_KM=ACTIVE_LENGTHS_KM,
    base_config=base_config,
    noise_f_grid=noise_f_grid,
    osa_csv_path=osa_csv_path,
    fiber_params=FIBER_PARAMS,
)
if not VALID_L_INDICES:
    raise RuntimeError(f"No valid lengths found for noise type {NOISE_TYPE!r}")

with profile_scope("startup: global y-axis range"):
    Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(ALL_BY_CH, model_keys)

# Precompute y-axis ranges for all power levels (avoids repeated _global_ranges
# calls on every slider callback — each call iterates all cached data).
with profile_scope("startup: y-axis ranges for all power levels"):
    _POWER_Y_RANGES: dict[float, tuple[tuple[float, float], tuple[float, float]]] = {}
    for p in PRECOMPUTE_POWER_LEVELS:
        data = _POWER_CACHE.get(("ch", p), _POWER_CACHE.get(("ch", 0.0), {}))
        _POWER_Y_RANGES[p] = _global_ranges(data, model_keys)
elapsed = time.perf_counter() - t0
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

if ARGS.profile_only or ARGS.export_only:
    # Export Excel files then exit (--export-only implies --export-excel)
    if ARGS.export_only or ARGS.export_excel:
        out_dir = _PROJECT_ROOT / "data" / "precomputed"
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / "noise_vs_frequency.xlsx"
        export_noise_vs_frequency_xlsx(
            all_by_ch=_POWER_CACHE.get(("ch", 0.0), {}),
            model_keys=model_keys,
            noise_f_grid=noise_f_grid,
            LENGTHS_KM=ACTIVE_LENGTHS_KM,
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

step = max(1, len(VALID_L_INDICES) // 10)
slider_marks = {
    i: {"label": f"{ACTIVE_LENGTHS_KM[VALID_L_INDICES[i]]:.0f}", "style": {"font-size": "9px"}}
    for i in range(0, len(VALID_L_INDICES), step)
}
if (len(VALID_L_INDICES) - 1) not in slider_marks:
    last = len(VALID_L_INDICES) - 1
    slider_marks[last] = {"label": f"{ACTIVE_LENGTHS_KM[VALID_L_INDICES[last]]:.0f}", "style": {"font-size": "9px"}}

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
    return f"Selected fiber length: {ACTIVE_LENGTHS_KM[l_idx]:.1f} km"


@app.callback(
    Output("power-display", "children"),
    Output("channel-graph", "figure"),
    Input("power-slider", "value"),
    State("length-slider", "value"),
    prevent_initial_call=False,
)
def update_power_and_graph(power_idx: int, length_selection_idx: int) -> tuple[str, go.Figure]:
    power_dbm = PRECOMPUTE_POWER_LEVELS[power_idx]
    set_power_override(power_dbm)
    cache_key = ("ch", power_dbm)
    all_by_ch = _POWER_CACHE.get(cache_key, _POWER_CACHE.get(("ch", 0.0), {}))
    valid_l_indices = VALID_L_INDICES
    label = f"Classical power: {power_dbm:.1f} dBm (precomputed)"
    global Y_LOG_RANGE, Y_DBM_RANGE
    _y_range = _POWER_Y_RANGES.get(power_dbm, (Y_LOG_RANGE, Y_DBM_RANGE))
    Y_LOG_RANGE, Y_DBM_RANGE = _y_range
    l_idx = valid_l_indices[length_selection_idx]
    fig = _make_figure(all_by_ch[l_idx], l_idx)
    return label, fig


@app.callback(
    Output("channel-graph", "figure", allow_duplicate=True),
    Input("length-slider", "value"),
    State("power-slider", "value"),
    prevent_initial_call=True,
)
def update_graph_on_length(length_selection_idx: int, power_idx: int) -> go.Figure:
    power_dbm = PRECOMPUTE_POWER_LEVELS[power_idx]
    set_power_override(power_dbm)
    cache_key = ("ch", power_dbm)
    all_by_ch = _POWER_CACHE.get(cache_key, _POWER_CACHE.get(("ch", 0.0), {}))
    global Y_LOG_RANGE, Y_DBM_RANGE
    _y_range = _POWER_Y_RANGES.get(power_dbm, (Y_LOG_RANGE, Y_DBM_RANGE))
    Y_LOG_RANGE, Y_DBM_RANGE = _y_range
    l_idx = VALID_L_INDICES[length_selection_idx]
    fig = _make_figure(all_by_ch[l_idx], l_idx)
    return fig


def _make_figure(sweep: dict, l_idx: int) -> go.Figure:
    if NOISE_TYPE == "only_signal":
        subplot_titles = ("Signal Power [W]", "Signal Power [dBm]")
    else:
        subplot_titles = ("Noise Power [W]", "Noise Power [dBm]")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=subplot_titles,
    )

    for model_key in model_keys:
        spec = specs[model_key]
        entry = sweep.get(model_key, {})
        if not entry or entry.get("fwd") is None or len(entry.get("fwd", [])) == 0:
            continue

        x_data = np.asarray(entry.get("x", []), dtype=np.float64)
        x_thz = x_data / 1e12

        _directions = (
            [("fwd", "solid")]
            if NOISE_TYPE == "only_signal"
            else [("fwd", "solid"), ("bwd", "dot")]
        )
        for direction, dash_style in _directions:
            arr_w = np.asarray(entry.get(direction, []), dtype=np.float64)

            if NOISE_TYPE == "only_signal":
                name = spec["label"]
                legendgroup = model_key
            else:
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
        height=560,
        legend=dict(groupclick="toggleitem"),
        title=f"Noise vs Channel Frequency [{NOISE_TYPE}]"
        + (f" | L = {ACTIVE_LENGTHS_KM[l_idx]:.1f} km" if NOISE_TYPE != "only_signal" else " | G_TX (fiber-length independent)"),
        annotations=[
            dict(
                x=0.5,
                y=-0.13,
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
    print("Dash running: http://127.0.0.1:8051")
    app.run(debug=False, port=8051)
