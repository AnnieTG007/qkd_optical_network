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
    DEFAULT_SKR_MODEL_KEY,
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
    _init_skr_model_registry,
    _resolve_osa_csv,
    _to_dbm,
    adaptive_linear_ticks,
    adaptive_log_ticks,
    add_strategy_cli_args,
    compute_skr_vs_length,
    ensure_port_free,
    export_noise_vs_length_csv,
    export_noise_vs_length_xlsx,
    export_simulation_report,
    get_noise_model_keys,
    load_skr_config_for_dash,
    override_strategy_from_cli,
    precompute_by_length_all_powers,
    print_compute_device,
    set_power_override,
    _POWER_CACHE,
)


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
parser.add_argument("--modulation", default="dp-16qam", choices=["ook", "dp-16qam"])
parser.add_argument("--data-rate", type=float, default=200e9,
                    help="Bit rate [bps]. DP-16QAM default: 200e9 (200 Gbps); OOK default: 10.3e9 (10.3 Gbps)")
parser.add_argument(
    "--export-excel",
    action="store_true",
    help="After precomputation, export analysis files (Excel + CSV) and exit.",
)
parser.add_argument(
    "--export-only",
    action="store_true",
    help="Shorthand for --export-excel.",
)
parser.add_argument("--skr-profile", choices=["custom", "reference"], default="custom",
                    help="SKR config profile to use for SKR subplots and CSV export.")
parser.add_argument("--active-threshold-db", type=float, default=-50.0,
                    help="FWM active frequency bin threshold [dB]. Default: -50.0.")
add_strategy_cli_args(parser)
ARGS = parser.parse_args()
NOISE_TYPE = ARGS.type
import scripts.dash_utils as _du
_du.MODULATION_FORMAT = ARGS.modulation.upper()
_du.ACTIVE_THRESHOLD_DB = ARGS.active_threshold_db

# SKR zero-value sentinel: replaces ≤0 SKR on log-scale plots
_SKR_ZERO_SENTINEL = 1e-15

# SKR model override
if ARGS.skr_model is not None:
    _du.DEFAULT_SKR_MODEL_KEY = ARGS.skr_model

_du.WDM_PARAMS["data_rate_bps"] = ARGS.data_rate

# Fail fast if a stale Dash instance is still holding 8050 — otherwise app.run
# would OSError after the full precompute and the browser would keep reading
# the zombie. --export-only never reaches app.run, so skip.
if not ARGS.export_only:
    ensure_port_free(8050)

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

osa_csv_path, osa_center_freq_hz = _resolve_osa_csv(ARGS.modulation)
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
    osa_center_freq_hz=osa_center_freq_hz,
)
if not VALID_INDICES:
    raise RuntimeError(f"No valid channels found for noise type {NOISE_TYPE!r}")

Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(ALL_BY_LEN, model_keys)

# --- SKR: load config and precompute cache ---
_FIBER_CFG, _SKR_CFG = load_skr_config_for_dash(ARGS.skr_profile)
# _SKR_MODEL_KEYS kept for cache compatibility; display uses DEFAULT_SKR_MODEL_KEY only
_SKR_MODEL_KEYS = list(_init_skr_model_registry().keys())

_SKR_CACHE_LEN: dict[float, dict[int, dict]] = {}  # power_dbm -> ch_idx -> skr_result


def _build_len_skr_cache(power_dbm: float, sweep_at_ch: dict, ch_idx: int) -> dict:
    """Build SKR cache for a single (power, channel) combo."""
    return compute_skr_vs_length(sweep_at_ch, ch_idx, LENGTHS_KM, _FIBER_CFG, _SKR_CFG,
                               optimize=_SKR_CFG.optimize_params,
                               model_keys=[DEFAULT_SKR_MODEL_KEY])


print("[SKR] Building SKR cache for all power levels and channels...")
if NOISE_TYPE == "with_signal":
    for p in PRECOMPUTE_POWER_LEVELS:
        cache_key = ("len", p)
        data = _POWER_CACHE.get(cache_key, _POWER_CACHE.get(("len", 0.0), {}))
        _SKR_CACHE_LEN[p] = {}
        for ch_idx in sorted(data.keys()):
            _SKR_CACHE_LEN[p][ch_idx] = _build_len_skr_cache(p, data[ch_idx], ch_idx)

elapsed = time.time() - t0
print(f"Precompute done in {elapsed:.1f}s. Valid selections: {len(VALID_INDICES)}")

_CURRENT_POWER_DBM: float = 0.0

if ARGS.export_only or ARGS.export_excel:
    xlsx_dir = _PROJECT_ROOT / "data" / "precomputed"
    xlsx_dir.mkdir(parents=True, exist_ok=True)
    export_dir = _PROJECT_ROOT / "data" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if NOISE_TYPE in ("with_signal", "only_signal"):
        channel_index_lookup = {
            outer_idx: int(WDM_PARAMS["start_channel"] + outer_idx)
            for outer_idx in _POWER_CACHE.get(("len", 0.0), {})
        }
    else:
        channel_index_lookup = {
            q_local_idx: int(base_quantum_indices_list[q_local_idx])
            for q_local_idx in _POWER_CACHE.get(("len", 0.0), {})
        }

    power_exports = {
        float(p): _POWER_CACHE.get(("len", float(p)), _POWER_CACHE.get(("len", 0.0), {}))
        for p in PRECOMPUTE_POWER_LEVELS
    }
    export_noise_vs_length_csv(
        all_by_len_by_power=power_exports,
        model_keys=model_keys,
        specs=specs,
        LENGTHS_KM=LENGTHS_KM,
        channel_index_lookup=channel_index_lookup,
        noise_type=NOISE_TYPE,
        modulation_format=ARGS.modulation,
        output_dir=export_dir,
        fiber_cfg=_FIBER_CFG,
        skr_cfg=_SKR_CFG,
    )

    # quantum_center_freqs: same as used for base_config
    quantum_center_freqs = np.array(
        [
            WDM_PARAMS["start_freq"] + (itn - WDM_PARAMS["start_channel"]) * WDM_PARAMS["channel_spacing"]
            for itn in base_quantum_indices_list
        ],
        dtype=np.float64,
    )
    xlsx_path = xlsx_dir / "noise_vs_length.xlsx"
    export_noise_vs_length_xlsx(
        all_by_len=_POWER_CACHE.get(("len", 0.0), {}),
        model_keys=model_keys,
        LENGTHS_KM=LENGTHS_KM,
        quantum_center_freqs=quantum_center_freqs,
        output_path=xlsx_path,
    )
    report_path = xlsx_dir / "simulation_report.txt"
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
                    value=0,
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


def _get_ch_idx_from_selection(selection_idx: int) -> int:
    if NOISE_TYPE == "with_signal":
        return VALID_INDICES[selection_idx]
    return base_quantum_indices_list[VALID_INDICES[selection_idx]]


def _compute_skr_y_ranges_len(skr_cache_ch: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute log y-axis ranges for SKR [bit/pulse] and SKR [bps] for len script."""
    bps_vals: list[float] = []
    bpp_vals: list[float] = []
    for mkey_data in skr_cache_ch.values():
        for direction in ("fwd", "bwd"):
            bps_arr, bpp_arr, _ = mkey_data.get(direction, (np.array([]), np.array([]), np.array([])))
            if len(bps_arr) > 0:
                valid = bps_arr[bps_arr > _SKR_ZERO_SENTINEL]
                if len(valid) > 0:
                    bps_vals.extend(valid.tolist())
                elif np.any(bps_arr <= 0):
                    bps_vals.append(_SKR_ZERO_SENTINEL)
            if len(bpp_arr) > 0:
                valid = bpp_arr[bpp_arr > _SKR_ZERO_SENTINEL]
                if len(valid) > 0:
                    bpp_vals.extend(valid.tolist())
                elif np.any(bpp_arr <= 0):
                    bpp_vals.append(_SKR_ZERO_SENTINEL)
    if not bps_vals:
        return (-12.0, 0.0), (-15.0, 0.0)
    bps_arr = np.asarray(bps_vals)
    bpp_arr = np.asarray(bpp_vals)
    return (float(np.log10(bpp_arr.min()) - 0.3), float(np.log10(bpp_arr.max()) + 0.3)), \
           (float(np.log10(bps_arr.min()) - 0.3), float(np.log10(bps_arr.max()) + 0.3))


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
    global Y_LOG_RANGE, Y_DBM_RANGE, _CURRENT_POWER_DBM
    Y_LOG_RANGE, Y_DBM_RANGE = _global_ranges(all_by_len, model_keys)
    _CURRENT_POWER_DBM = power_dbm
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
    power_dbm = PRECOMPUTE_POWER_LEVELS[power_idx]
    set_power_override(power_dbm)
    global _CURRENT_POWER_DBM
    _CURRENT_POWER_DBM = power_dbm
    if NOISE_TYPE == "with_signal":
        ch_idx = VALID_INDICES[channel_selection_idx]
    else:
        q_local = VALID_INDICES[channel_selection_idx]
        ch_idx = base_quantum_indices_list[q_local]
    fig = _make_figure(ALL_BY_LEN[ch_idx], ch_idx)
    return fig


def _make_figure(sweep: dict, ch_idx: int) -> go.Figure:
    has_skr = NOISE_TYPE == "with_signal"
    if has_skr:
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Noise Power [W]", "Noise Power [dBm]", "SKR [bit/pulse]", "SKR [bps]"),
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=True,
            subplot_titles=("Noise Power [W]", "Noise Power [dBm]"),
        )

    # --- Noise traces (row 1) ---
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
                    x=LENGTHS_KM[mask], y=arr_w[mask],
                    mode="lines+markers",
                    line=dict(color=spec["color"], width=2.0, dash=dash_style),
                    marker=dict(size=6, color=spec["color"]),
                    name=name, legendgroup=legendgroup, showlegend=True,
                    hovertemplate=hover_w,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=LENGTHS_KM[mask], y=_to_dbm(arr_w[mask]),
                    mode="lines+markers",
                    line=dict(color=spec["color"], width=2.0, dash=dash_style),
                    marker=dict(size=6, color=spec["color"]),
                    name=name, legendgroup=legendgroup, showlegend=False,
                    hovertemplate=hover_dbm,
                ),
                row=1, col=2,
            )

    # --- SKR traces (row 2) ---
    if has_skr:
        global _CURRENT_POWER_DBM
        pwr = _CURRENT_POWER_DBM
        skr_cache_ch = _SKR_CACHE_LEN.get(pwr, {}).get(ch_idx, {})

        skr_model_key = DEFAULT_SKR_MODEL_KEY
        for model_key in model_keys:
            spec = specs[model_key]
            if model_key not in skr_cache_ch:
                continue
            for direction in ("fwd", "bwd"):
                skr_data = skr_cache_ch.get(model_key, {}).get(direction)
                if skr_data is None:
                    continue
                bps_arr, bpp_arr, _qber_arr = skr_data
                if len(bps_arr) == 0:
                    continue

                skr_name = f"{spec['label']} ({direction})"
                skr_lg = f"{model_key}-{direction}-skr"

                # Use the masked lengths for x-axis (same mask as noise)
                fwd_w = np.asarray(sweep[model_key].get("fwd", []), dtype=np.float64)
                x_mask = fwd_w >= NOISE_FLOOR_W
                x_skr = LENGTHS_KM[x_mask]
                if len(x_skr) != len(bps_arr):
                    x_skr = LENGTHS_KM[:len(bps_arr)]

                _bpp_pos = bpp_arr > 0
                bpp_plot = np.where(_bpp_pos, bpp_arr, _SKR_ZERO_SENTINEL)
                bpp_cd = np.where(_bpp_pos, bpp_arr, 0.0)
                _bps_pos = bps_arr > 0
                bps_plot = np.where(_bps_pos, bps_arr, _SKR_ZERO_SENTINEL)
                bps_cd = np.where(_bps_pos, bps_arr, 0.0)

                fig.add_trace(
                    go.Scatter(x=x_skr, y=bpp_plot, mode="lines+markers",
                               customdata=bpp_cd,
                               line=dict(color=spec["color"], width=1.5, dash="solid"),
                               marker=dict(size=4, color=spec["color"]),
                               connectgaps=False, name=skr_name, legendgroup=skr_lg,
                               showlegend=True,
                               hovertemplate=f"L=%{{x:.1f}} km<br>SKR=%{{customdata:.3e}} bit/pulse<extra>{skr_name}</extra>"),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=x_skr, y=bps_plot, mode="lines+markers",
                               customdata=bps_cd,
                               line=dict(color=spec["color"], width=1.5, dash="solid"),
                               marker=dict(size=4, color=spec["color"]),
                               connectgaps=False, name=skr_name, legendgroup=skr_lg,
                               showlegend=False,
                               hovertemplate=f"L=%{{x:.1f}} km<br>SKR=%{{customdata:.3e}} bps<extra>{skr_name}</extra>"),
                    row=2, col=2,
                )

        skr_bpp_range, skr_bps_range = _compute_skr_y_ranges_len(skr_cache_ch)
    else:
        skr_bpp_range, skr_bps_range = (-15.0, 0.0), (-12.0, 0.0)

    # --- Axes ---
    fig.update_xaxes(title_text="Fiber Length [km]", row=1, col=1)
    fig.update_xaxes(title_text="Fiber Length [km]", row=1, col=2)
    if has_skr:
        fig.update_xaxes(title_text="Fiber Length [km]", row=2, col=1)
        fig.update_xaxes(title_text="Fiber Length [km]", row=2, col=2)

    fig.update_yaxes(
        title_text="Power [W]", type="log", range=list(Y_LOG_RANGE),
        showgrid=True, row=1, col=1, **adaptive_log_ticks(*Y_LOG_RANGE),
    )
    fig.update_yaxes(
        title_text="Power [dBm]", type="linear", range=list(Y_DBM_RANGE),
        showgrid=True, row=1, col=2, **adaptive_linear_ticks(*Y_DBM_RANGE),
    )

    if has_skr:
        fig.update_yaxes(
            title_text="SKR [bit/pulse]", type="log", range=list(skr_bpp_range),
            showgrid=True, row=2, col=1, **adaptive_log_ticks(*skr_bpp_range),
        )
        fig.update_yaxes(
            title_text="SKR [bps]", type="log", range=list(skr_bps_range),
            showgrid=True, row=2, col=2, **adaptive_log_ticks(*skr_bps_range),
        )

    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=1000 if has_skr else 550,
        legend=dict(groupclick="toggleitem"),
        title=f"Noise vs Fiber Length [{NOISE_TYPE}]",
        annotations=[
            dict(
                x=0.5, y=-0.05, xref="paper", yref="paper",
                text=_build_caption(), showarrow=False,
                font=dict(size=10, color="gray"), align="center",
                xanchor="center", yanchor="top",
            )
        ],
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
