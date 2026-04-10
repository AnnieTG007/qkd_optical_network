"""Dash App 1 - FWM Length Sweep with Quantum Channel Slider.

Run:
    python scripts/plot_fwm_noise_dash_q.py
    Open http://127.0.0.1:8050

Content: choose a quantum channel with the slider and update both subplots together.
  - Left: forward FWM noise vs fiber length (W, log scale)
  - Right: backward FWM noise vs fiber length (W, log scale)

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
    adaptive_log_ticks,
    _resolve_osa_csv,
)


def _sweep_to_json(data: dict) -> dict:
    """Serialize ALL_SWEEP by converting numpy arrays to lists."""
    return {
        str(qk): {
            mk: {"fwd": v["fwd"].tolist(), "bwd": v["bwd"].tolist()}
            for mk, v in qv.items()
        }
        for qk, qv in data.items()
    }


print("=" * 60)
print("FWM Length Sweep Dash - precomputing all quantum channels x fiber lengths")
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

specs = load_model_specs("fwm_noise")
model_keys = list(specs.keys())
fwm_solver = DiscreteFWMSolver()
N_q = len(base_quantum_indices)
N_L = len(LENGTHS_KM)

print(f"  N_quantum={N_q}, N_length={N_L}, N_models={len(model_keys)}")
print(f"  Total work: {N_q} x {N_L} = {N_q * N_L} FWM solves")
print(f"  ({N_L} solves make one full length sweep for a single quantum channel)")

# ALL_SWEEP[q_local_idx][model_key] = {"fwd": np.array(N_L), "bwd": np.array(N_L)}
ALL_SWEEP: dict[int, dict[str, dict[str, np.ndarray]]] = {}
for qk in range(N_q):
    ALL_SWEEP[qk] = {mk: {"fwd": np.zeros(N_L), "bwd": np.zeros(N_L)} for mk in model_keys}

for q_local_idx, q_global_idx in enumerate(base_quantum_indices):
    single_q_config = _build_wdm_config([q_global_idx])

    for model_key, spec in specs.items():
        grid = _build_model_grid(model_key, spec, single_q_config, noise_f_grid, osa_csv_path)
        for Li, L_km in enumerate(LENGTHS_KM):
            fp = dict(FIBER_PARAMS)
            fp["L_km"] = float(L_km)
            fiber = Fiber(FiberConfig(**fp))

            if spec["continuous"]:
                fwd = fwm_solver.compute_forward_conti(fiber, grid, noise_f_grid)
                bwd = fwm_solver.compute_backward_conti(fiber, grid, noise_f_grid)
            else:
                fwd = fwm_solver.compute_forward(fiber, grid)
                bwd = fwm_solver.compute_backward(fiber, grid)

            ALL_SWEEP[q_local_idx][model_key]["fwd"][Li] = float(fwd[0])
            ALL_SWEEP[q_local_idx][model_key]["bwd"][Li] = float(bwd[0])

    if (q_local_idx + 1) % 5 == 0 or q_local_idx == N_q - 1:
        elapsed = time.time() - t0
        rate = (q_local_idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (N_q - q_local_idx - 1) / rate if rate > 0 else 0
        print(
            f"  Progress: {q_local_idx+1}/{N_q} quantum channels "
            f"({100*(q_local_idx+1)/N_q:.0f}%) | "
            f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s"
        )

VALID_Q_INDICES = []
for qk in range(N_q):
    has_nonzero = any(
        ALL_SWEEP[qk][mk]["fwd"][Li] > 0
        for mk in model_keys
        for Li in range(N_L)
    )
    if has_nonzero:
        VALID_Q_INDICES.append(qk)

if not VALID_Q_INDICES:
    raise RuntimeError(
        "No valid quantum channel indices with non-zero forward FWM noise were found."
    )

ALL_SWEEP_JSON = _sweep_to_json(ALL_SWEEP)

elapsed_total = time.time() - t0
print(f"\nPrecomputation complete. Total elapsed: {elapsed_total:.1f}s")
print(f"Valid quantum channels: {len(VALID_Q_INDICES)}/{N_q}")
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
            f"Sim Parameters  |  Fiber L = {FIBER_PARAMS['L_km']:.0f} km  |  "
            f"N_classical = {n_classical}  |  "
            f"Spacing = {WDM_PARAMS['channel_spacing'] / 1e9:.0f} GHz  |  "
            f"P0 = {WDM_PARAMS['P0'] * 1e3:.0f} mW"
        ),
        "  |  ".join(ch_lines),
    ]
    return "\n".join(lines)


N_valid_q = len(VALID_Q_INDICES)
_n_marks = max(1, N_valid_q // 10)
slider_marks = {
    i: {
        "label": f"Ch{base_quantum_indices[VALID_Q_INDICES[i]]}",
        "style": {"font-size": "8px", "white-space": "nowrap"},
    }
    for i in range(0, N_valid_q, _n_marks)
}
if (N_valid_q - 1) not in slider_marks:
    slider_marks[N_valid_q - 1] = {
        "label": f"Ch{base_quantum_indices[VALID_Q_INDICES[N_valid_q - 1]]}",
        "style": {"font-size": "8px"},
    }

app.layout = html.Div(
    [
        html.H2("FWM Length Sweep - Quantum Channel Slider (Pre-computed)"),
        html.P(
            "Choose a quantum channel with the slider. Sweeps update from precomputed data.",
            style=dict(color="gray"),
        ),
        html.Div(
            [
                html.Label("Quantum Channel Index"),
                dcc.Slider(
                    id="q-slider",
                    min=0,
                    max=N_valid_q - 1,
                    step=1,
                    value=N_valid_q // 2,
                    marks=slider_marks,
                ),
            ],
            style=dict(width="90%", padding="10px"),
        ),
        html.Div(
            id="q-display",
            style=dict(padding="5px 10px", fontFamily="Courier New", fontSize="13px"),
        ),
        dcc.Store(id="sweep-store", data=ALL_SWEEP_JSON),
        dcc.Graph(id="length-graph"),
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
    Output("q-display", "children"),
    Input("q-slider", "value"),
    Input("sweep-store", "data"),
)
def update_display(q_idx: int, store_data: dict) -> str:
    _ = store_data
    ch_global = base_quantum_indices[VALID_Q_INDICES[q_idx]]
    freq_hz = WDM_PARAMS["start_freq"] + ch_global * WDM_PARAMS["channel_spacing"]
    wl_nm = 299792458 / freq_hz * 1e9
    return (
        f"Selected: {_display_channel_label(ch_global)} | "
        f"f = {freq_hz / 1e12:.4f} THz | "
        f"lambda ~ {wl_nm:.2f} nm"
    )


@app.callback(
    Output("param-display", "children"),
    Input("q-slider", "value"),
)
def update_param_display(q_idx: int) -> str:
    _ = q_idx
    return _build_param_text()


@app.callback(
    Output("length-graph", "figure"),
    Input("q-slider", "value"),
    Input("sweep-store", "data"),
)
def update_graph(q_idx: int, store_data: dict) -> go.Figure:
    sweep = store_data[str(VALID_Q_INDICES[q_idx])]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Forward FWM Noise (W, Log Scale)",
            "Backward FWM Noise (W, Log Scale)",
        ),
    )

    L = LENGTHS_KM

    for model_key, spec in specs.items():
        d = sweep.get(model_key)
        if d is None:
            continue

        fwd_arr = np.array(d["fwd"], dtype=np.float64)
        bwd_arr = np.array(d["bwd"], dtype=np.float64)

        for col_idx, arr in enumerate([fwd_arr, bwd_arr], 1):
            mask = arr > 0
            if not np.any(mask):
                continue

            ht = f"L=%{{x:.1f}} km<br>P=%{{y:.3e}} W<extra>{spec['label']}</extra>"
            fig.add_trace(
                go.Scatter(
                    x=L[mask],
                    y=arr[mask],
                    mode="lines+markers",
                    line=dict(color=spec["color"], width=2.0),
                    marker=dict(size=6, color=spec["color"]),
                    name=spec["label"],
                    legendgroup=model_key,
                    showlegend=(col_idx == 1),
                    hovertemplate=ht,
                ),
                row=1,
                col=col_idx,
            )

    all_positive: list[float] = []
    for qk_str in store_data.keys():
        sweep_at_q = store_data.get(qk_str, {})
        for mk in model_keys:
            d = sweep_at_q.get(mk)
            if d is None:
                continue
            fwd_vals = np.array(d["fwd"], dtype=np.float64)
            bwd_vals = np.array(d["bwd"], dtype=np.float64)
            all_positive.extend(fwd_vals[fwd_vals > 0].tolist())
            all_positive.extend(bwd_vals[bwd_vals > 0].tolist())

    if all_positive:
        y_bot_log = round(np.log10(min(all_positive)) - 0.3, 1)
        y_top_log = round(np.log10(max(all_positive)) + 0.3, 1)
    else:
        y_bot_log, y_top_log = -15.5, -4.5

    for col in [1, 2]:
        fig.update_xaxes(
            title_text="Fiber Length [km]",
            type="log",
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text="Noise Power [W]",
            type="log",
            range=[y_bot_log, y_top_log],
            **adaptive_log_ticks(y_bot_log, y_top_log),
            showgrid=True,
            row=1,
            col=col,
        )

    fig.update_layout(
        title="FWM Noise vs Fiber Length - Forward (left) / Backward (right)",
        template="plotly_white",
        width=1500,
        height=500,
        legend=dict(groupclick="toggleitem"),
        uirevision="fixed",
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
