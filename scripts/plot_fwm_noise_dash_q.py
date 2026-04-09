"""Dash App 1 — FWM Length Sweep with Quantum Channel Slider.

运行：
    python scripts/plot_fwm_noise_dash_q.py
    浏览器打开 http://127.0.0.1:8050

内容：滑条选择量子信道（0-76），双子图显示：
  - 左：前向 FWM 噪声 vs 光纤长度（W 对数坐标）
  - 右：后向 FWM 噪声 vs 光纤长度（W 对数坐标）

所有数据在启动时预计算完毕，滑条仅做索引（无延迟）。

依赖：pip install dash plotly numpy pandas
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# ---- 项目路径 setup ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import dash
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qkd_sim.config.plot_config import load_model_specs
from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import build_wdm_grid
from qkd_sim.physical.noise import DiscreteFWMSolver

# ============================================================================
# 参数
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


# ============================================================================
# 辅助函数
# ============================================================================

def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV in {OSA_CSV_PATH}")
    return csv_files[0]


def _build_noise_frequency_grid(config: WDMConfig) -> np.ndarray:
    half_span = (config.N_ch - 1) / 2 * config.channel_spacing
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = config.f_center - half_span - padding
    f_max = config.f_center + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / NOISE_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_wdm_config(quantum_indices: list[int]) -> WDMConfig:
    return WDMConfig(**WDM_PARAMS, quantum_channel_indices=list(quantum_indices))


def _build_model_grid(
    model_key: str,
    spec: dict,
    base_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
):
    """构建单模型 WDMGrid（内部函数，避免循环引用）。"""
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


def _sweep_to_json(data: dict) -> dict:
    """将 ALL_SWEEP 字典（numpy array → list）序列化为 JSON 兼容格式。"""
    return {
        str(qk): {
            mk: {"fwd": v["fwd"].tolist(), "bwd": v["bwd"].tolist()}
            for mk, v in qv.items()
        }
        for qk, qv in data.items()
    }


# ============================================================================
# 预计算（启动时一次性完成）
# ============================================================================

print("=" * 60)
print("FWM Length Sweep Dash — 预计算所有量子信道 × 光纤长度数据")
print("=" * 60)
t0 = time.time()

osa_csv_path = _resolve_osa_csv()
base_quantum_indices = [
    i for i in range(WDM_PARAMS["N_ch"]) if i not in CLASSICAL_INDICES
]
base_config = _build_wdm_config(base_quantum_indices)
noise_f_grid = _build_noise_frequency_grid(base_config)

specs = load_model_specs("fwm_noise")
model_keys = list(specs.keys())
fwm_solver = DiscreteFWMSolver()
N_q = len(base_quantum_indices)
N_L = len(LENGTHS_KM)

print(f"  N_quantum={N_q}, N_length={N_L}, N_models={len(model_keys)}")
print(f"  预计总计算量: {N_q} × {N_L} = {N_q * N_L} 次 FWM 求解")
print(f"  (每 {N_L} 次构成一次完整的量子信道长度扫描)")

# ALL_SWEEP[q_local_idx][model_key] = {"fwd": np.array(N_L), "bwd": np.array(N_L)}
ALL_SWEEP: dict[int, dict[str, dict[str, np.ndarray]]] = {}
for qk in range(N_q):
    ALL_SWEEP[qk] = {mk: {"fwd": np.zeros(N_L), "bwd": np.zeros(N_L)} for mk in model_keys}

total_computations = N_q * len(model_keys)
done = 0
t_last_print = time.time()

for q_local_idx, q_global_idx in enumerate(base_quantum_indices):
    single_q_config = _build_wdm_config([q_global_idx])

    for model_key, spec in specs.items():
        grid = _build_model_grid(
            model_key, spec, single_q_config, noise_f_grid, osa_csv_path
        )
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

        done += 1

    # 每 5 个量子信道打印一次进度
    if (q_local_idx + 1) % 5 == 0 or q_local_idx == N_q - 1:
        elapsed = time.time() - t0
        rate = (q_local_idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (N_q - q_local_idx - 1) / rate if rate > 0 else 0
        print(
            f"  进度: {q_local_idx+1}/{N_q} 量子信道 "
            f"({100*(q_local_idx+1)/N_q:.0f}%) | "
            f"耗时: {elapsed:.0f}s | 预计剩余: {remaining:.0f}s"
        )

# 序列化（存入 dcc.Store）
ALL_SWEEP_JSON = _sweep_to_json(ALL_SWEEP)

elapsed_total = time.time() - t0
print(f"\n预计算完成！总耗时: {elapsed_total:.1f}s")
print("启动 Dash 服务...")
print("=" * 60)


# ============================================================================
# Dash 应用
# ============================================================================

app = Dash(__name__)


def _build_param_annotation() -> dict:
    """构建仿真参数注释。"""
    n_classical = len(CLASSICAL_INDICES)
    f_classical = [
        f"  Ch{i}: "
        f"{(WDM_PARAMS['f_center'] + (i - (WDM_PARAMS['N_ch']-1)/2) * WDM_PARAMS['channel_spacing'])/1e12:.4f} THz"
        for i in CLASSICAL_INDICES
    ]
    text = (
        f"Sim Parameters\n"
        f"  Fiber L = {FIBER_PARAMS['L_km']:.0f} km\n"
        f"  N_classical = {n_classical}\n"
        f"  Spacing = {WDM_PARAMS['channel_spacing']/1e9:.0f} GHz\n"
        f"  P0 = {WDM_PARAMS['P0']*1e3:.0f} mW ({10*np.log10(WDM_PARAMS['P0']*1e3):.1f} dBm)\n"
        + "\n".join(f_classical)
    )
    return dict(
        text=text,
        align="left",
        showarrow=False,
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=6,
        bgcolor="#f9f9f9",
        font=dict(size=9, family="Courier New"),
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.98,
    )


# Slider 刻度标记（每 N_q//10 个取一个，避免拥挤）
_n_marks = max(1, N_q // 10)
slider_marks = {
    i: {
        "label": f"Ch{base_quantum_indices[i]}",
        "style": {"font-size": "8px", "white-space": "nowrap"},
    }
    for i in range(0, N_q, _n_marks)
}
# 确保最后一个索引有刻度
if (N_q - 1) not in slider_marks:
    slider_marks[N_q - 1] = {
        "label": f"Ch{base_quantum_indices[N_q-1]}",
        "style": {"font-size": "8px"},
    }

app.layout = html.Div(
    [
        html.H2("FWM Length Sweep — Quantum Channel Slider (Pre-computed)"),
        html.P(
            "滑条选择量子信道，前向/后向长度扫描同步更新（启动时预计算完毕，无重新计算）",
            style=dict(color="gray"),
        ),
        # ---- 滑条 ----
        html.Div(
            [
                html.Label("量子信道索引 (Quantum Channel Index)"),
                dcc.Slider(
                    id="q-slider",
                    min=0,
                    max=N_q - 1,
                    step=1,
                    value=N_q // 2,
                    marks=slider_marks,
                ),
            ],
            style=dict(width="90%", padding="10px"),
        ),
        # 当前选择信息
        html.Div(
            id="q-display",
            style=dict(padding="5px 10px", fontFamily="Courier New", fontSize="13px"),
        ),
        # 预计算数据（隐式）
        dcc.Store(id="sweep-store", data=ALL_SWEEP_JSON),
        # 图
        dcc.Graph(id="length-graph"),
    ],
    style=dict(fontFamily="Arial", padding="20px"),
)


@app.callback(
    Output("q-display", "children"),
    Input("q-slider", "value"),
    Input("sweep-store", "data"),
)
def update_display(q_idx: int, store_data: dict) -> str:
    ch_global = base_quantum_indices[q_idx]
    freq_hz = (
        WDM_PARAMS["f_center"]
        + (ch_global - (WDM_PARAMS["N_ch"] - 1) / 2) * WDM_PARAMS["channel_spacing"]
    )
    wl_nm = 299792458 / freq_hz * 1e9
    return (
        f"Selected:  Ch {ch_global}  |  "
        f"f = {freq_hz/1e12:.4f} THz  |  "
        f"λ ≈ {wl_nm:.2f} nm"
    )


@app.callback(
    Output("length-graph", "figure"),
    Input("q-slider", "value"),
    Input("sweep-store", "data"),
)
def update_graph(q_idx: int, store_data: dict) -> go.Figure:
    sweep = store_data[str(q_idx)]  # {model_key: {"fwd": [...], "bwd": [...]}}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Forward FWM Noise (W, Log Scale)",
            "Backward FWM Noise (W, Log Scale)",
        ),
    )

    L = LENGTHS_KM  # shape (N_L=13,)

    for model_key, spec in specs.items():
        d = sweep[model_key]
        fwd_arr = np.array(d["fwd"], dtype=np.float64)
        bwd_arr = np.array(d["bwd"], dtype=np.float64)

        for col_idx, (arr, ytitle) in enumerate(
            [(fwd_arr, "Forward"), (bwd_arr, "Backward")], 1
        ):
            mask = arr > 0
            if not np.any(mask):
                continue
            ht = (
                f"L=%{{x:.1f}} km<br>P=%{{y:.3e}} W<extra>"
                + spec["label"]
                + "</extra>"
            )
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

    # ---- 动态 Y 轴范围（基于 ALL 数据，不只是当前 q_idx）----
    all_vals: list[float] = []
    for qk_str in store_data.keys():
        for mk in store_data[qk_str].keys():
            all_vals.extend(store_data[qk_str][mk]["fwd"])
            all_vals.extend(store_data[qk_str][mk]["bwd"])
    nonzero = [v for v in all_vals if v > 0]
    if nonzero:
        y_bot = min(nonzero) / 10.0
        y_top = max(nonzero) * 10.0
    else:
        y_bot, y_top = 1e-15, 1e-5

    for col in [1, 2]:
        fig.update_xaxes(
            title_text="Fiber Length [km]", type="log", row=1, col=col
        )
        fig.update_yaxes(
            title_text="Noise Power [W]",
            type="log",
            range=[y_bot, y_top],
            tickformat="%.0e",
            exponentformat="none",
            row=1,
            col=col,
        )

    fig.update_layout(
        title="FWM Noise vs Fiber Length — Forward (left) / Backward (right)",
        template="plotly_white",
        width=1500,
        height=500,
        legend=dict(groupclick="toggleitem"),
        annotations=[_build_param_annotation()],
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
