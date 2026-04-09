"""Dash App 2 — FWM Noise PSD Comparison with Fiber Length Slider.

运行：
    python scripts/plot_fwm_noise_dash_l.py
    浏览器打开 http://127.0.0.1:8051

内容：滑条选择光纤长度（1-100 km），双子图显示：
  - 左：FWM 噪声 PSD（W 对数坐标）
  - 右：FWM 噪声 PSD（dBm 线性坐标）

所有数据在启动时预计算完毕，滑条仅做索引（无延迟）。

依赖：pip install dash plotly numpy pandas
"""

from __future__ import annotations

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


def _psd_to_json(data: dict) -> dict:
    """将 ALL_PSD 字典（numpy array → list）序列化为 JSON。"""
    return {
        str(Li): {mk: v[mk].tolist() for mk in v.keys()} for Li, v in data.items()
    }


# ============================================================================
# 预计算（启动时一次性完成）
# ============================================================================

print("=" * 60)
print("FWM PSD vs Length Dash — 预计算所有光纤长度 × 所有模型 PSD")
print("=" * 60)
t0 = time.time()

osa_csv_path = _resolve_osa_csv()
base_quantum_indices = [
    i for i in range(WDM_PARAMS["N_ch"]) if i not in CLASSICAL_INDICES
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
print(f"  预计总计算量: {N_L} × {len(model_keys)} = {N_L * len(model_keys)} 次 PSD 求解")

# ALL_PSD[Li][model_key] = np.array(N_f,)  — 第 Li 个长度对应的各模型 PSD
ALL_PSD: dict[int, dict[str, np.ndarray]] = {}
for Li in range(N_L):
    ALL_PSD[Li] = {mk: np.zeros(N_f) for mk in model_keys}

for Li, L_km in enumerate(LENGTHS_KM):
    fp = dict(FIBER_PARAMS)
    fp["L_km"] = float(L_km)
    fiber = Fiber(FiberConfig(**fp))

    for model_key, spec in specs.items():
        grid = _build_model_grid(
            model_key, spec, base_config, noise_f_grid, osa_csv_path
        )
        psd = fwm_solver.compute_fwm_spectrum_conti(
            fiber, grid, noise_f_grid, direction="forward"
        )
        ALL_PSD[Li][model_key] = psd.astype(np.float64)

    elapsed = time.time() - t0
    rate = (Li + 1) / elapsed if elapsed > 0 else 0
    remaining = (N_L - Li - 1) / rate if rate > 0 else 0
    print(
        f"  进度: {Li+1}/{N_L} ({100*(Li+1)/N_L:.0f}%) | "
        f"耗时: {elapsed:.0f}s | 预计剩余: {remaining:.0f}s"
    )

ALL_PSD_JSON = _psd_to_json(ALL_PSD)

elapsed_total = time.time() - t0
print(f"\n预计算完成！总耗时: {elapsed_total:.1f}s")
print("启动 Dash 服务...")
print("=" * 60)


# ============================================================================
# Dash 应用
# ============================================================================

app = Dash(__name__)


def _build_param_annotation() -> dict:
    n_classical = len(CLASSICAL_INDICES)
    f_classical = [
        f"  Ch{i}: "
        f"{(WDM_PARAMS['f_center'] + (i - (WDM_PARAMS['N_ch']-1)/2) * WDM_PARAMS['channel_spacing'])/1e12:.4f} THz"
        for i in CLASSICAL_INDICES
    ]
    text = (
        f"Sim Parameters\n"
        f"  (可变) L = 由滑条控制\n"
        f"  N_classical = {n_classical}\n"
        f"  Spacing = {WDM_PARAMS['channel_spacing']/1e9:.0f} GHz\n"
        f"  P0 = {WDM_PARAMS['P0']*1e3:.0f} mW\n"
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


def _to_dBm(v: np.ndarray | float) -> np.ndarray | float:
    v_arr = np.asarray(v, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(v_arr, 1e-30)) + 30.0


# Slider 刻度（每 N_L//5 取一个）
_n_marks = max(1, N_L // 5)
slider_marks = {
    i: {"label": f"{LENGTHS_KM[i]:.0f} km", "style": {"font-size": "10px"}}
    for i in range(0, N_L, _n_marks)
}
if (N_L - 1) not in slider_marks:
    slider_marks[N_L - 1] = {"label": f"{LENGTHS_KM[N_L-1]:.0f} km"}

app.layout = html.Div(
    [
        html.H2("FWM Noise PSD — Fiber Length Slider (Pre-computed)"),
        html.P(
            "滑条选择光纤长度，噪声 PSD 曲线同步更新（启动时预计算完毕，无重新计算）",
            style=dict(color="gray"),
        ),
        # ---- 滑条 ----
        html.Div(
            [
                html.Label("光纤长度 [km] (Fiber Length [km])"),
                dcc.Slider(
                    id="l-slider",
                    min=0,
                    max=N_L - 1,
                    step=1,
                    value=list(LENGTHS_KM).index(50.0),  # 默认 L=50 km
                    marks=slider_marks,
                ),
            ],
            style=dict(width="90%", padding="10px"),
        ),
        # 当前长度信息
        html.Div(
            id="l-display",
            style=dict(padding="5px 10px", fontFamily="Courier New", fontSize="13px"),
        ),
        # 预计算数据
        dcc.Store(id="psd-store", data=ALL_PSD_JSON),
        # 图
        dcc.Graph(id="psd-graph"),
    ],
    style=dict(fontFamily="Arial", padding="20px"),
)


@app.callback(
    Output("l-display", "children"),
    Input("l-slider", "value"),
    Input("psd-store", "data"),
)
def update_display(Li: int, store_data: dict) -> str:
    L_km = float(LENGTHS_KM[Li])
    return f"Selected Fiber Length: L = {L_km:.0f} km"


@app.callback(
    Output("psd-graph", "figure"),
    Input("l-slider", "value"),
    Input("psd-store", "data"),
)
def update_graph(Li: int, store_data: dict) -> go.Figure:
    psd_dict = store_data[str(Li)]  # {model_key: [...]} — PSD array per model

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "FWM Noise Power per Bin (W, Log Scale)",
            "FWM Noise Power per Bin (dBm, Linear Scale)",
        ),
    )

    for model_key, spec in specs.items():
        psd = np.array(psd_dict[model_key], dtype=np.float64)
        power_bin_W = psd * df  # bin 功率 [W]
        power_bin_dBm = _to_dBm(power_bin_W)

        mask = power_bin_W > 0
        f_THz = freq_THz[mask]
        y_W = power_bin_W[mask]
        y_dBm = power_bin_dBm[mask]

        _ht_W = (
            f"f=%{{x:.4f}} THz<br>P=%{{y:.3e}} W<extra>"
            + spec["label"]
            + "</extra>"
        )
        _ht_dBm = (
            f"f=%{{x:.4f}} THz<br>P=%{{y:.2f}} dBm<extra>"
            + spec["label"]
            + "</extra>"
        )

        mode = "markers" if not spec["continuous"] else "lines"
        line_dict = dict(color=spec["color"], width=2.0) if spec["continuous"] else None
        marker_dict = dict(size=6, color=spec["color"], symbol="circle")

        # W 子图
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
        # dBm 子图
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

    # ---- 动态坐标轴范围（基于 ALL 长度数据）----
    all_pmax: list[float] = []
    for Li_str in store_data.keys():
        for mk in store_data[Li_str].keys():
            arr = np.array(store_data[Li_str][mk], dtype=np.float64)
            nonzero = arr[arr > 0]
            if nonzero.size > 0:
                all_pmax.append(float(nonzero.max()))
    if all_pmax:
        y_bot_lin = min(all_pmax) / 1e6
        y_top_lin = max(all_pmax) * 10.0
        y_bot_dBm = _to_dBm(y_bot_lin)
        y_top_dBm = _to_dBm(y_top_lin)
    else:
        y_bot_lin, y_top_lin = 1e-15, 1e-8
        y_bot_dBm, y_top_dBm = -120.0, -60.0

    f_min, f_max = float(freq_THz.min()), float(freq_THz.max())

    for col in [1, 2]:
        fig.update_xaxes(
            title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=col
        )

    fig.update_yaxes(
        title_text="Power per Bin [W]",
        range=[y_bot_lin, y_top_lin],
        tickformat="%.0e",
        exponentformat="none",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Power per Bin [dBm]",
        range=[y_bot_dBm, y_top_dBm],
        tickformat="%.1f",
        exponentformat="none",
        row=1,
        col=2,
    )
    fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(
        title="FWM Noise PSD — W (left) / dBm (right)",
        template="plotly_white",
        width=1500,
        height=500,
        legend=dict(groupclick="toggleitem"),
        annotations=[_build_param_annotation()],
    )
    return fig


if __name__ == "__main__":
    print("Dash running: http://127.0.0.1:8051")
    app.run(debug=False, port=8051)
