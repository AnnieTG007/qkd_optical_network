"""FWM 噪声功率谱对比绘图脚本（交互式 Plotly）。

比较以下信号模型的 FWM 噪声特性：
  - Discrete: delta 近似（stem markers）
  - Raised Cosine β=0.2（连续曲线）
  - OSA（连续曲线）

包含三个图：
  1. FWM 噪声功率谱对比（双子图：W 对数 + dBm 线性）
  2. 前向 FWM 噪声 vs 光纤长度
  3. 后向 FWM 噪声 vs 光纤长度

输出到: outputs/phase4_N61_C3/FWM_Noise/

纵轴量纲（与 signal_tx 保持一致）：
  - 离散模型 stem 高度 = 该量子信道的积分 FWM 噪声功率 [W]
  - 连续模型曲线高度 = G_fwm(f) × Δf [W]（bin 功率），量纲统一为 [W]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Plotly 用于交互式绘图（支持点击图例动态开关各模型曲线）
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ---- 项目路径 setup ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.plot_config import get_color, load_model_specs
from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import SpectrumType, WDMGrid, build_wdm_grid
from qkd_sim.physical.noise import DiscreteFWMSolver

# ============================================================================
# 配置参数
# ============================================================================

WDM_PARAMS = dict(
    start_freq=190.1e12,
    start_channel=1,
    end_channel=61,
    channel_spacing=100e9,
    B_s=32e9,
    P0=1e-3,
    beta_rolloff=0.2,
)

# 泵浦（经典信道）indices：39, 40, 41（居中）
CLASSICAL_INDICES = [38, 39, 40]

# 频率网格分辨率
FREQ_GRID_RESOLUTION_HZ = 0.1e9      # 信号 grid（连续模型 PSD）
NOISE_GRID_RESOLUTION_HZ = 5e9      # 噪声 PSD grid（避免 O(N_f×N_c²) 性能问题）
FREQ_GRID_PADDING_FACTOR = 1.5

# 长度扫描数组 [km]
LENGTHS_KM = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# OSA
OSA_RBW_HZ = 1.0e9
OSA_CSV_PATH = _PROJECT_ROOT / "data" / "osa"


# ---- 光纤参数（defaults/fiber_para/fiber_smf.yaml）----
FIBER_PARAMS = dict(
    alpha_dB_per_km=0.2,
    gamma_per_W_km=1.3,
    D_ps_nm_km=17.0,
    D_slope_ps_nm2_km=0.056,
    L_km=50.0,          # 默认长度，扫描时会覆盖
    A_eff=80e-12,       # 80 μm² → 8e-11 m²
    rayleigh_coeff=4.8e-8,
    T_kelvin=300.0,
)


# ============================================================================
# 辅助函数
# ============================================================================

def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV files found in {OSA_CSV_PATH}")
    return csv_files[0]


def _n_channels_from_params() -> int:
    return int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 1)


def _build_noise_frequency_grid(config: WDMConfig) -> np.ndarray:
    """构建噪声 PSD 绘图的频率网格（粗分辨率）。"""
    half_span = (config.end_channel - config.start_channel) / 2.0 * config.channel_spacing
    center_freq = config.start_freq + half_span
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = center_freq - half_span - padding
    f_max = center_freq + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / NOISE_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_signal_frequency_grid(config: WDMConfig) -> np.ndarray:
    """构建信号 PSD 的精细频率网格（连续模型积分需要）。"""
    half_span = (config.end_channel - config.start_channel) / 2.0 * config.channel_spacing
    center_freq = config.start_freq + half_span
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = center_freq - half_span - padding
    f_max = center_freq + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / FREQ_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_model_wdm_grid(
    model_key: str,
    spec: dict,
    base_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
) -> WDMGrid:
    """为指定信号模型构建独立的 WDMGrid（支持 per-model beta_rolloff）。

    Parameters
    ----------
    model_key : str
        模型键名
    spec : dict
        从 load_model_specs() 获取的模型规格字典（已含 spectrum_type/color/continuous 等）
    base_config : WDMConfig
        基础 WDM 配置
    f_grid : ndarray
        频率网格
    osa_csv_path : Path
        OSA CSV 文件路径
    """
    if spec["beta_rolloff"] is not None:
        model_config = WDMConfig(
            start_freq=base_config.start_freq,
            start_channel=base_config.start_channel,
            end_channel=base_config.end_channel,
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


def _to_dBm_for_plotly(v: np.ndarray | float) -> np.ndarray | float:
    """将功率 [W] 转换为 dBm（向量化，支持数组输入）。"""
    v_arr = np.asarray(v, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(v_arr, 1e-30)) + 30.0


def _display_channel_label(channel_index: int) -> str:
    return f"C{channel_index + WDM_PARAMS['start_channel']}"


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


# ============================================================================
# 数据计算
# ============================================================================

@dataclass
class FWMPSDResult:
    """FWM 噪声 PSD 计算结果（单模型）。"""
    key: str
    label: str
    color: str
    f_hz: np.ndarray          # 频率网格 [Hz]
    fwm_psd_W_per_Hz: np.ndarray  # G_fwm(f) [W/Hz]，shape (N_f,)
    # 离散模型：每信道的积分噪声功率 [W]，shape (N_q,)
    fwm_channel_power_W: np.ndarray | None = None


@dataclass
class FWMLengthSweepResult:
    """FWM 长度扫描结果（单模型）。"""
    key: str
    label: str
    color: str
    lengths_km: np.ndarray
    fwd_noise_W: np.ndarray   # shape (N_lengths,)
    bwd_noise_W: np.ndarray   # shape (N_lengths,)


def compute_fwm_psd_results(
    base_config: WDMConfig,
    osa_csv_path: Path,
) -> list[FWMPSDResult]:
    """??????????? FWM ??? PSD??
    ?????????????????????????????????
    ????????ompute_fwm_spectrum_conti ????????PSD [W/Hz]
    """
    fwm_solver = DiscreteFWMSolver()
    f_grid = _build_noise_frequency_grid(base_config)

    fp = dict(FIBER_PARAMS)
    fiber = Fiber(FiberConfig(**fp))

    results: list[FWMPSDResult] = []
    specs = load_model_specs("fwm_noise")

    for model_key, spec in specs.items():
        grid = _build_model_wdm_grid(model_key, spec, base_config, f_grid, osa_csv_path)
        q_chs = grid.get_quantum_channels()
        f_q_hz = np.array([ch.f_center for ch in q_chs], dtype=np.float64)

        if spec["continuous"]:
            fwm_psd = fwm_solver.compute_fwm_spectrum_conti(
                fiber, grid, f_grid, direction="forward"
            )
            fwm_channel_power = None
        else:
            fwm_channel_power = fwm_solver.compute_forward(fiber, grid)
            fwm_psd = _build_discrete_fwm_psd(f_grid, f_q_hz, fwm_channel_power)

        results.append(FWMPSDResult(
            key=model_key,
            label=spec["label"],
            color=spec["color"] if "color" in spec else get_model_color(model_key),
            f_hz=f_grid,
            fwm_psd_W_per_Hz=fwm_psd,
            fwm_channel_power_W=fwm_channel_power,
        ))

    return results


def compute_fwm_length_sweep_results(
    base_config: WDMConfig,
    osa_csv_path: Path,
) -> list[FWMLengthSweepResult]:
    """计算所有模型的 FWM 噪声 vs 光纤长度。

    在中心量子信道（index 40, f≈193.4 THz）处提取噪声功率。
    同时计算前向和后向。
    """
    fwm_solver = DiscreteFWMSolver()
    f_grid = _build_noise_frequency_grid(base_config)

    results: list[FWMLengthSweepResult] = []
    specs = load_model_specs("fwm_noise")

    for model_key, spec in specs.items():
        grid = _build_model_wdm_grid(model_key, spec, base_config, f_grid, osa_csv_path)

        # 找到中心量子信道索引
        q_chs = grid.get_quantum_channels()
        center_idx = len(q_chs) // 2  # 居中量子信道

        fwd_arr = np.zeros_like(LENGTHS_KM, dtype=np.float64)
        bwd_arr = np.zeros_like(LENGTHS_KM, dtype=np.float64)

        for i, L_km in enumerate(LENGTHS_KM):
            fp = dict(FIBER_PARAMS)
            fp["L_km"] = float(L_km)
            fiber = Fiber(FiberConfig(**fp))

            if spec["continuous"]:
                fwd = fwm_solver.compute_forward_conti(fiber, grid, f_grid)
                bwd = fwm_solver.compute_backward_conti(fiber, grid, f_grid)
            else:
                fwd = fwm_solver.compute_forward(fiber, grid)
                bwd = fwm_solver.compute_backward(fiber, grid)

            fwd_arr[i] = float(fwd[center_idx])
            bwd_arr[i] = float(bwd[center_idx])

        results.append(FWMLengthSweepResult(
            key=model_key,
            label=spec["label"],
            color=spec["color"] if "color" in spec else get_model_color(model_key),
            lengths_km=LENGTHS_KM,
            fwd_noise_W=fwd_arr,
            bwd_noise_W=bwd_arr,
        ))

    return results


# ============================================================================
# CSV 导出
# ============================================================================

def _export_psd_csv(results: list[FWMPSDResult], out_dir: Path) -> None:
    """导出 FWM PSD 数据到 CSV。"""
    ref_f = results[0].f_hz
    f_THz = ref_f / 1e12
    df = float(np.mean(np.diff(ref_f)))

    header = ["f_THz"]
    for r in results:
        header.append(f"{r.key}_W_per_Hz")
        header.append(f"{r.key}_dBm_per_Hz")

    rows = []
    for i in range(len(ref_f)):
        row = [f"{f_THz[i]:.6f}"]
        for r in results:
            psd_val = r.fwm_psd_W_per_Hz[i]
            row.append(f"{psd_val:.6e}")
            row.append(f"{_to_dBm_for_plotly(psd_val):.3f}")
        rows.append(",".join(row))

    csv_lines = [",".join(header)] + rows
    (out_dir / "fwm_psd_comparison.csv").write_text("\n".join(csv_lines), encoding="utf-8")


def _export_length_csv(results: list[FWMLengthSweepResult], out_dir: Path) -> None:
    """导出 FWM 长度扫描数据到 CSV。"""
    header = ["L_km"]
    for r in results:
        header.append(f"{r.key}_fwd_W")
        header.append(f"{r.key}_bwd_W")

    rows = []
    for i, L in enumerate(results[0].lengths_km):
        row = [f"{L:.1f}"]
        for r in results:
            row.append(f"{r.fwd_noise_W[i]:.6e}")
            row.append(f"{r.bwd_noise_W[i]:.6e}")
        rows.append(",".join(row))

    csv_lines = [",".join(header)] + rows
    (out_dir / "fwm_length_sweep.csv").write_text("\n".join(csv_lines), encoding="utf-8")


# ============================================================================
# Plotly 绘图
# ============================================================================

# ---- 从 plot_signal_tx.py 复制的图例交互 JS（完全一致的行为）----
_LEGEND_SYNC_JS = """
function syncLegendClicks() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;

    // 单击图例项：切换该模型的显示/隐藏
    // 修复：直接用 eventData.curveNumber（Plotly 原生提供），不再依赖 DOM 遍历
    gd.on('plotly_legendclick', function(eventData) {
        var curveNumber = eventData.curveNumber;
        var fullData = gd._fullData || [];
        var clickedGroup = null;

        if (curveNumber !== null && curveNumber !== undefined && fullData[curveNumber]) {
            clickedGroup = fullData[curveNumber].legendgroup;
        }
        if (!clickedGroup) return true;

        var groupOn = false;
        for (var k = 0; k < gd.data.length; k++) {
            if (gd.data[k].legendgroup === clickedGroup && gd.data[k].visible === true) {
                groupOn = true;
                break;
            }
        }

        var newVal = groupOn ? 'legendonly' : true;
        for (var m = 0; m < gd.data.length; m++) {
            if (gd.data[m].legendgroup === clickedGroup) {
                gd.data[m].visible = newVal;
            }
        }
        Plotly.redraw(gd);
        return false;
    });

    // 双击图例项：isolate（隔离）或恢复全部
    gd.on('plotly_legenddoubleclick', function(eventData) {
        var curveNumber = eventData.curveNumber;
        var fullData = gd._fullData || [];
        if (curveNumber === null || curveNumber === undefined || !fullData[curveNumber]) return true;

        var hasHidden = false;
        for (var h = 0; h < gd.data.length; h++) {
            if (gd.data[h].visible === 'legendonly') { hasHidden = true; break; }
        }

        if (hasHidden) {
            for (var ri = 0; ri < gd.data.length; ri++) {
                gd.data[ri].visible = true;
            }
            Plotly.redraw(gd);
        }
        return false;
    });
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', syncLegendClicks);
} else {
    syncLegendClicks();
}
"""


def _build_param_annotation() -> list:
    """构建仿真参数注释文本（用于 Plotly figures 右侧/下侧标注）。"""
    n_classical = len(CLASSICAL_INDICES)
    f_classical = [
        f"  {_display_channel_label(i)}: {(WDM_PARAMS['start_freq'] + i * WDM_PARAMS['channel_spacing']) / 1e12:.4f} THz"
        for i in CLASSICAL_INDICES
    ]
    classical_freqs = "\n".join(f_classical)
    text = (
        f"Sim Parameters\n"
        f"  Fiber L = {FIBER_PARAMS['L_km']:.0f} km\n"
        f"  N_classical = {n_classical}\n"
        f"  Spacing = {WDM_PARAMS['channel_spacing']/1e9:.0f} GHz\n"
        f"  P0 = {WDM_PARAMS['P0']*1e3:.0f} mW ({10*np.log10(WDM_PARAMS['P0']*1e3):.1f} dBm)\n"
        f"{classical_freqs}"
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
        xref="paper", yref="paper",
        x=1.02, y=0.98,
    )


def make_fwm_psd_comparison_figure(results: list[FWMPSDResult]) -> go.Figure:
    """生成 FWM 噪声 PSD 对比图（双子图：W 对数 + dBm 线性）。"""
    if not _PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Run: pip install plotly")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "FWM Noise Power per Bin (W, Log Scale)",
            "FWM Noise Power per Bin (dBm, Linear Scale)",
        ),
        shared_xaxes=False,
    )

    df = float(np.mean(np.diff(results[0].f_hz)))

    for r in results:
        f_THz = r.f_hz / 1e12
        power_bin_W = r.fwm_psd_W_per_Hz * df  # [W]

        # 过滤零功率点
        mask = power_bin_W > 0
        f_plot = f_THz[mask]
        y_lin = power_bin_W[mask]
        y_dbm = _to_dBm_for_plotly(y_lin)

        # 修复 hover：Plotly hovertemplate 需要 %{x} 和 %{y}（前面有 %）
        _ht_W = "f=%{x:.4f} THz<br>P=%{y:.3e} W<extra>" + r.label + "</extra>"
        _ht_dBm = "f=%{x:.4f} THz<br>P=%{y:.2f} dBm<extra>" + r.label + "</extra>"

        if r.key == "discrete":
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_lin,
                    mode="markers",
                    marker=dict(size=6, color=r.color, symbol="circle"),
                    name=r.label,
                    legendgroup=r.key,
                    showlegend=True,
                    hovertemplate=_ht_W,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_dbm,
                    mode="markers",
                    marker=dict(size=6, color=r.color, symbol="circle"),
                    name=r.label,
                    legendgroup=r.key,
                    showlegend=False,
                    hovertemplate=_ht_dBm,
                ),
                row=1, col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_lin,
                    mode="lines",
                    line=dict(color=r.color, width=2.0),
                    name=r.label,
                    legendgroup=r.key,
                    showlegend=True,
                    hovertemplate=_ht_W,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_dbm,
                    mode="lines",
                    line=dict(color=r.color, width=2.0),
                    name=r.label,
                    legendgroup=r.key,
                    showlegend=False,
                    hovertemplate=_ht_dBm,
                ),
                row=1, col=2,
            )

    # ---- 计算动态坐标轴范围 ----
    all_f_nonzero: list[np.ndarray] = []
    for r in results:
        power_bin = r.fwm_psd_W_per_Hz * df
        all_f_nonzero.append((r.f_hz / 1e12)[power_bin > 0])
    if all_f_nonzero:
        all_f = np.concatenate(all_f_nonzero)
        f_min = float(all_f.min()) - 0.1
        f_max = float(all_f.max()) + 0.1
    else:
        f_min, f_max = 191.0, 196.0

    all_pmax: list[float] = []
    for r in results:
        power_bin = r.fwm_psd_W_per_Hz * df
        nonzero = power_bin[power_bin > 0]
        if nonzero.size > 0:
            all_pmax.append(float(nonzero.max()))
    if all_pmax:
        y_bot_lin = min(all_pmax) / 1e6
        y_top_lin = max(all_pmax) * 10.0
        y_bot_dbm = _to_dBm_for_plotly(y_bot_lin)
        y_top_dbm = _to_dBm_for_plotly(y_top_lin)
    else:
        y_bot_lin, y_top_lin = 1e-15, 1e-8
        y_bot_dbm, y_top_dbm = -120.0, -60.0

    fig.update_xaxes(title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=1)
    fig.update_xaxes(title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=2)
    fig.update_yaxes(
        title_text="Power per Bin [W]",
        range=[y_bot_lin, y_top_lin],
        tickformat="%.0e",
        exponentformat="none",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Power per Bin [dBm]",
        range=[y_bot_dbm, y_top_dbm],
        tickformat="%.1f",
        exponentformat="none",
        row=1, col=2,
    )
    fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(
        title=dict(
            text="FWM Noise Power per Bin — Interactive (click legend to toggle, double-click to isolate)",
            x=0.5, xanchor="center",
        ),
        legend=dict(title=dict(text="Signal Model"), groupclick="toggleitem"),
        template="plotly_white",
        width=1500,
        height=500,
        annotations=list(fig.layout.annotations) + [_build_param_annotation()],
    )
    return fig


def _make_fwm_length_sweep_subfigure(
    results: list[FWMLengthSweepResult],
    direction: str,  # "forward" or "backward"
) -> go.Figure:
    """生成 FWM 长度扫描双子图（W 对数 + dBm 线性）。"""
    if not _PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Run: pip install plotly")

    label_map = {"forward": "Forward FWM", "backward": "Backward FWM"}
    noise_attr = "fwd_noise_W" if direction == "forward" else "bwd_noise_W"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{label_map[direction]} Noise Power (W, Log Scale)",
            f"{label_map[direction]} Noise Power (dBm, Linear Scale)",
        ),
        shared_xaxes=False,
    )

    for r in results:
        lengths = r.lengths_km
        noise_W = getattr(r, noise_attr)
        noise_dBm = _to_dBm_for_plotly(noise_W)

        # 过滤零噪声点
        mask = noise_W > 0
        if not np.any(mask):
            continue

        # W 子图（col=1）：对数坐标
        fig.add_trace(
            go.Scatter(
                x=lengths[mask],
                y=noise_W[mask],
                mode="lines+markers",
                line=dict(color=r.color, width=2.0),
                marker=dict(size=6, color=r.color),
                name=r.label,
                legendgroup=r.key,
                showlegend=True,
                hovertemplate=(
                    "L=%{x:.1f} km<br>P=%{y:.3e} W<extra>"
                    + r.label
                    + "</extra>"
                ),
            ),
            row=1, col=1,
        )

        # dBm 子图（col=2）：线性坐标
        fig.add_trace(
            go.Scatter(
                x=lengths[mask],
                y=noise_dBm[mask],
                mode="lines+markers",
                line=dict(color=r.color, width=2.0),
                marker=dict(size=6, color=r.color),
                name=r.label,
                legendgroup=r.key,
                showlegend=False,
                hovertemplate=(
                    "L=%{x:.1f} km<br>P=%{y:.2f} dBm<extra>"
                    + r.label
                    + "</extra>"
                ),
            ),
            row=1, col=2,
        )

    # 动态范围
    all_noise_W: list[np.ndarray] = [getattr(r, noise_attr)[getattr(r, noise_attr) > 0] for r in results]
    all_noise_W = [x for x in all_noise_W if x.size > 0]
    if all_noise_W:
        y_bot_W = min(np.concatenate(all_noise_W)) / 10.0
        y_top_W = max(np.concatenate(all_noise_W)) * 10.0
        y_bot_dBm = _to_dBm_for_plotly(y_bot_W)
        y_top_dBm = _to_dBm_for_plotly(y_top_W)
    else:
        y_bot_W, y_top_W = 1e-15, 1e-5
        y_bot_dBm, y_top_dBm = -150.0, -50.0

    # X 轴：log scale（两列独立）
    fig.update_xaxes(title_text="Fiber Length [km]", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Fiber Length [km]", type="log", row=1, col=2)
    # Y 轴
    fig.update_yaxes(
        title_text="Noise Power [W]",
        range=[y_bot_W, y_top_W],
        type="log",
        tickformat="%.0e",
        exponentformat="none",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Noise Power [dBm]",
        range=[y_bot_dBm, y_top_dBm],
        tickformat="%.1f",
        exponentformat="none",
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"{label_map[direction]} Noise vs Fiber Length — Interactive (click legend to toggle, double-click to isolate)",
            x=0.5, xanchor="center",
        ),
        legend=dict(title=dict(text="Signal Model"), groupclick="toggleitem"),
        template="plotly_white",
        width=1500,
        height=500,
        annotations=list(fig.layout.annotations) + [_build_param_annotation()],
    )
    return fig


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    n_classical = len(CLASSICAL_INDICES)
    n_ch = _n_channels_from_params()
    run_tag = f"phase4_N{n_ch}_C{n_classical}"
    output_dir = _PROJECT_ROOT / "outputs" / run_tag / "FWM_Noise"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # 量子信道 = 所有非 CLASSICAL_INDICES 的信道（避免与经典信道重叠）
    quantum_channel_indices = [i for i in range(int(WDM_PARAMS["end_channel"] - WDM_PARAMS["start_channel"] + 1)) if i not in CLASSICAL_INDICES]
    wdm_config = WDMConfig(**WDM_PARAMS, quantum_channel_indices=quantum_channel_indices)
    osa_csv_path = _resolve_osa_csv()

    # ---- Figure 1: FWM PSD 对比 ----
    print("Computing FWM PSD for all models...")
    psd_results = compute_fwm_psd_results(wdm_config, osa_csv_path)

    print("Generating FWM PSD comparison figure...")
    fig_psd = make_fwm_psd_comparison_figure(psd_results)

    html_path = output_dir / "fwm_psd_comparison.html"
    fig_psd.write_html(str(html_path), post_script=_LEGEND_SYNC_JS, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved: {html_path}")

    png_path = output_dir / "fwm_psd_comparison.png"
    fig_psd.write_image(str(png_path), width=1400, height=500, scale=2)
    print(f"  Saved: {png_path}")

    _export_psd_csv(psd_results, output_dir)
    print(f"  Saved: {output_dir / 'fwm_psd_comparison.csv'}")

    # ---- Figures 2a & 2b: FWM 长度扫描 ----
    print("\nComputing FWM length sweep for all models...")
    sweep_results = compute_fwm_length_sweep_results(wdm_config, osa_csv_path)

    # 打印前几个长度点验证
    print("\nForward FWM noise at center quantum channel (sample lengths):")
    for r in sweep_results:
        print(f"  {r.key}: " + " ".join(f"L={L}km,P={p:.3e}W" for L, p in zip(r.lengths_km[:5], r.fwd_noise_W[:5])))

    print("\nGenerating forward FWM length sweep figure...")
    fig_fwd = _make_fwm_length_sweep_subfigure(sweep_results, "forward")
    html_fwd = output_dir / "fwm_length_sweep_forward.html"
    fig_fwd.write_html(str(html_fwd), post_script=_LEGEND_SYNC_JS, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved: {html_fwd}")
    png_fwd = output_dir / "fwm_length_sweep_forward.png"
    fig_fwd.write_image(str(png_fwd), width=900, height=500, scale=2)
    print(f"  Saved: {png_fwd}")

    print("\nGenerating backward FWM length sweep figure...")
    fig_bwd = _make_fwm_length_sweep_subfigure(sweep_results, "backward")
    html_bwd = output_dir / "fwm_length_sweep_backward.html"
    fig_bwd.write_html(str(html_bwd), post_script=_LEGEND_SYNC_JS, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved: {html_bwd}")
    png_bwd = output_dir / "fwm_length_sweep_backward.png"
    fig_bwd.write_image(str(png_bwd), width=900, height=500, scale=2)
    print(f"  Saved: {png_bwd}")

    _export_length_csv(sweep_results, output_dir)
    print(f"  Saved: {output_dir / 'fwm_length_sweep.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
