"""信号发射功率谱对比绘图脚本。

比较以下信号模型的发射功率（离散 vs 连续）：
  - Discrete: delta 近似（stem）
  - Raised Cosine β=0 (≡矩形)
  - Raised Cosine β=0.01
  - Raised Cosine β=0.5
  - Raised Cosine β=1
  - OSA

输出到: outputs/phase4_N80_C3/Signal_TX/

物理说明（见 spectrum.make_signal_psd_comparison_figure 图注）：
  离散模型的 stem 高度 = 信道功率 P [W]；
  连续模型的曲线高度 = PSD × Δf [W]（bin 功率）；量纲统一为 [W]。
"""

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
sys.path.insert(0, str(_PROJECT_ROOT))   # 让 scripts/ 可作为包导入
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.schema import WDMConfig
from qkd_sim.config.plot_config import get_color, load_model_specs
from qkd_sim.physical.spectrum import SignalPSDResult

# ============================================================================
# 配置参数（与 plot_noise_spectrum.py 保持一致）
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

FREQ_GRID_RESOLUTION_HZ = 0.1e9
FREQ_GRID_PADDING_FACTOR = 1.5

OSA_RBW_HZ = 1.0e9
OSA_CSV_PATH = _PROJECT_ROOT / "data" / "osa"



def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV files found in {OSA_CSV_PATH}")
    return csv_files[0]


def _build_frequency_grid(config: WDMConfig) -> np.ndarray:
    half_span = (config.N_ch - 1) / 2 * config.channel_spacing
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = config.f_center - half_span - padding
    f_max = config.f_center + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / FREQ_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_discrete_signal_psd(
    wdm_grid, f_grid: np.ndarray
) -> np.ndarray:
    """离散模型信号功率谱：每个格点存储信道功率 P [W]。

    离散信道没有 PSD 概念，只有信道功率 P。
    这里用 P（而非 P/df）作为"谱"值，使与连续模型的 PSD×Δf 量纲一致。
    """
    psd = np.zeros_like(f_grid, dtype=np.float64)
    for ch in wdm_grid.get_classical_channels():
        idx = int(np.argmin(np.abs(f_grid - ch.f_center)))
        psd[idx] += ch.power
    return psd


def _integrate_psd(f_grid: np.ndarray, psd: np.ndarray, is_discrete: bool = False) -> float:
    """均匀网格黎曼和积分。

    Parameters
    ----------
    f_grid : ndarray
        频率网格 [Hz]
    psd : ndarray
        PSD [W/Hz] 或离散功率 [W]
    is_discrete : bool
        若为 True，psd 存储的是信道功率 P [W]（不含 df），
        积分即为 sum(psd)。
        若为 False，psd 存储的是 PSD [W/Hz]，积分用 sum(psd * df)。
    """
    if is_discrete:
        return float(np.sum(psd))
    df = float(np.mean(np.diff(f_grid)))
    return float(np.sum(psd * df))


def _compute_signal_tx_results(
    wdm_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
    specs: dict[str, dict],
) -> list[SignalPSDResult]:
    """计算所有信号模型的 PSD 结果。

    Parameters
    ----------
    wdm_config : WDMConfig
        基础 WDM 配置
    f_grid : ndarray
        频率网格
    osa_csv_path : Path
        OSA CSV 文件路径
    specs : dict[str, dict]
        从 load_model_specs("signal_tx") 获取的模型规格字典

    Returns
    -------
    list[SignalPSDResult]
    """
    from scripts.plot_noise_spectrum import _build_signal_tx_grid  # noqa: E402

    results: list[SignalPSDResult] = []

    for model_key, spec in specs.items():
        grid = _build_signal_tx_grid(model_key, wdm_config, f_grid, osa_csv_path)

        if model_key == "discrete":
            psd = _build_discrete_signal_psd(grid, f_grid)
            integrated_power = _integrate_psd(f_grid, psd, is_discrete=True)
        else:
            psd = grid.get_total_psd()
            integrated_power = _integrate_psd(f_grid, psd, is_discrete=False)

        results.append(
            SignalPSDResult(
                key=model_key,
                label=spec["label"],
                color=spec["color"],
                f_hz=f_grid,
                psd_W_per_Hz=psd,
                integrated_power_W=integrated_power,
            )
        )

    return results


def _export_csv(results: list[SignalPSDResult], out_dir: Path) -> None:
    """导出信号 PSD 数据到 CSV（W + dBm）。"""
    header = ["f_THz"]
    rows: list[list[float]] = []

    # 对齐频率网格
    ref_f = results[0].f_hz
    for r in results:
        assert np.allclose(r.f_hz, ref_f), f"Frequency mismatch for {r.key}"

    for r in results:
        header.append(f"{r.key}_W_per_Hz")
        header.append(f"{r.key}_dBm_per_Hz")
        rows.append([])

    csv_lines = [",".join(header)]
    f_THz = ref_f / 1e12

    def _to_dBm(v: float) -> float:
        return 10 * np.log10(max(v, 1e-30)) + 30

    for i in range(len(ref_f)):
        row = [f"{f_THz[i]:.6f}"]
        for r in results:
            row.append(f"{r.psd_W_per_Hz[i]:.6e}")
            row.append(f"{_to_dBm(r.psd_W_per_Hz[i]):.3f}")
        csv_lines.append(",".join(row))

    # 积分功率行
    int_row = ["# integrated_power_W"]
    for r in results:
        int_row.append(f"{r.key}:{r.integrated_power_W:.6e}")
    csv_lines.append("")
    csv_lines.append(",".join(int_row))

    (out_dir / "signal_tx.csv").write_text("\n".join(csv_lines), encoding="utf-8")


def _to_dBm_for_plotly(v: np.ndarray | float) -> np.ndarray | float:
    """将功率 [W] 转换为 dBm（向量化，支持数组输入）。"""
    v_arr = np.asarray(v, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(v_arr, 1e-30)) + 30.0


def _compute_power_per_bin(result: SignalPSDResult) -> np.ndarray:
    """计算每 bin 功率 [W]（离散模型直接返回 psd，连续模型返回 psd × df）。"""
    if result.key == "discrete":
        return result.psd_W_per_Hz  # 信道功率 P [W]
    df = float(np.mean(np.diff(result.f_hz)))
    return result.psd_W_per_Hz * df  # [W]


def make_signal_psd_plotly(results: list[SignalPSDResult]) -> go.Figure:
    """生成 Plotly 交互式信号 PSD 对比图（双 y 轴：线性 + 对数）。

    通过点击图例可以动态开关各模型曲线。
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Run: pip install plotly")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Signal Launch Power per Bin (W, Log Scale)",
                        "Signal Launch Power per Bin (dBm, Linear Scale)"),
        shared_xaxes=False,
    )

    for idx, result in enumerate(results):
        f_THz = result.f_hz / 1e12
        power_bin_W = _compute_power_per_bin(result)
        power_bin_dBm = _to_dBm_for_plotly(power_bin_W)

        # 过滤零功率点
        mask = power_bin_W > 0
        f_plot = f_THz[mask]
        y_lin = power_bin_W[mask]
        y_log = power_bin_dBm[mask]

        if result.key == "discrete":
            # 离散模型：散点（marker）
            # legendgroup 确保点击图例时同时切换左右子图的曲线
            # col=1 的 trace 在图例中显示
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_lin,
                    mode="markers",
                    marker=dict(size=6, color=result.color, symbol="circle"),
                    name=result.label,
                    legendgroup=result.key,
                    showlegend=True,
                    hovertemplate=f"f={{x:.4f}} THz<br>P={{y:.3e}} W<extra>{result.label}</extra>",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_log,
                    mode="markers",
                    marker=dict(size=6, color=result.color, symbol="circle"),
                    name=result.label,
                    legendgroup=result.key,
                    showlegend=False,
                    hovertemplate=f"f={{x:.4f}} THz<br>P={{y:.2f}} dBm<extra>{result.label}</extra>",
                ),
                row=1, col=2,
            )
        else:
            # 连续模型：线状曲线
            # legendgroup 确保点击图例时同时切换左右子图的曲线
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_lin,
                    mode="lines",
                    line=dict(color=result.color, width=2.0),
                    name=result.label,
                    legendgroup=result.key,
                    showlegend=True,
                    hovertemplate=f"f={{x:.4f}} THz<br>P={{y:.3e}} W<extra>{result.label}</extra>",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=f_plot, y=y_log,
                    mode="lines",
                    line=dict(color=result.color, width=2.0),
                    name=result.label,
                    legendgroup=result.key,
                    showlegend=False,
                    hovertemplate=f"f={{x:.4f}} THz<br>P={{y:.2f}} dBm<extra>{result.label}</extra>",
                ),
                row=1, col=2,
            )

    # ---- 计算动态 xlim（基于非零功率频率范围）----
    all_f_nonzero: list[np.ndarray] = []
    for result in results:
        power_bin = _compute_power_per_bin(result)
        all_f_nonzero.append((result.f_hz / 1e12)[power_bin > 0])
    if all_f_nonzero:
        all_f = np.concatenate(all_f_nonzero)
        f_min = float(all_f.min()) - 0.1
        f_max = float(all_f.max()) + 0.1
    else:
        f_min, f_max = 191.0, 196.0

    # ---- 计算动态 ylim ----
    all_pmax: list[float] = []
    for result in results:
        power_bin = _compute_power_per_bin(result)
        nonzero = power_bin[power_bin > 0]
        if nonzero.size > 0:
            all_pmax.append(float(nonzero.max()))
    if all_pmax:
        y_bot_lin = min(all_pmax) / 10.0
        y_top_lin = max(all_pmax) * 10.0
        y_bot_log_dBm = _to_dBm_for_plotly(y_bot_lin)
        y_top_log_dBm = _to_dBm_for_plotly(y_top_lin)
    else:
        y_bot_lin, y_top_lin = 1e-7, 1e-1
        y_bot_log_dBm, y_top_log_dBm = -50.0, 10.0

    # ---- 更新子图布局 ----
    # 左图（col=1）：W 单位 → 对数刻度；右图（col=2）：dBm 单位 → 线性刻度
    fig.update_xaxes(title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=1)
    fig.update_xaxes(title_text="Frequency [THz]", range=[f_min, f_max], row=1, col=2)
    fig.update_yaxes(title_text="Power per Bin [W]", range=[y_bot_lin, y_top_lin], row=1, col=1)
    fig.update_yaxes(title_text="Power per Bin [dBm]", range=[y_bot_log_dBm, y_top_log_dBm], row=1, col=2)

    # 左图切换为对数刻度
    fig.update_yaxes(type="log", row=1, col=1)

    # ---- 样式更新 ----
    fig.update_layout(
        title=dict(
            text="Signal Launch Power per Bin — Interactive (click legend to toggle models, double-click to isolate)",
            x=0.5, xanchor="center",
        ),
        legend=dict(
            title=dict(text="Signal Model"),
            groupclick="toggleitem",
        ),
        template="plotly_white",
        width=1400,
        height=500,
    )

    return fig


_LEGEND_SYNC_JS = """
function syncLegendClicks() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;

    // 从 legend item DOM 节点找到对应的 curveNumber
    function getCurveNumber(node) {
        if (node._plotlyCurveNumber !== undefined) return node._plotlyCurveNumber;
        var parent = node.parentElement;
        while (parent) {
            if (parent._plotlyCurveNumber !== undefined) return parent._plotlyCurveNumber;
            parent = parent.parentElement;
        }
        return null;
    }

    // 单击图例项：切换该模型的显示/隐藏
    gd.on('plotly_legendclick', function(eventData) {
        var curveNumber = getCurveNumber(eventData.node);
        var fullData = gd._fullData || [];
        var clickedGroup = null;

        if (curveNumber !== null && fullData[curveNumber]) {
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

    // 双击图例项：利用 Plotly 内置隔离行为（默认在 handler 之前执行），
    // 在 handler 内检测是否有隐藏项：
    //   - 有隐藏项（Plotly 刚做完隔离）→ 恢复全部
    //   - 无隐藏项（当前全部可见）→ 无操作（Plotly 已完成隔离，保持）
    // 效果：单击图例切换显示/隐藏；双击图例隔离单个模型，再次双击任意图例恢复全部
    gd.on('plotly_legenddoubleclick', function(eventData) {
        var curveNumber = getCurveNumber(eventData.node);
        var fullData = gd._fullData || [];
        if (!curveNumber || !fullData[curveNumber]) return true;

        // Plotly 默认隔离行为已经执行，检查是否有 legendonly 项
        var hasHidden = false;
        for (var h = 0; h < gd.data.length; h++) {
            if (gd.data[h].visible === 'legendonly') { hasHidden = true; break; }
        }

        if (hasHidden) {
            // Plotly 刚隔离了某个模型，恢复全部显示
            for (var ri = 0; ri < gd.data.length; ri++) {
                gd.data[ri].visible = true;
            }
            Plotly.redraw(gd);
        }
        // 如果无隐藏项（当前全部可见），Plotly 的隔离已生效，无需额外操作
        return false;
    });
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', syncLegendClicks);
} else {
    syncLegendClicks();
}
"""


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    output_dir = _PROJECT_ROOT / "outputs" / "phase4_N80_C3" / "Signal_TX"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    wdm_config = WDMConfig(**WDM_PARAMS, quantum_channel_indices=[1, 2])
    f_grid = _build_frequency_grid(wdm_config)
    osa_csv_path = _resolve_osa_csv()

    print("Computing signal PSD for all models...")
    specs = load_model_specs("signal_tx")
    results = _compute_signal_tx_results(wdm_config, f_grid, osa_csv_path, specs)

    # 打印积分功率验证
    print("\nIntegrated powers (should all be ~P0 = 1e-3 W):")
    for r in results:
        print(f"  {r.key:15s}: {r.integrated_power_W:.6e} W")

    print("\nGenerating Plotly interactive figure...")
    fig = make_signal_psd_plotly(results)
    html_path = output_dir / "signal_tx_interactive.html"
    fig.write_html(
        str(html_path),
        post_script=_LEGEND_SYNC_JS,
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"  Saved: {html_path}")

    # 同时导出 PNG（静态备份）
    png_path = output_dir / "signal_tx.png"
    fig.write_image(str(png_path), width=1400, height=500, scale=2)
    print(f"  Saved: {png_path}")

    _export_csv(results, output_dir)
    print(f"  Saved: {output_dir / 'signal_tx.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
