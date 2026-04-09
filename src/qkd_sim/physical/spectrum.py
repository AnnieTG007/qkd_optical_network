"""噪声功率谱绘图模块（离散模型）。

提供以下绘图函数：
  plot_signal_spectrum        — 经典/量子信道信号功率谱
  plot_noise_spectrum         — 量子信道噪声功率谱（SpRS / FWM / 合并）
  make_noise_figures          — 一次生成 W + dBm 双版本的全套谱图

"噪声功率谱"的定义（离散模型）：
  - x 轴：量子信道扫描频率 [THz]
  - y 轴：该量子信道位置处的噪声功率 [W] 或 [dBm]
  - 通过 sweep_noise_spectrum() 脚本预先计算每个频率点的噪声，
    本模块只负责绘图，不做计算。

使用示例（最小版本）：
    >>> from qkd_sim.physical.spectrum import plot_signal_spectrum, plot_noise_spectrum
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    >>> plot_signal_spectrum(wdm_grid, axes[0], unit="dBm")
    >>> plot_noise_spectrum(f_q_hz, {"sprs_fwd": ..., "fwm_fwd": ...}, axes[1], unit="dBm")
    >>> plt.tight_layout()
    >>> plt.show()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qkd_sim.physical.signal import WDMGrid
from qkd_sim.utils.units import power_W_to_dBm


def _configure_cjk_font() -> None:
    """尝试配置支持中文的 matplotlib 字体。

    优先级：Microsoft YaHei → SimHei → SimSun → 不做处理（保持默认）。
    在 module import 时自动调用，无需用户手动配置。
    """
    import matplotlib.font_manager as fm
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


_configure_cjk_font()

# ---------------------------------------------------------------------------
# 绘图常量
# ---------------------------------------------------------------------------

_COLORS = {
    "classical": "#1f77b4",   # 经典信道信号
    "quantum":   "#d62728",   # 量子信道信号（参考）
    "sprs_fwd":  "#2ca02c",   # SpRS 前向
    "sprs_bwd":  "#98df8a",   # SpRS 后向
    "fwm_fwd":   "#ff7f0e",   # FWM 前向
    "fwm_bwd":   "#ffbb78",   # FWM 后向
    "total_fwd": "#9467bd",   # 总噪声前向
    "total_bwd": "#c5b0d5",   # 总噪声后向
}

_NOISE_LABELS = {
    "sprs_fwd": "SpRS 前向",
    "sprs_bwd": "SpRS 后向",
    "fwm_fwd":  "FWM 前向",
    "fwm_bwd":  "FWM 后向",
    "total_fwd": "SpRS+FWM 前向",
    "total_bwd": "SpRS+FWM 后向",
}

_FLOOR_DBM = -200.0   # dBm 下限（替代 log(0)）


def _to_display(P_W: np.ndarray, unit: str) -> np.ndarray:
    """将功率 [W] 转换为显示单位（W 或 dBm）。

    Parameters
    ----------
    P_W : ndarray
        功率 [W]
    unit : {"W", "dBm"}
        目标单位

    Returns
    -------
    ndarray
        转换后的功率数组
    """
    if unit == "W":
        return P_W
    # dBm：P=0 时使用下限
    positive = P_W > 0
    result = np.full_like(P_W, _FLOOR_DBM, dtype=np.float64)
    result[positive] = power_W_to_dBm(P_W[positive])
    return result


def _ylabel(unit: str) -> str:
    return "功率 [W]" if unit == "W" else "功率 [dBm]"


# ---------------------------------------------------------------------------
# 信号功率谱
# ---------------------------------------------------------------------------

def plot_signal_spectrum(
    wdm_grid: WDMGrid,
    ax: Axes,
    unit: str = "dBm",
    title: str = "信号功率谱",
) -> None:
    """绘制 WDM 信道信号功率谱（茎图）。

    经典信道用蓝色，量子信道用红色（通常功率为 0 或极低）。

    Parameters
    ----------
    wdm_grid : WDMGrid
        WDM 信道网格
    ax : Axes
        目标坐标轴
    unit : {"W", "dBm"}
        纵轴单位
    title : str
        子图标题
    """
    all_chs = wdm_grid.channels
    f_all = np.array([ch.f_center for ch in all_chs])
    P_all = np.array([ch.power for ch in all_chs])
    types = [ch.channel_type for ch in all_chs]

    f_THz = f_all / 1e12  # Hz → THz

    # 追踪已添加图例的信道类型，避免重复标签
    labeled: set[str] = set()
    for f, P, t in zip(f_THz, P_all, types):
        if t == "classical":
            # 经典信道：绘制功率茎图
            color = _COLORS["classical"]
            P_disp = float(_to_display(np.array([P]), unit)[0])
            markerline, stemlines, baseline = ax.stem(
                [f], [P_disp], linefmt=color, markerfmt="o", basefmt=" "
            )
            markerline.set_color(color)
            stemlines.set_color(color)
            if "经典信道" not in labeled:
                markerline.set_label("经典信道")
                labeled.add("经典信道")
        else:
            # 量子信道：仅标注频率位置，不显示功率
            ax.axvline(x=f, color=_COLORS["quantum"], linestyle=":",
                       linewidth=1.2, alpha=0.7)
            if "量子信道位置" not in labeled:
                ax.axvline(x=f, color=_COLORS["quantum"], linestyle=":",
                           linewidth=1.2, alpha=0.7, label="量子信道位置")
                labeled.add("量子信道位置")

    ax.set_xlabel("频率 [THz]")
    ax.set_ylabel(_ylabel(unit))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 噪声功率谱
# ---------------------------------------------------------------------------

def plot_noise_spectrum(
    f_q_hz: np.ndarray,
    noise_dict: dict[str, np.ndarray],
    ax: Axes,
    unit: str = "dBm",
    show_keys: Sequence[str] | None = None,
    title: str = "量子信道噪声功率谱",
    f_c_hz: np.ndarray | None = None,
    discrete: bool = True,
) -> None:
    """绘制量子信道噪声功率谱曲线。

    Parameters
    ----------
    f_q_hz : ndarray, shape (N_q,)
        量子信道扫描频率 [Hz]
    noise_dict : dict
        键为噪声类型，值为对应噪声功率数组 [W]，shape (N_q,)。
        支持的键：'sprs_fwd', 'sprs_bwd', 'fwm_fwd', 'fwm_bwd', 'total_fwd', 'total_bwd'
    ax : Axes
        目标坐标轴
    unit : {"W", "dBm"}
        纵轴单位
    show_keys : list of str or None
        指定绘制哪些噪声类型。None 时绘制 noise_dict 中所有键。
    title : str
        子图标题
    f_c_hz : ndarray or None
        经典信道频率 [Hz]，若提供则在 x 轴底部画竖线标记泵浦位置。
    discrete : bool
        True（默认）：离散模型，绘制带标记折线（"o-"），用于当前信道中心频率噪声；
        False：连续模型，绘制平滑曲线（"-"），用于矩形/升余弦/OSA 连续谱噪声。
    """
    f_THz = f_q_hz / 1e12

    keys = show_keys if show_keys is not None else list(noise_dict.keys())

    for key in keys:
        if key not in noise_dict:
            continue
        P_W = noise_dict[key]
        P_disp = _to_display(P_W, unit)
        color = _COLORS.get(key, "gray")
        label = _NOISE_LABELS.get(key, key)
        line_style = "o-" if discrete else "-"
        markersize = 4 if discrete else 0
        ax.plot(f_THz, P_disp, line_style, color=color, label=label,
                linewidth=1.5, markersize=markersize)

    # 标记经典信道（泵浦）位置
    if f_c_hz is not None:
        for fc in f_c_hz / 1e12:
            ax.axvline(fc, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
        # 仅加一次图例
        ax.axvline(f_c_hz[0] / 1e12, color="gray", linewidth=0.7,
                   linestyle="--", alpha=0.5, label="经典信道位置")

    ax.set_xlabel("量子信道频率 [THz]")
    ax.set_ylabel(_ylabel(unit))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 一次性生成全套谱图（W + dBm 双版本）
# ---------------------------------------------------------------------------

def make_noise_figures(
    f_q_hz: np.ndarray,
    noise_dict: dict[str, np.ndarray],
    wdm_grid_ref: WDMGrid,
    output_dir: str | Path | None = None,
    dpi: int = 150,
    discrete: bool = True,
    save_csv: bool = False,
) -> dict[str, Figure]:
    """生成全套噪声功率谱图（W 和 dBm 各一套），可选保存 CSV。

    按项目需求生成以下 8 张图（CLAUDE.md 第 A 节）：
      1. 信号功率谱 (W)
      2. 信号功率谱 (dBm)
      3. SpRS 噪声谱 (W)
      4. SpRS 噪声谱 (dBm)
      5. FWM 噪声谱 (W)
      6. FWM 噪声谱 (dBm)
      7. FWM+SpRS 合并噪声谱 (W)
      8. FWM+SpRS 合并噪声谱 (dBm)

    Parameters
    ----------
    f_q_hz : ndarray, shape (N_q,)
        量子信道频率 [Hz]
    noise_dict : dict
        噪声结果字典，键包括 "sprs_fwd"/"sprs_bwd"/"fwm_fwd"/"fwm_bwd"，
        每个值 shape (N_q,)
    wdm_grid_ref : WDMGrid
        参考 WDMGrid（用于信号功率谱图 + CSV 信道信息）
    output_dir : str, Path or None
        输出目录。None 时不保存文件。
    dpi : int
        图像分辨率
    discrete : bool
        True（默认）：离散模型，带标记折线；False：连续模型，平滑曲线。
    save_csv : bool
        True 时在 output_dir 下保存 signal_spectrum.csv 和 noise_spectrum.csv。

    Returns
    -------
    dict[str, Figure]
        键为图名，值为 Figure 对象
    """
    out: Path | None = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    # 计算合并噪声
    noise_plot = dict(noise_dict)  # shallow copy
    if "sprs_fwd" in noise_dict and "fwm_fwd" in noise_dict:
        noise_plot["total_fwd"] = noise_dict["sprs_fwd"] + noise_dict["fwm_fwd"]
    if "sprs_bwd" in noise_dict and "fwm_bwd" in noise_dict:
        noise_plot["total_bwd"] = noise_dict["sprs_bwd"] + noise_dict["fwm_bwd"]

    # 经典信道位置（用于在噪声谱图上标记泵浦）
    f_c_hz = np.array([ch.f_center for ch in wdm_grid_ref.get_classical_channels()])

    figures: dict[str, Figure] = {}

    def _save_fig(fig: Figure, name: str) -> None:
        figures[name] = fig
        if out is not None:
            fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")

    # 1-2. 信号功率谱
    for unit in ("W", "dBm"):
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_signal_spectrum(wdm_grid_ref, ax, unit=unit, title=f"信号功率谱 [{unit}]")
        plt.tight_layout()
        _save_fig(fig, f"signal_spectrum_{unit}")

    # 3-4. SpRS 噪声谱
    for unit in ("W", "dBm"):
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_noise_spectrum(
            f_q_hz, noise_plot, ax,
            unit=unit,
            show_keys=["sprs_fwd", "sprs_bwd"],
            title=f"量子信道 SpRS 噪声功率谱 [{unit}]",
            f_c_hz=f_c_hz,
            discrete=discrete,
        )
        plt.tight_layout()
        _save_fig(fig, f"sprs_noise_spectrum_{unit}")

    # 5-6. FWM 噪声谱
    for unit in ("W", "dBm"):
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_noise_spectrum(
            f_q_hz, noise_plot, ax,
            unit=unit,
            show_keys=["fwm_fwd", "fwm_bwd"],
            title=f"量子信道 FWM 噪声功率谱 [{unit}]",
            f_c_hz=f_c_hz,
            discrete=discrete,
        )
        plt.tight_layout()
        _save_fig(fig, f"fwm_noise_spectrum_{unit}")

    # 7-8. FWM+SpRS 合并噪声谱
    for unit in ("W", "dBm"):
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_noise_spectrum(
            f_q_hz, noise_plot, ax,
            unit=unit,
            show_keys=["total_fwd", "total_bwd"],
            title=f"量子信道 FWM+SpRS 合并噪声功率谱 [{unit}]",
            f_c_hz=f_c_hz,
            discrete=discrete,
        )
        plt.tight_layout()
        _save_fig(fig, f"total_noise_spectrum_{unit}")

    # --- CSV 输出 ---
    if save_csv and out is not None:
        _save_signal_csv(wdm_grid_ref, out)
        _save_noise_csv(f_q_hz, noise_plot, out)

    return figures


def _save_signal_csv(wdm_grid_ref: WDMGrid, out: Path) -> None:
    """保存信号功率谱 CSV（每行一个 WDM 信道）。"""
    chs = wdm_grid_ref.channels
    f_THz = np.array([ch.f_center / 1e12 for ch in chs])
    P_W = np.array([ch.power for ch in chs])
    types = [ch.channel_type for ch in chs]

    # dBm：P=0 时用 -200 填充
    P_dBm = np.where(P_W > 0, power_W_to_dBm(P_W), _FLOOR_DBM)

    lines = ["frequency_THz,power_W,power_dBm,channel_type"]
    for f, pw, pd, t in zip(f_THz, P_W, P_dBm, types):
        pw_str = f"{pw:.6e}" if (abs(pw) < 1e-3 or abs(pw) >= 1e4) else f"{pw:.6f}"
        lines.append(f"{f:.6f},{pw_str},{pd:.3f},{t}")
    (out / "signal_spectrum.csv").write_text("\n".join(lines), encoding="utf-8")


def _save_noise_csv(f_q_hz: np.ndarray, noise_plot: dict[str, np.ndarray], out: Path) -> None:
    """保存噪声功率谱 CSV（每行一个量子信道）。

    列顺序：f_q_THz, sprs_fwd_W, sprs_fwd_dBm, sprs_bwd_W, sprs_bwd_dBm,
            fwm_fwd_W, fwm_fwd_dBm, fwm_bwd_W, fwm_bwd_dBm,
            total_fwd_W, total_fwd_dBm, total_bwd_W, total_bwd_dBm
    """
    f_THz = f_q_hz / 1e12
    COLS = ["sprs_fwd", "sprs_bwd", "fwm_fwd", "fwm_bwd", "total_fwd", "total_bwd"]

    # 按列收集数据
    col_names = ["f_q_THz"]
    col_vals: list[np.ndarray] = [f_THz]
    for key in COLS:
        if key in noise_plot:
            P_W = noise_plot[key]
            P_dBm = np.where(P_W > 0, power_W_to_dBm(P_W), _FLOOR_DBM)
            col_names.append(f"{key}_W")
            col_names.append(f"{key}_dBm")
            col_vals.append(P_W)
            col_vals.append(P_dBm)

    data = np.column_stack(col_vals)  # shape (N_q, n_cols)
    fmt = _FLOOR_DBM

    lines = [",".join(col_names)]
    for row in data:
        row_strs = []
        for v in row:
            if abs(v) < 1e-3 or abs(v) >= 1e4:
                row_strs.append(f"{v:.6e}")
            else:
                row_strs.append(f"{v:.6f}")
        lines.append(",".join(row_strs))
    (out / "noise_spectrum.csv").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 4: 多信号模型对比绘图（连续 vs 离散）
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Sequence

_MODEL_COLORS = {
    "discrete": "#202020",
    "rc_beta0": "#1f77b4",
    "rc_beta001": "#aec7e8",
    "rc_beta01": "#d95f02",
    "rc_beta05": "#ff7f0e",
    "osa": "#2ca02c",
}


def _ylabel_psd(unit: str) -> str:
    return "Launch PSD [W/Hz]" if unit == "W" else "Launch PSD [dBm/Hz]"


def _ylabel_power_bin(unit: str) -> str:
    return "Power per Bin [W]" if unit == "W" else "Power per Bin [dBm]"


@dataclass
class ModelSpectrumResult:
    """单信号模型的噪声频谱结果（用于模型对比图）。

    使用噪声 PSD（连续曲线）而非信道积分噪声（标量）。
    f_noise_hz 上的 G_noise(f) 即为噪声功率谱密度 [W/Hz]。
    """
    key: str
    label: str
    color: str
    f_signal_hz: np.ndarray
    signal_psd_W_per_Hz: np.ndarray
    f_noise_hz: np.ndarray
    noise_df_hz: float
    fwm_psd_W_per_Hz: np.ndarray
    sprs_psd_W_per_Hz: np.ndarray

    @property
    def total_psd_W_per_Hz(self) -> np.ndarray:
        return self.fwm_psd_W_per_Hz + self.sprs_psd_W_per_Hz


@dataclass
class ModelLengthSweepResult:
    """单信号模型的噪声-光纤长度扫描结果。"""
    key: str
    label: str
    color: str
    length_km: np.ndarray
    fwm_W: np.ndarray
    sprs_W: np.ndarray

    @property
    def total_W(self) -> np.ndarray:
        return self.fwm_W + self.sprs_W


@dataclass
class SignalPSDResult:
    """单信号模型的发射功率谱结果（用于信号 PSD 对比图）。

    纵轴语义（修正 2026-04）：
      - Discrete: psd_W_per_Hz = 信道功率 P [W]（stem 高度 = P）
      - 连续模型: psd_W_per_Hz = PSD [W/Hz]；绘图时乘 df 得 bin 功率 [W]
    """
    key: str
    label: str
    color: str
    f_hz: np.ndarray          # 频率网格 [Hz]
    psd_W_per_Hz: np.ndarray  # PSD [W/Hz]（离散模型存 P [W]）
    integrated_power_W: float  # 积分功率（验证 = P0）


def make_signal_psd_comparison_figure(
    results: Sequence[SignalPSDResult],
    unit: str = "W",
) -> Figure:
    """绘制多信号模型的发射 PSD 对比图（1×2 布局：线性 + 对数）。

    对比的模型：
      - Discrete: delta 近似（stem 竖线）
      - Raised Cosine β=0 (≡矩形)
      - Raised Cosine β=0.01
      - Raised Cosine β=0.1
      - Raised Cosine β=0.5
      - OSA

    **物理说明（修正 2026-04）**：
      - 离散模型 stem：高度 = 信道功率 P [W]（不是 P/df）
      - 连续模型曲线：高度 = PSD × Δf [W]（每个采样点的 bin 功率）
      - 两者纵轴量纲统一为 [W]，可直观比较；积分均为 P0 ✓

    Parameters
    ----------
    results : Sequence[SignalPSDResult]
        各信号模型的 PSD 结果
    unit : str
        "W" 或 "dBm"

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax in axes:
        for result in results:
            f_THz = result.f_hz / 1e12
            if result.key == "discrete":
                # 离散模型：psd_W_per_Hz 存的是信道功率 P [W]
                y = _to_display(result.psd_W_per_Hz, unit)
                mask = result.psd_W_per_Hz > 0.0
                markerline, stemlines, _ = ax.stem(
                    f_THz[mask], y[mask],
                    linefmt=result.color, markerfmt=" ", basefmt=" ",
                    label=result.label,
                )
                stemlines.set_linewidth(1.5)
                markerline.set_visible(False)
            else:
                # 连续模型：PSD [W/Hz] × df [Hz] = bin 功率 [W]
                df = float(np.mean(np.diff(result.f_hz)))
                power_per_bin = result.psd_W_per_Hz * df
                y = _to_display(power_per_bin, unit)
                ax.plot(
                    f_THz, y,
                    color=result.color, linewidth=2.0, label=result.label,
                )
        ax.set_xlabel("Frequency [THz]")
        ax.set_ylabel(_ylabel_power_bin(unit))
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_title("Signal Launch Power per Bin (Linear Scale)")
    axes[0].set_ylim(bottom=0.0)

    axes[1].set_title("Signal Launch Power per Bin (Log Scale)")

    # ---- xlim: 显示全 C 波段（80 信道），以便看到 RC/OS rolloff 形状差异 ----
    # x 轴范围从网格最小频率到最大频率
    f_plot_min = min(r.f_hz.min() for r in results)
    f_plot_max = max(r.f_hz.max() for r in results)
    for ax in axes:
        ax.set_xlim(f_plot_min / 1e12, f_plot_max / 1e12)

    # ---- log 图 ylim：从全部数据的动态范围计算 ----
    # 收集所有连续模型在全部频率网格上的功率值
    all_power_W = []
    for result in results:
        if result.key == "discrete":
            continue
        df = float(np.mean(np.diff(result.f_hz)))
        power_per_bin = result.psd_W_per_Hz * df  # [W]
        all_power_W.append(power_per_bin[power_per_bin > 0])

    if all_power_W:
        all_W = np.concatenate(all_power_W)
        p_min = float(all_W.min())
        p_max = float(all_W.max())
        # headroom: -30 dB below min, +10 dB above max
        y_bot_W = p_min / 1000   # -30 dB
        y_top_W = p_max * 10     # +10 dB
        axes[1].set_ylim(y_bot_W, y_top_W)
        axes[1].set_yscale("log")
        axes[1].axhline(y=y_bot_W, color="#cccccc", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[], loc="upper center", ncol=len(results), frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    non_empty = [h for h, l in zip(handles, labels) if l]
    fig.legend(non_empty, [l for l in labels if l],
               loc="upper center", ncol=len(results), frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.text(
        0.5, -0.04,
        "Note: Discrete stems show channel power P [W]. "
        "Continuous curves show PSD × Δf [W] (power per bin). "
        "Both share the same unit [W] — integrals equal total channel power P0.",
        ha="center", fontsize=8, style="italic", color="#555555",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def _noise_bin_power(psd: np.ndarray, df: float, unit: str) -> np.ndarray:
    """将噪声 PSD [W/Hz] 转换为每 bin 功率用于绘图。"""
    floor_W = np.finfo(np.float64).tiny
    power_W = np.maximum(psd * df, floor_W)
    if unit == "W":
        return power_W
    return _to_display(power_W, unit)


def _ylabel_noise_bin(unit: str) -> str:
    return "G_noise(f) × df [W]" if unit == "W" else "G_noise(f) × df [dBm]"


def make_model_comparison_figure(
    results: Sequence[ModelSpectrumResult],
    unit: str = "W",
) -> Figure:
    """绘制 2×2 多信号模型噪声功率谱对比图（使用噪声 PSD 曲线）。

    噪声子图使用 G_noise(f) × df 曲线，而非信道积分噪声标量。
    这使得连续模型的宽带低峰特性得以体现。

    子图布局：
      (0,0) FWM 噪声 PSD 谱 vs 频率
      (0,1) SpRS 噪声 PSD 谱 vs 频率
      (1,0) 总噪声 PSD 谱 vs 频率
      (1,1) 信号发射 PSD（离散=竖线，连续=曲线）

    Parameters
    ----------
    results : Sequence[ModelSpectrumResult]
        各信号模型的计算结果
    unit : {"W", "dBm"}
        显示单位

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_fwm, ax_sprs = axes[0]
    ax_total, ax_signal = axes[1]

    for result in results:
        f_noise_THz = result.f_noise_hz / 1e12
        ax_fwm.plot(
            f_noise_THz,
            _noise_bin_power(result.fwm_psd_W_per_Hz, result.noise_df_hz, unit),
            color=result.color, linewidth=2.0, label=result.label,
        )
        ax_sprs.plot(
            f_noise_THz,
            _noise_bin_power(result.sprs_psd_W_per_Hz, result.noise_df_hz, unit),
            color=result.color, linewidth=2.0, label=result.label,
        )
        ax_total.plot(
            f_noise_THz,
            _noise_bin_power(result.total_psd_W_per_Hz, result.noise_df_hz, unit),
            color=result.color, linewidth=2.0, label=result.label,
        )

    for ax, title in [
        (ax_fwm, "FWM Noise PSD Spectrum"),
        (ax_sprs, "SpRS Noise PSD Spectrum"),
        (ax_total, "Total Noise PSD Spectrum"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Frequency [THz]")
        ax.set_ylabel(_ylabel_noise_bin(unit))
        ax.grid(True, alpha=0.3, which="both")
        if unit == "W":
            ax.set_yscale("log")

    # 信号 PSD 子图
    reference = results[0]
    for result in results:
        f_signal_THz = result.f_signal_hz / 1e12
        y_signal = _to_display(result.signal_psd_W_per_Hz, unit)
        if result.key == "discrete":
            mask = result.signal_psd_W_per_Hz > 0.0
            markerline, stemlines, _ = ax_signal.stem(
                f_signal_THz[mask], y_signal[mask],
                linefmt=result.color, markerfmt=" ", basefmt=" ",
                label=result.label,
            )
            stemlines.set_linewidth(1.5)
            markerline.set_visible(False)
        else:
            ax_signal.plot(
                f_signal_THz, y_signal,
                color=result.color, linewidth=2.0, label=result.label,
            )

    for fq in reference.f_noise_hz / 1e12:
        ax_signal.axvline(fq, color="#999999", linestyle=":", linewidth=0.5, alpha=0.15)

    ax_signal.set_title("Signal Launch PSD")
    ax_signal.set_xlabel("Frequency [THz]")
    ax_signal.set_ylabel(_ylabel_psd(unit))
    ax_signal.grid(True, alpha=0.3)

    handles, labels = ax_fwm.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(results), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def make_noise_vs_length_figure(
    results: Sequence[ModelLengthSweepResult],
) -> Figure:
    """绘制 1×3 噪声功率随光纤长度变化对比图。

    子图布局：
      (0,) FWM 噪声 vs L
      (1,) SpRS 噪声 vs L
      (2,) 总噪声 vs L

    X 轴: 光纤长度 [km], Y 轴: 噪声功率 [W] (log scale)

    Parameters
    ----------
    results : Sequence[ModelLengthSweepResult]
        各信号模型的 L 扫描结果

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    panels = [
        ("FWM Noise", "fwm_W"),
        ("SpRS Noise", "sprs_W"),
        ("Total Noise", "total_W"),
    ]

    floor = np.finfo(np.float64).tiny

    for ax, (title, field_name) in zip(axes, panels):
        for result in results:
            y = getattr(result, field_name)
            ax.plot(
                result.length_km,
                np.maximum(y, floor),
                color=result.color, linewidth=2.0, label=result.label,
            )
        ax.set_title(title)
        ax.set_xlabel("Fiber Length [km]")
        ax.set_ylabel("Noise Power [W]")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(results), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def get_model_color(model_key: str) -> str:
    """返回信号模型对应的绘图颜色。"""
    return _MODEL_COLORS[model_key]
