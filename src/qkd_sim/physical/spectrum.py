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
