"""离散噪声功率谱生成脚本（终版）。

运行方式：
    python scripts/plot_noise_spectrum.py

物理模型（可在 CONFIG 块修改）：
  - C 波段 WDM 网格，80 信道，50 GHz 间隔，中心 193.4 THz
  - 经典信道（泵浦）由 CLASSICAL_INDICES 指定，其余信道同时作为量子信道
  - 一次 compute_noise 计算全部量子信道的 SpRS + FWM 噪声
  - 物理假设：经典信道与量子信道同向传播（co-propagating）
  - 前向噪声 → 接收端 z=L（Bob）；后向噪声 → 发射端 z=0（Alice）

输出（outputs/discrete_N{N_ch}_C{n_classical}/ 目录）：
  signal_spectrum_W.png / signal_spectrum_dBm.png
  sprs_noise_spectrum_W.png / sprs_noise_spectrum_dBm.png
  fwm_noise_spectrum_W.png  / fwm_noise_spectrum_dBm.png
  total_noise_spectrum_W.png / total_noise_spectrum_dBm.png
  signal_spectrum.csv        （每行一个 WDM 信道，for Origin）
  noise_spectrum.csv         （每行一个量子信道，含全部噪声分量 W/dBm）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无 GUI 后端，适合脚本保存文件

# 将 src 加入路径（兼容直接 python 调用和 pytest）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import WDMGrid, SpectrumType, build_wdm_grid
from qkd_sim.physical.noise import DiscreteSPRSSolver, DiscreteFWMSolver, compute_noise
from qkd_sim.physical.spectrum import make_noise_figures

# ===========================================================================
# CONFIG — 在此修改仿真参数
# ===========================================================================

FIBER_PARAMS = dict(
    alpha_dB_per_km=0.2,        # 衰减 [dB/km]
    gamma_per_W_km=1.3,         # 非线性系数 [1/(W·km)]
    D_ps_nm_km=17.0,           # 色散 [ps/(nm·km)]
    D_slope_ps_nm2_km=0.056,   # 色散斜率 [ps/(nm²·km)]
    L_km=50.0,                 # 光纤长度 [km]
    A_eff=80e-12,              # 有效模场面积 [m²]
    rayleigh_coeff=4.8e-8,     # 瑞利散射系数 [1/m³]
    T_kelvin=300.0,            # 温度 [K]
)

# C 波段 WDM 网格（80 信道，50 GHz 间隔，中心 193.4 THz，~191.4–195.4 THz）
WDM_PARAMS = dict(
    f_center=193.4e12,         # C 波段中心频率 [Hz]
    N_ch=80,                   # C 波段总信道数（50 GHz 间隔，~191.4–195.4 THz）
    channel_spacing=50e9,     # 信道间隔 [Hz]
    B_s=32e9,                  # 信号带宽 [Hz]
    P0=1e-3,                   # 经典信道发射功率 [W] (0 dBm)
)

# -----------------------------------------------------------------------------
# 预设经典信道位置（取消注释其中一个，或自定义 CLASSICAL_INDICES）
# 物理假设：经典信道与量子信道同向传播（co-propagating）
# 前向噪声 → 接收端 z=L（Bob）；后向噪声 → 发射端 z=0（Alice）
# -----------------------------------------------------------------------------

# --- 1 个经典信道 ---
# CLASSICAL_INDICES = [40]              # 居中（193.4 THz）
# CLASSICAL_INDICES = [1]              # 居左（低频端）

# --- 3 个经典信道 ---
CLASSICAL_INDICES = [39, 40, 41]       # 居中（C 波段中间）
# CLASSICAL_INDICES = [1, 2, 3]        # 居左（低频端）

# --- 24 个经典信道 ---
# CLASSICAL_INDICES = list(range(28, 52))   # 居中（连续 24 个）
# CLASSICAL_INDICES = list(range(0, 24))    # 居左

# -----------------------------------------------------------------------------

RUN_TAG = ""      # 留空时自动生成为 discrete_N{N_ch}_C{n_classical}
SAVE_CSV = True   # True：保存 CSV 文件（for Origin 绘图）

# ===========================================================================
# 核心函数
# ===========================================================================

def compute_all_quantum_noise(
    fiber: Fiber,
    wdm_params: dict,
    classical_indices: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray], WDMGrid]:
    """一次性计算 C 波段全部量子信道的 SpRS 和 FWM 噪声。

    Parameters
    ----------
    classical_indices : list[int]
        经典信道索引（0-based）；其余信道自动设为量子信道。
        物理假设：经典/量子信道同向传播，泵浦总功率 = wdm_params['P0']。

    Returns
    -------
    f_q_hz : ndarray, shape (N_q,)
        量子信道中心频率 [Hz]
    noise_dict : dict, 每值 shape (N_q,)
        'sprs_fwd', 'sprs_bwd', 'fwm_fwd', 'fwm_bwd'
    wdm_grid_ref : WDMGrid
        参考 WDMGrid（含全部信道配置，用于信号功率谱图）
    """
    N_ch = wdm_params["N_ch"]
    q_indices = [i for i in range(N_ch) if i not in classical_indices]
    assert len(q_indices) > 0, "至少需要 1 个量子信道"
    assert len(classical_indices) > 0, "至少需要 1 个经典信道（泵浦）"

    cfg = WDMConfig(**wdm_params, quantum_channel_indices=q_indices)
    grid = build_wdm_grid(cfg, spectrum_type=SpectrumType.SINGLE_FREQ)

    result = compute_noise(
        "all", fiber, grid,
        sprs_solver=DiscreteSPRSSolver(),
        fwm_solver=DiscreteFWMSolver(),
    )

    f_q_hz = np.array([ch.f_center for ch in grid.get_quantum_channels()])
    return f_q_hz, result, grid


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("离散噪声功率谱生成脚本（C 波段）")
    print("=" * 60)

    # 构造光纤
    fiber = Fiber(FiberConfig(**FIBER_PARAMS))
    print(f"\n光纤参数：")
    print(f"  L = {FIBER_PARAMS['L_km']} km")
    print(f"  α = {FIBER_PARAMS['alpha_dB_per_km']} dB/km → {fiber.alpha:.4e} /m")
    print(f"  γ = {FIBER_PARAMS['gamma_per_W_km']} 1/(W·km) → {fiber.gamma:.4e} 1/(W·m)")
    print(f"  D = {FIBER_PARAMS['D_ps_nm_km']} ps/(nm·km)")
    print(f"  T = {FIBER_PARAMS['T_kelvin']} K")

    N_ch = WDM_PARAMS["N_ch"]
    n_classical = len(CLASSICAL_INDICES)
    n_quantum = N_ch - n_classical

    print(f"\nWDM 参数（C 波段）：")
    print(f"  N_ch = {N_ch}，间隔 = {WDM_PARAMS['channel_spacing']/1e9:.0f} GHz")
    print(f"  f_center = {WDM_PARAMS['f_center']/1e12:.1f} THz")
    print(f"  P0 = {WDM_PARAMS['P0']*1e3:.1f} mW ({10*np.log10(WDM_PARAMS['P0']*1e3):.1f} dBm)")

    # C 波段覆盖率（~5 THz = 191.4–196.4 THz）
    c_band_coverage = N_ch * WDM_PARAMS["channel_spacing"] / 5e12 * 100
    print(f"\nC 波段覆盖：{N_ch} 信道 × 50 GHz = {c_band_coverage:.0f}% （C 波段 ~5 THz）")
    print(f"  经典信道：{n_classical} 个（索引 {CLASSICAL_INDICES}）")
    print(f"  量子信道：{n_quantum} 个")

    print(f"\n计算量子信道噪声（共 {n_quantum} 个）...")
    f_q_hz, noise_dict, wdm_grid_ref = compute_all_quantum_noise(
        fiber, WDM_PARAMS, CLASSICAL_INDICES
    )

    # 打印各噪声分量范围
    print(f"\n噪声功率范围：")
    for key in ["sprs_fwd", "sprs_bwd", "fwm_fwd", "fwm_bwd"]:
        vals = noise_dict[key]
        print(f"  {key:12s}：min={vals.min():.3e} W，max={vals.max():.3e} W")

    # 输出目录
    tag = RUN_TAG or f"discrete_N{N_ch}_C{n_classical}"
    output_dir = _PROJECT_ROOT / "outputs" / tag
    print(f"\n输出目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n生成噪声功率谱图（8 PNG + CSV）...")
    figs = make_noise_figures(
        f_q_hz=f_q_hz,
        noise_dict=noise_dict,
        wdm_grid_ref=wdm_grid_ref,
        output_dir=output_dir,
        dpi=150,
        discrete=True,
        save_csv=SAVE_CSV,
    )
    print(f"  已生成 {len(figs)} 张图 + {('2 CSV' if SAVE_CSV else '0 CSV')}")

    # 打印文件列表
    if SAVE_CSV:
        csv_files = list(output_dir.glob("*.csv"))
        print(f"\nCSV 文件：")
        for f in csv_files:
            print(f"  {f.name}")

    print("\n完成。")


if __name__ == "__main__":
    main()
