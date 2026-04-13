from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from qkd_sim.config.schema import WDMConfig
from qkd_sim.physical.signal import build_wdm_grid

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# WDM 参数从 YAML 加载，保持唯一真值
_CONFIG_DIR = _PROJECT_ROOT / "src" / "qkd_sim" / "config"
_YAML_WDM_PATH = _CONFIG_DIR / "defaults" / "wdm_para" / "wdm_100ghz.yaml"


def _load_wdm_params() -> dict:
    from qkd_sim.config.schema import load_wdm_config
    cfg = load_wdm_config(_YAML_WDM_PATH)
    return dict(
        start_freq=cfg.start_freq,
        start_channel=cfg.start_channel,
        end_channel=cfg.end_channel,
        channel_spacing=cfg.channel_spacing,
        B_s=cfg.B_s,
        P0=cfg.P0,
        beta_rolloff=cfg.beta_rolloff,
        # quantum_channel_indices 由 _build_wdm_config 单独传入，不加入此 dict
    )


WDM_PARAMS = _load_wdm_params()
CLASSICAL_INDICES = [38, 39, 40]
NOISE_GRID_RESOLUTION_HZ = 5e9
NOISE_FLOOR_W = 1e-23
FREQ_GRID_PADDING_FACTOR = 1.5
LENGTHS_KM = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200])
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

# --- Power override for classical channel launch power ---
_POWER_OVERRIDE_DBM: Optional[float] = None


def set_power_override(dbm: Optional[float]) -> None:
    """Override P0 (classical channel launch power in dBm).

    Set to None to revert to YAML-configured default.
    """
    global _POWER_OVERRIDE_DBM
    _POWER_OVERRIDE_DBM = dbm


def _get_P0() -> float:
    """Return effective P0 in linear Watts."""
    if _POWER_OVERRIDE_DBM is not None:
        return 1e-3 * 10 ** (_POWER_OVERRIDE_DBM / 10.0)
    return WDM_PARAMS["P0"]


_LEGEND_SYNC_JS = """
(function() {
    function getLegendGroup(el, curveNumber) {
        var fd = el._fullData || [];
        return (fd[curveNumber] && fd[curveNumber].legendgroup) ? fd[curveNumber].legendgroup : null;
    }

    function getLegendGroups(el) {
        var groups = [];
        var seen = {};
        var data = el.data || [];
        for (var i = 0; i < data.length; i++) {
            var grp = data[i].legendgroup;
            if (!grp || seen[grp]) { continue; }
            seen[grp] = true;
            groups.push(grp);
        }
        return groups;
    }

    function setAllGroupsVisible(el) {
        var data = el.data || [];
        for (var i = 0; i < data.length; i++) {
            if (data[i].legendgroup) {
                data[i].visible = true;
            }
        }
    }

    function isolateGroup(el, targetGroup) {
        var data = el.data || [];
        for (var i = 0; i < data.length; i++) {
            var grp = data[i].legendgroup;
            if (!grp) { continue; }
            data[i].visible = (grp === targetGroup) ? true : 'legendonly';
        }
    }

    function isGroupIsolated(el, targetGroup) {
        var groups = getLegendGroups(el);
        if (groups.length <= 1) { return false; }

        var visibleGroups = {};
        var data = el.data || [];
        for (var i = 0; i < data.length; i++) {
            var grp = data[i].legendgroup;
            if (!grp || data[i].visible === 'legendonly') { continue; }
            visibleGroups[grp] = true;
        }

        var visibleCount = 0;
        for (var j = 0; j < groups.length; j++) {
            if (visibleGroups[groups[j]]) {
                visibleCount += 1;
            }
        }

        return visibleCount === 1 && !!visibleGroups[targetGroup];
    }

    function attachLegendSync() {
        var el = document.querySelector('.js-plotly-plot');
        if (!el) { setTimeout(attachLegendSync, 500); return; }
        if (el.dataset.legendSyncAttached === '1') { return; }
        el.dataset.legendSyncAttached = '1';

        el.on('plotly_legendclick', function(ev) {
            var grp = getLegendGroup(el, ev.curveNumber);
            if (!grp) { return true; }

            var on = false;
            var data = el.data || [];
            for (var i = 0; i < data.length; i++) {
                if (data[i].legendgroup === grp && data[i].visible !== 'legendonly') {
                    on = true;
                    break;
                }
            }

            var nv = on ? 'legendonly' : true;
            for (var j = 0; j < data.length; j++) {
                if (data[j].legendgroup === grp) {
                    data[j].visible = nv;
                }
            }
            Plotly.redraw(el);
            return false;
        });

        el.on('plotly_legenddoubleclick', function(ev) {
            var grp = getLegendGroup(el, ev.curveNumber);
            if (!grp) { return true; }

            var groups = getLegendGroups(el);
            if (groups.length <= 1) { return false; }

            if (isGroupIsolated(el, grp)) {
                setAllGroupsVisible(el);
            } else {
                isolateGroup(el, grp);
            }

            Plotly.redraw(el);
            return false;
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachLegendSync);
    } else { attachLegendSync(); }
})();
"""


def _resolve_osa_csv() -> Path:
    csv_files = sorted(OSA_CSV_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OSA CSV in {OSA_CSV_PATH}")
    return csv_files[0]


def _build_noise_frequency_grid(config: WDMConfig) -> np.ndarray:
    half_span = (config.end_channel - config.start_channel) / 2.0 * config.channel_spacing
    center_freq = config.start_freq + half_span
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = center_freq - half_span - padding
    f_max = center_freq + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / NOISE_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_wdm_config(quantum_indices: list[int]) -> WDMConfig:
    return WDMConfig(**WDM_PARAMS, quantum_channel_indices=list(quantum_indices), P0=_get_P0())


def _build_model_grid(
    model_key: str,
    spec: dict,
    base_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
):
    _ = model_key
    from qkd_sim.physical.signal import SpectrumType

    if spec["beta_rolloff"] is not None:
        model_config = WDMConfig(
            start_freq=base_config.start_freq,
            start_channel=base_config.start_channel,
            end_channel=base_config.end_channel,
            channel_spacing=base_config.channel_spacing,
            B_s=base_config.B_s,
            P0=_get_P0(),
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


def _display_channel_label(channel_index: int) -> str:
    return f"C{channel_index + WDM_PARAMS['start_channel']}"


def adaptive_log_ticks(
    y_bot_log: float, y_top_log: float, max_ticks: int = 8
) -> dict:
    return dict(
        tickmode="auto",
        nticks=max_ticks,
        tickformat="1e",
        exponentformat="power",
    )


def adaptive_linear_ticks(
    y_bot: float, y_top: float, max_ticks: int = 8
) -> dict:
    return dict(
        tickmode="auto",
        nticks=max_ticks,
        tickformat=".2f",
    )


def _build_all_classical_grid(
    spec: dict,
    base_config: WDMConfig,
    f_grid: np.ndarray,
    osa_csv_path: Path,
):
    from qkd_sim.physical.signal import SpectrumType

    model_config = WDMConfig(
        start_freq=base_config.start_freq,
        start_channel=base_config.start_channel,
        end_channel=base_config.end_channel,
        channel_spacing=base_config.channel_spacing,
        B_s=base_config.B_s,
        P0=_get_P0(),
        beta_rolloff=base_config.beta_rolloff if spec["beta_rolloff"] is None else spec["beta_rolloff"],
        quantum_channel_indices=[],
    )
    all_indices = list(range(int(base_config.end_channel - base_config.start_channel + 1)))
    if spec["spectrum_type"] == SpectrumType.OSA_SAMPLED:
        return build_wdm_grid(
            config=model_config,
            spectrum_type=spec["spectrum_type"],
            f_grid=f_grid,
            osa_csv_path=osa_csv_path,
            osa_rbw=OSA_RBW_HZ,
            classical_channel_indices=all_indices,
        )
    return build_wdm_grid(
        config=model_config,
        spectrum_type=spec["spectrum_type"],
        f_grid=f_grid,
        classical_channel_indices=all_indices,
    )


def _make_fiber(fiber_params: dict, length_km: float):
    from qkd_sim.config.schema import FiberConfig
    from qkd_sim.physical.fiber import Fiber

    params = dict(fiber_params)
    params["L_km"] = float(length_km)
    return Fiber(FiberConfig(**params))


def _integrate_signal_per_channel(grid, f_grid: np.ndarray | None) -> np.ndarray:
    powers = np.zeros(len(grid.channels), dtype=np.float64)
    if f_grid is None or len(f_grid) < 2:
        for idx, ch in enumerate(grid.channels):
            if ch.channel_type == "classical":
                powers[idx] = float(ch.power)
        return powers

    df = float(np.mean(np.diff(f_grid)))
    from qkd_sim.physical.signal import SpectrumType
    for idx, ch in enumerate(grid.channels):
        if ch.channel_type == "classical":
            if ch.spectrum_type == SpectrumType.SINGLE_FREQ:
                # Delta approximation: sum(G*df) = P at nearest bin → G = P/df
                powers[idx] = float(ch.power)
            else:
                powers[idx] = float(np.sum(ch.get_psd(f_grid)) * df)
    return powers


def _integrate_noise_psd_per_channel(
    psd_fwd: np.ndarray,
    psd_bwd: np.ndarray,
    grid,
    f_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """积分连续 PSD 到各量子信道带宽。

    compute_forward_conti 返回完整 PSD (N_f,)，需积分到每个量子信道带宽
    得到各量子信道的积分噪声功率 (N_q,)。

    Parameters
    ----------
    psd_fwd, psd_bwd : ndarray (N_f,)
        前向/后向噪声 PSD
    grid : WDMGrid
    f_grid : ndarray (N_f,)

    Returns
    -------
    (N_q,) 前向/后向积分噪声功率
    """
    quantum_chs = grid.get_quantum_channels()
    n_q = len(quantum_chs)
    df = float(np.mean(np.diff(f_grid)))
    fwd_integrated = np.zeros(n_q, dtype=np.float64)
    bwd_integrated = np.zeros(n_q, dtype=np.float64)

    for i, ch in enumerate(quantum_chs):
        f_lo = ch.f_center - ch.B_s / 2.0
        f_hi = ch.f_center + ch.B_s / 2.0
        mask = (f_grid >= f_lo) & (f_grid < f_hi)
        fwd_integrated[i] = float(np.sum(psd_fwd[mask]) * df)
        bwd_integrated[i] = float(np.sum(psd_bwd[mask]) * df)

    return fwd_integrated, bwd_integrated


def _compute_noise_power_pair(
    noise_type: str,
    fiber,
    grid,
    continuous: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 FWM/SpRS 噪声积分功率 (N_q,)。

    连续模型：调用 compute_forward_conti/compute_backward_conti，返回 (N_q,) 积分功率
    离散模型：调用 compute_forward/compute_backward，返回 (N_q,) 积分功率

    Args:
        noise_type: "fwm" | "sprs" | "both"
        fiber: Fiber instance
        grid: WDMGrid
        continuous: 是否为连续模型

    Returns
    -------
    (fwd, bwd), each shape (N_q,) — per-channel integrated noise power [W]
    """
    from qkd_sim.physical.noise import DiscreteFWMSolver, DiscreteSPRSSolver

    n_q = len(grid.get_quantum_channels())
    fwd = np.zeros(n_q, dtype=np.float64)
    bwd = np.zeros(n_q, dtype=np.float64)
    _f_grid = grid.f_grid

    def _call_solver(solver):
        if continuous:
            # compute_forward_conti 返回 (N_q,) 积分功率，不是 (N_f,)
            return (
                np.asarray(solver.compute_forward_conti(fiber, grid, _f_grid), dtype=np.float64),
                np.asarray(solver.compute_backward_conti(fiber, grid, _f_grid), dtype=np.float64),
            )
        # 离散模型：返回 (N_q,) 积分功率
        return (
            np.asarray(solver.compute_forward(fiber, grid), dtype=np.float64),
            np.asarray(solver.compute_backward(fiber, grid), dtype=np.float64),
        )

    if noise_type in ("fwm", "both"):
        solver = DiscreteFWMSolver()
        f_i, b_i = _call_solver(solver)
        fwd += f_i
        bwd += b_i

    if noise_type in ("sprs", "both"):
        solver = DiscreteSPRSSolver()
        f_i, b_i = _call_solver(solver)
        fwd += f_i
        bwd += b_i

    return fwd, bwd


def _compute_noise_spectrum_pair(
    noise_type: str,
    fiber,
    grid,
    f_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回完整 (N_f,) PSD 数组，仅适用于连续模型。

    fwm: compute_fwm_spectrum_conti
    sprs: compute_sprs_spectrum_conti
    both: 两者逐点相加

    Returns
    -------
    (fwd, bwd), each shape (N_f,)
    """
    from qkd_sim.physical.noise import DiscreteFWMSolver, DiscreteSPRSSolver

    fwd = np.zeros_like(f_grid, dtype=np.float64)
    bwd = np.zeros_like(f_grid, dtype=np.float64)

    if noise_type in ("fwm", "both"):
        solver = DiscreteFWMSolver()
        fwd += solver.compute_fwm_spectrum_conti(fiber, grid, f_grid, direction="forward")
        bwd += solver.compute_fwm_spectrum_conti(fiber, grid, f_grid, direction="backward")

    if noise_type in ("sprs", "both"):
        solver = DiscreteSPRSSolver()
        df_grid = float(np.mean(np.diff(f_grid)))
        fwd += solver.compute_sprs_spectrum_conti(fiber, grid, f_grid, direction="forward") / df_grid
        bwd += solver.compute_sprs_spectrum_conti(fiber, grid, f_grid, direction="backward") / df_grid

    return np.asarray(fwd, dtype=np.float64), np.asarray(bwd, dtype=np.float64)


def _compute_nli_pair(fiber, grid) -> tuple[np.ndarray, np.ndarray]:
    try:
        from qkd_sim.physical.noise import GNModelSolver
        gn_solver = GNModelSolver()
    except ImportError:
        n_ch = len(grid.channels)
        return np.zeros(n_ch, dtype=np.float64), np.zeros(n_ch, dtype=np.float64)

    result = gn_solver.compute_nli_per_channel(fiber, grid, grid.f_grid)
    return (
        np.asarray(result["nli_fwd"], dtype=np.float64),
        np.asarray(result["nli_bwd"], dtype=np.float64),
    )


def precompute_by_length(
    noise_type: str,
    specs: dict,
    LENGTHS_KM: np.ndarray,
    base_config: WDMConfig,
    noise_f_grid: np.ndarray,
    osa_csv_path: Path,
    fiber_params: dict,
) -> tuple[dict, list]:
    """Precompute noise vs fiber length for all quantum channels.

    Returns (ALL_BY_LEN, VALID_Q_INDICES)
    ALL_BY_LEN[q_idx][model_key] = {"fwd": np.array(N_L), "bwd": np.array(N_L)}
    VALID_Q_INDICES: list of q_idx that have non-zero noise for at least one model
    """
    model_keys = get_noise_model_keys(noise_type)
    quantum_indices = list(base_config.quantum_channel_indices)
    n_q = len(quantum_indices)
    n_l = len(LENGTHS_KM)
    n_ch = int(base_config.end_channel - base_config.start_channel + 1)

    if noise_type == "with_signal":
        # Continuous PSD: noise (FWM+SpRS) + signal, integrate to power per length
        df = float(np.mean(np.diff(noise_f_grid)))
        all_by_len = {
            ch_idx: {mk: {"fwd": np.zeros(n_l), "bwd": np.zeros(n_l)} for mk in model_keys}
            for ch_idx in range(n_ch)
        }
        for model_key in model_keys:
            spec = specs[model_key]
            # Full grid with all classical channels (raised-cosine spectrum)
            grid_all = _build_all_classical_grid(
                dict(spec, beta_rolloff=WDM_PARAMS["beta_rolloff"]),
                base_config,
                noise_f_grid,
                osa_csv_path,
            )
            # Precompute signal PSD on the full frequency grid (sum of all classical channels)
            signal_psd = np.zeros_like(noise_f_grid, dtype=np.float64)
            for ch in grid_all.channels:
                if ch.channel_type == "classical":
                    signal_psd += ch.get_psd(noise_f_grid)

            for li, length_km in enumerate(LENGTHS_KM):
                fiber = _make_fiber(fiber_params, length_km)
                # Noise PSD: FWM + SpRS on the full grid
                noise_fwd_psd, noise_bwd_psd = _compute_noise_spectrum_pair(
                    "both", fiber, grid_all, noise_f_grid
                )
                total_psd_fwd = noise_fwd_psd + signal_psd
                total_psd_bwd = noise_bwd_psd + signal_psd
                # Integrate over full grid to get total power [W]
                total_fwd_power = float(np.sum(total_psd_fwd) * df)
                total_bwd_power = float(np.sum(total_psd_bwd) * df)
                for ch_idx in range(n_ch):
                    all_by_len[ch_idx][model_key]["fwd"][li] = total_fwd_power
                    all_by_len[ch_idx][model_key]["bwd"][li] = total_bwd_power

        valid_indices = [
            ch_idx
            for ch_idx in range(n_ch)
            if any(
                np.any(all_by_len[ch_idx][mk]["fwd"] > 0)
                or np.any(all_by_len[ch_idx][mk]["bwd"] > 0)
                for mk in model_keys
            )
        ]
        return all_by_len, valid_indices

    if noise_type == "only_signal":
        all_by_len = {
            ch_idx: {mk: {"fwd": np.zeros(n_l), "bwd": np.zeros(n_l)} for mk in model_keys}
            for ch_idx in range(n_ch)
        }
        for model_key in model_keys:
            spec = specs[model_key]
            grid_all = _build_all_classical_grid(spec, base_config, noise_f_grid, osa_csv_path)
            signal = _integrate_signal_per_channel(grid_all, grid_all.f_grid)
            classical_mask = np.array([ch.channel_type == "classical" for ch in grid_all.channels], dtype=bool)

            for li, length_km in enumerate(LENGTHS_KM):
                fiber = _make_fiber(fiber_params, length_km)
                nli_fwd, nli_bwd = _compute_nli_pair(fiber, grid_all)
                fwd = np.zeros(n_ch, dtype=np.float64)
                bwd = np.zeros(n_ch, dtype=np.float64)
                fwd[classical_mask] = nli_fwd + signal[classical_mask]
                bwd[classical_mask] = nli_bwd + signal[classical_mask]

                for ch_idx in range(n_ch):
                    all_by_len[ch_idx][model_key]["fwd"][li] = float(fwd[ch_idx])
                    all_by_len[ch_idx][model_key]["bwd"][li] = float(bwd[ch_idx])

        valid_indices = [
            ch_idx
            for ch_idx in range(n_ch)
            if any(
                np.any(all_by_len[ch_idx][mk]["fwd"] > 0)
                or np.any(all_by_len[ch_idx][mk]["bwd"] > 0)
                for mk in model_keys
            )
        ]
        return all_by_len, valid_indices

    all_by_len = {
        q_local_idx: {mk: {"fwd": np.zeros(n_l), "bwd": np.zeros(n_l)} for mk in model_keys}
        for q_local_idx in range(n_q)
    }
    for q_local_idx, q_idx in enumerate(quantum_indices):
        single_q_config = _build_wdm_config([q_idx])
        for model_key in model_keys:
            spec = specs[model_key]
            grid = _build_model_grid(
                model_key,
                spec,
                single_q_config,
                noise_f_grid,
                osa_csv_path,
            )
            for li, length_km in enumerate(LENGTHS_KM):
                fiber = _make_fiber(fiber_params, length_km)
                fwd, bwd = _compute_noise_power_pair(
                    noise_type,
                    fiber,
                    grid,
                    continuous=bool(spec["continuous"]),
                )
                if len(fwd) > 0:
                    all_by_len[q_local_idx][model_key]["fwd"][li] = float(fwd[0])
                    all_by_len[q_local_idx][model_key]["bwd"][li] = float(bwd[0])

    valid_q_indices = [
        q_local_idx
        for q_local_idx in range(n_q)
        if any(
            np.any(all_by_len[q_local_idx][mk]["fwd"] > 0)
            or np.any(all_by_len[q_local_idx][mk]["bwd"] > 0)
            for mk in model_keys
        )
    ]
    return all_by_len, valid_q_indices


def precompute_by_channel(
    noise_type: str,
    specs: dict,
    LENGTHS_KM: np.ndarray,
    base_config: WDMConfig,
    noise_f_grid: np.ndarray,
    osa_csv_path: Path,
    fiber_params: dict,
) -> tuple[dict, list]:
    """Precompute noise vs quantum channel frequency for all lengths.

    Returns (ALL_BY_CH, VALID_L_INDICES)

    ALL_BY_CH[L_idx][model_key] = {
        "fwd": np.ndarray,       # (N_f,) for continuous PSD, (N_q,) for discrete channel power
        "bwd": np.ndarray,
        "x": np.ndarray,         # noise_f_grid for continuous, channel center freqs for discrete
        "x_kind": str,           # "frequency_grid" | "channel_center"
        "y_kind": str,           # "psd" | "channel_power"
    }
    VALID_L_INDICES: list of L indices with non-zero noise
    """
    model_keys = get_noise_model_keys(noise_type)
    quantum_indices = list(base_config.quantum_channel_indices)
    n_q = len(quantum_indices)
    n_l = len(LENGTHS_KM)
    n_ch = int(base_config.end_channel - base_config.start_channel + 1)

    if noise_type == "with_signal":
        # Continuous PSD: noise (FWM+SpRS) + signal, output (N_f,) per length
        df = float(np.mean(np.diff(noise_f_grid)))
        all_by_ch = {
            li: {mk: {
                "fwd": np.zeros(len(noise_f_grid)),
                "bwd": np.zeros(len(noise_f_grid)),
                "x": np.asarray(noise_f_grid, dtype=np.float64),
                "x_kind": "frequency_grid",
                "y_kind": "power_per_bin",
            } for mk in model_keys}
            for li in range(n_l)
        }
        for li, length_km in enumerate(LENGTHS_KM):
            fiber = _make_fiber(fiber_params, length_km)
            for model_key in model_keys:
                spec = specs[model_key]
                # Full grid with all classical channels (raised-cosine spectrum)
                grid_all = _build_all_classical_grid(
                    dict(spec, beta_rolloff=WDM_PARAMS["beta_rolloff"]),
                    base_config,
                    noise_f_grid,
                    osa_csv_path,
                )
                # Signal PSD: sum raised-cosine PSD of all classical channels
                signal_psd = np.zeros_like(noise_f_grid, dtype=np.float64)
                for ch in grid_all.channels:
                    if ch.channel_type == "classical":
                        signal_psd += ch.get_psd(noise_f_grid)
                # Noise PSD: FWM + SpRS on the full grid
                noise_fwd_psd, noise_bwd_psd = _compute_noise_spectrum_pair(
                    "both", fiber, grid_all, noise_f_grid
                )
                total_fwd = (noise_fwd_psd + signal_psd) * df
                total_bwd = (noise_bwd_psd + signal_psd) * df
                all_by_ch[li][model_key]["fwd"] = np.asarray(total_fwd, dtype=np.float64)
                all_by_ch[li][model_key]["bwd"] = np.asarray(total_bwd, dtype=np.float64)

        valid_l_indices = [
            li
            for li in range(n_l)
            if any(
                np.any(all_by_ch[li][mk]["fwd"] > 0)
                or np.any(all_by_ch[li][mk]["bwd"] > 0)
                for mk in model_keys
            )
        ]
        return all_by_ch, valid_l_indices

    if noise_type == "only_signal":
        # x-axis: all channel center frequencies (classical + quantum)
        all_ch_freqs = np.array(
            [
                WDM_PARAMS["start_freq"] + idx * WDM_PARAMS["channel_spacing"]
                for idx in range(n_ch)
            ],
            dtype=np.float64,
        )
        all_by_ch = {
            li: {mk: {
                "fwd": np.zeros(n_ch),
                "bwd": np.zeros(n_ch),
                "x": np.asarray(all_ch_freqs, dtype=np.float64),
                "x_kind": "channel_center",
                "y_kind": "channel_power",
            } for mk in model_keys}
            for li in range(n_l)
        }
        for li, length_km in enumerate(LENGTHS_KM):
            fiber = _make_fiber(fiber_params, length_km)
            for model_key in model_keys:
                spec = specs[model_key]
                grid_all = _build_all_classical_grid(spec, base_config, noise_f_grid, osa_csv_path)
                signal = _integrate_signal_per_channel(grid_all, grid_all.f_grid)
                classical_mask = np.array([ch.channel_type == "classical" for ch in grid_all.channels], dtype=bool)

                fwd = np.zeros(n_ch, dtype=np.float64)
                bwd = np.zeros(n_ch, dtype=np.float64)
                nli_fwd, nli_bwd = _compute_nli_pair(fiber, grid_all)
                fwd[classical_mask] = nli_fwd + signal[classical_mask]
                bwd[classical_mask] = nli_bwd + signal[classical_mask]

                all_by_ch[li][model_key]["fwd"] = np.asarray(fwd, dtype=np.float64)
                all_by_ch[li][model_key]["bwd"] = np.asarray(bwd, dtype=np.float64)

        valid_l_indices = [
            li
            for li in range(n_l)
            if any(
                np.any(all_by_ch[li][mk]["fwd"] > 0)
                or np.any(all_by_ch[li][mk]["bwd"] > 0)
                for mk in model_keys
            )
        ]
        return all_by_ch, valid_l_indices

    # Precompute quantum channel center frequencies for discrete-model x-axis
    q_center_freqs = np.array(
        [
            WDM_PARAMS["start_freq"] + idx * WDM_PARAMS["channel_spacing"]
            for idx in quantum_indices
        ],
        dtype=np.float64,
    )

    # Initialize all_by_ch with metadata structure
    all_by_ch: dict = {}
    for li in range(n_l):
        all_by_ch[li] = {}
        for mk in model_keys:
            all_by_ch[li][mk] = {
                "fwd": np.array([]),
                "bwd": np.array([]),
                "x": np.array([]),
                "x_kind": "",
                "y_kind": "",
            }

    for li, length_km in enumerate(LENGTHS_KM):
        fiber = _make_fiber(fiber_params, length_km)
        for model_key in model_keys:
            spec = specs[model_key]

            if spec["continuous"]:
                # Continuous model: full PSD spectrum (N_f,) on noise_f_grid
                grid = _build_model_grid(
                    model_key,
                    spec,
                    base_config,  # Use full quantum channel config
                    noise_f_grid,
                    osa_csv_path,
                )
                df = float(np.mean(np.diff(noise_f_grid)))
                fwd_psd, bwd_psd = _compute_noise_spectrum_pair(
                    noise_type, fiber, grid, noise_f_grid
                )
                all_by_ch[li][model_key] = {
                    "fwd": np.asarray(fwd_psd * df, dtype=np.float64),
                    "bwd": np.asarray(bwd_psd * df, dtype=np.float64),
                    "x": np.asarray(noise_f_grid, dtype=np.float64),
                    "x_kind": "frequency_grid",
                    "y_kind": "power_per_bin",
                }
            else:
                # Discrete model: per-channel integrated power (N_q,) at channel centers
                fwd_arr = np.zeros(n_q, dtype=np.float64)
                bwd_arr = np.zeros(n_q, dtype=np.float64)
                for q_local_idx, q_idx in enumerate(quantum_indices):
                    grid = _build_model_grid(
                        model_key,
                        spec,
                        _build_wdm_config([q_idx]),
                        noise_f_grid,
                        osa_csv_path,
                    )
                    fwd, bwd = _compute_noise_power_pair(
                        noise_type,
                        fiber,
                        grid,
                        continuous=False,
                    )
                    if len(fwd) > 0:
                        fwd_arr[q_local_idx] = float(fwd[0])
                        bwd_arr[q_local_idx] = float(bwd[0])
                all_by_ch[li][model_key] = {
                    "fwd": np.asarray(fwd_arr, dtype=np.float64),
                    "bwd": np.asarray(bwd_arr, dtype=np.float64),
                    "x": np.asarray(q_center_freqs, dtype=np.float64),
                    "x_kind": "channel_center",
                    "y_kind": "channel_power",
                }

    valid_l_indices = [
        li
        for li in range(n_l)
        if any(
            np.any(all_by_ch[li][mk]["fwd"] > 0)
            or np.any(all_by_ch[li][mk]["bwd"] > 0)
            for mk in model_keys
        )
    ]
    return all_by_ch, valid_l_indices


def get_noise_model_keys(noise_type: str) -> list[str]:
    """Return model keys for the given noise_type."""
    _ = noise_type
    try:
        from qkd_sim.config.plot_config import load_model_specs

        return list(load_model_specs("fwm_noise").keys())
    except Exception:
        return ["discrete", "osa"]
