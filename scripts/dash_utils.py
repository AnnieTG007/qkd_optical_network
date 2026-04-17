from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from typing import Optional

import numpy as np

from qkd_sim.config.schema import WDMConfig
from qkd_sim.physical.signal import build_wdm_grid

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# WDM 参数从 YAML 加载，保持唯一真值
_CONFIG_DIR = _PROJECT_ROOT / "src" / "qkd_sim" / "config"
_YAML_WDM_PATH = _CONFIG_DIR / "defaults" / "wdm_para" / "wdm_100ghz.yaml"
_YAML_FIBER_PATH = _CONFIG_DIR / "defaults" / "fiber_para" / "fiber_smf.yaml"


def _load_wdm_params() -> dict:
    from qkd_sim.config.schema import load_wdm_config
    cfg = load_wdm_config(_YAML_WDM_PATH)
    num_ch = cfg.num_channels
    if num_ch is None:
        num_ch = int(cfg.end_channel - cfg.start_channel + 1)
    return dict(
        start_freq=cfg.start_freq,
        start_channel=cfg.start_channel,
        end_channel=cfg.end_channel,
        channel_spacing=cfg.channel_spacing,
        B_s=cfg.B_s,
        P0=cfg.P0,
        beta_rolloff=cfg.beta_rolloff,
        num_channels=num_ch,
        # quantum_channel_indices 由 _build_wdm_config 单独传入，不加入此 dict
    )


WDM_PARAMS = _load_wdm_params()
CLASSICAL_INDICES = [38, 39, 40]
NOISE_GRID_RESOLUTION_HZ = 5e9
NOISE_FLOOR_W = 1e-23
FREQ_GRID_PADDING_FACTOR = 1.5
def _load_fiber_params() -> dict:
    from qkd_sim.config.schema import load_fiber_config
    cfg = load_fiber_config(_YAML_FIBER_PATH)
    return dict(
        alpha_dB_per_km=cfg.alpha_dB_per_km,
        gamma_per_W_km=cfg.gamma_per_W_km,
        D_ps_nm_km=cfg.D_ps_nm_km,
        D_slope_ps_nm2_km=cfg.D_slope_ps_nm2_km,
        L_km=cfg.L_km,
        A_eff=cfg.A_eff,
        rayleigh_coeff=cfg.rayleigh_coeff,
        T_kelvin=cfg.T_kelvin,
        length_km_samples=cfg.length_km_samples,
    )

_FIBER_CFG = _load_fiber_params()
FIBER_PARAMS = {k: v for k, v in _FIBER_CFG.items() if k != 'length_km_samples'}
LENGTHS_KM = np.array(_FIBER_CFG['length_km_samples'])
OSA_RBW_HZ = 1.0e9
OSA_CSV_PATH = _PROJECT_ROOT / "data" / "osa"


def _build_caption() -> str:
    """Build figure caption string from WDM_PARAMS and FIBER_PARAMS.

    Returns a multi-line string showing:
    - Channel spacing, data rate, OSA RBW
    - Classical channel positions
    - Fiber parameters (alpha, gamma, D)
    """
    ch_spacing_ghz = WDM_PARAMS["channel_spacing"] / 1e9
    data_rate_gbaud = WDM_PARAMS["B_s"] / 1e9
    classical_freqs = [
        WDM_PARAMS["start_freq"] + idx * WDM_PARAMS["channel_spacing"]
        for idx in CLASSICAL_INDICES
    ]
    classical_labels = [
        f"C{CLASSICAL_INDICES[i] + int(WDM_PARAMS['start_channel'])}"
        for i in range(len(CLASSICAL_INDICES))
    ]
    classical_desc = ", ".join(
        f"{lbl} ({freq / 1e12:.3f} THz)"
        for lbl, freq in zip(classical_labels, classical_freqs)
    )
    return (
        f"Channel spacing: {ch_spacing_ghz:.0f} GHz | "
        f"Data rate: {data_rate_gbaud:.0f} GBaud | "
        f"OSA RBW: {OSA_RBW_HZ / 1e9:.0f} GHz | "
        f"Classical channels: {classical_desc}"
    )

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

# --- Multiprocessing for precomputation ---
# Disabled on Windows (nt): scipy Cython DLLs fail to load in spawned subprocesses.
# Enable via QKD_DASH_MP=1 env var on non-Windows systems for MP speedup.
_MP_ENABLED = os.environ.get("QKD_DASH_MP", "1") != "0" and os.name != "nt"
_PROFILE_ENABLED = os.environ.get("QKD_DASH_PROFILE", "1").lower() not in ("0", "false", "no")
_CSV_CACHE_ENABLED = os.environ.get("QKD_DASH_CSV_CACHE", "1").lower() not in ("0", "false", "no")
_PROFILE_INDENT = 0


def describe_compute_device() -> str:
    """Return the active compute backend used by Dash noise precomputation."""
    try:
        from qkd_sim.utils.gpu_utils import get_array_module, has_cupy

        if has_cupy():
            xp = get_array_module()
            device_id = xp.cuda.runtime.getDevice()
            props = xp.cuda.runtime.getDeviceProperties(device_id)
            name = props.get("name", b"CUDA GPU")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return f"CUDA GPU: {name} (CuPy; FWM continuous spectrum path)"
    except Exception as exc:
        return f"CPU (NumPy/SciPy; GPU detection failed: {exc})"
    return "CPU (NumPy/SciPy)"


def print_compute_device() -> None:
    """Print compute backend once at script startup."""
    print(f"Compute device: {describe_compute_device()}")


def set_csv_cache_enabled(enabled: bool) -> None:
    """Enable or disable Dash precompute CSV cache I/O."""
    global _CSV_CACHE_ENABLED
    _CSV_CACHE_ENABLED = bool(enabled)


@contextmanager
def profile_scope(label: str) -> Iterator[None]:
    """Print elapsed time for a named Dash precomputation step."""
    global _PROFILE_INDENT
    if not _PROFILE_ENABLED:
        yield
        return

    indent = "  " * _PROFILE_INDENT
    _PROFILE_INDENT += 1
    t_start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t_start
        _PROFILE_INDENT -= 1
        print(f"[profile] {indent}{label}: {elapsed:.3f}s")


def _get_mp_workers() -> int:
    """Number of worker processes for precomputation."""
    return max(1, os.cpu_count() or 1)


# --- Worker functions for multiprocessing (must be module-level for Windows spawn) ---

def _precompute_length_worker(
    power_dbm: float,
    noise_type: str,
    specs: dict,
    LENGTHS_KM: np.ndarray,
    base_config: WDMConfig,
    noise_f_grid: np.ndarray,
    osa_csv_path: Path,
    fiber_params: dict,
) -> tuple[float, dict, list]:
    """Worker: precompute by length for one power level. Runs in subprocess."""
    set_power_override(float(power_dbm))
    all_by_idx, valid_ch = precompute_by_length(
        noise_type, specs, LENGTHS_KM, base_config,
        noise_f_grid, osa_csv_path, fiber_params,
    )
    return float(power_dbm), all_by_idx, valid_ch


def _precompute_channel_worker(
    power_dbm: float,
    noise_type: str,
    specs: dict,
    LENGTHS_KM: np.ndarray,
    base_config: WDMConfig,
    noise_f_grid: np.ndarray,
    osa_csv_path: Path,
    fiber_params: dict,
) -> tuple[float, dict, list]:
    """Worker: precompute by channel for one power level. Runs in subprocess."""
    set_power_override(float(power_dbm))
    all_by_idx, valid_l = precompute_by_channel(
        noise_type, specs, LENGTHS_KM, base_config,
        noise_f_grid, osa_csv_path, fiber_params,
    )
    return float(power_dbm), all_by_idx, valid_l


# --- Power levels for startup precomputation (step=5 dBm, 7 values) ---
PRECOMPUTE_POWER_LEVELS = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]


# --- Clear stale CSV cache ---
def _clear_precomputed_csv_files(index_prefix: str | None = None) -> None:
    """Delete CSV files in precomputed dir to avoid stale data from changed params.

    index_prefix: if None, delete all; if "ch" or "len", only that subset.
    """
    if not _CSV_CACHE_ENABLED:
        return
    skipped: list[str] = []
    for f in _PRECOMPUTED_DIR.glob("*.csv"):
        if index_prefix is None or f.name.startswith(index_prefix):
            try:
                f.unlink()
            except PermissionError:
                skipped.append(f.name)
    if skipped:
        preview = ", ".join(skipped[:3])
        suffix = "" if len(skipped) <= 3 else f", ... +{len(skipped) - 3} more"
        print(f"[profile] warning: skipped {len(skipped)} locked CSV cache files: {preview}{suffix}")


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


# --- Power-based CSV caching for instant slider response ---
_PRECOMPUTED_DIR = _PROJECT_ROOT / "data" / "precomputed"
_POWER_CACHE: dict[float, dict] = {}  # power_dbm -> precomputed result dict


def _power_csv_path(noise_type: str, model_key: str, direction: str, power_dbm: float, index_prefix: str = "ch") -> Path:
    """CSV path for a specific (noise_type, model, direction, power, index_prefix) combination."""
    return _PRECOMPUTED_DIR / f"{index_prefix}_{noise_type}_{model_key}_{direction}_p{int(power_dbm):+d}.csv"


def _save_power_csv(data: np.ndarray, filepath: Path) -> None:
    """Save 2D power array to CSV (rows = outer index, cols = inner index)."""
    import csv

    if not _CSV_CACHE_ENABLED:
        return
    _PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow([f"{v:.6e}" for v in row])
    except PermissionError as exc:
        print(f"[profile] warning: skip CSV cache write for {filepath.name}: {exc}")


def _load_power_csv(filepath: Path) -> np.ndarray:
    """Load 2D power array from CSV."""
    import csv

    rows = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([float(x) for x in row])
    return np.array(rows, dtype=np.float64)


def _cache_precomputed_result(
    all_by_idx: dict, power_dbm: float, noise_type: str, model_keys: list[str], index_prefix: str = "ch"
) -> None:
    """Save precomputed result to CSV for each (model, direction) at given power.

    index_prefix: "ch" for precompute_by_channel (rows = length indices), "len" for precompute_by_length (rows = channel indices).
    """
    if not all_by_idx:
        return
    first_key = min(all_by_idx.keys())

    for mk in model_keys:
        ref_shape = all_by_idx[first_key][mk]["fwd"].shape
        if not all(all_by_idx[k][mk]["fwd"].shape == ref_shape for k in all_by_idx):
            continue  # Skip mixed-shape models
        for direction in ("fwd", "bwd"):
            data = np.array(
                [all_by_idx[k][mk][direction] for k in sorted(all_by_idx.keys())],
                dtype=np.float64,
            )
            _save_power_csv(data, _power_csv_path(noise_type, mk, direction, power_dbm, index_prefix))


def _load_cached_power(noise_type: str, model_keys: list[str], index_prefix: str = "ch") -> tuple[dict | None, float]:
    """Try to load cached precomputed result for current power override.

    index_prefix: "ch" for length-indexed data (precompute_by_channel), "len" for channel-indexed data (precompute_by_length).
    """
    power = _POWER_OVERRIDE_DBM if _POWER_OVERRIDE_DBM is not None else 0.0
    # Try exact power match in memory cache (distinguish by prefix)
    cache_key = (index_prefix, power)
    if cache_key in _POWER_CACHE:
        return _POWER_CACHE[cache_key], power
    # Try CSV files for this power (only for continuous models)
    # Find first continuous model as reference
    first_cont_mk = None
    for mk in model_keys:
        fwd_path = _power_csv_path(noise_type, mk, "fwd", power, index_prefix)
        if fwd_path.exists():
            first_cont_mk = mk
            break
    if first_cont_mk is None:
        return None, power  # No cached continuous models
    # Load continuous models from CSV
    # For "ch": outer index = length (n_l=18), data shape = (n_l, n_f)
    # For "len": outer index = channel (n_ch=61), data shape = (n_ch, n_l)
    # Determine outer dimension from the first CSV's row count
    ref_path = _power_csv_path(noise_type, first_cont_mk, "fwd", power, index_prefix)
    ref_data = _load_power_csv(ref_path)
    n_outer = ref_data.shape[0]

    all_by_idx = {}
    for idx in range(n_outer):
        all_by_idx[idx] = {}
        for mk in model_keys:
            fwd_path = _power_csv_path(noise_type, mk, "fwd", power, index_prefix)
            bwd_path = _power_csv_path(noise_type, mk, "bwd", power, index_prefix)
            if fwd_path.exists() and bwd_path.exists():
                fwd_data = _load_power_csv(fwd_path)
                bwd_data = _load_power_csv(bwd_path)
                all_by_idx[idx][mk] = {
                    "fwd": fwd_data[idx] if idx < len(fwd_data) else np.array([]),
                    "bwd": bwd_data[idx] if idx < len(bwd_data) else np.array([]),
                    "x": np.array([]),
                    "x_kind": "",
                    "y_kind": "",
                }
    _POWER_CACHE[cache_key] = all_by_idx
    return all_by_idx, power


def _scale_precomputed_result(all_by_idx: dict, scale: float) -> dict:
    """Scale cached fwd/bwd noise arrays while preserving x-axis metadata."""
    scaled: dict = {}
    for outer_idx, model_data in all_by_idx.items():
        scaled[outer_idx] = {}
        for model_key, entry in model_data.items():
            scaled[outer_idx][model_key] = {
                "fwd": np.asarray(entry.get("fwd", []), dtype=np.float64) * scale,
                "bwd": np.asarray(entry.get("bwd", []), dtype=np.float64) * scale,
                "x": np.asarray(entry.get("x", []), dtype=np.float64),
                "x_kind": entry.get("x_kind", ""),
                "y_kind": entry.get("y_kind", ""),
            }
    return scaled


def _combine_precomputed_results(
    fwm_by_idx: dict,
    sprs_by_idx: dict,
    fwm_scale: float,
    sprs_scale: float,
) -> dict:
    """Combine separately precomputed FWM and SpRS data for ``both`` plots."""
    combined: dict = {}
    for outer_idx in sorted(set(fwm_by_idx.keys()) & set(sprs_by_idx.keys())):
        combined[outer_idx] = {}
        model_keys = sorted(set(fwm_by_idx[outer_idx].keys()) & set(sprs_by_idx[outer_idx].keys()))
        for model_key in model_keys:
            fwm_entry = fwm_by_idx[outer_idx][model_key]
            sprs_entry = sprs_by_idx[outer_idx][model_key]
            combined[outer_idx][model_key] = {
                "fwd": (
                    np.asarray(fwm_entry.get("fwd", []), dtype=np.float64) * fwm_scale
                    + np.asarray(sprs_entry.get("fwd", []), dtype=np.float64) * sprs_scale
                ),
                "bwd": (
                    np.asarray(fwm_entry.get("bwd", []), dtype=np.float64) * fwm_scale
                    + np.asarray(sprs_entry.get("bwd", []), dtype=np.float64) * sprs_scale
                ),
                "x": np.asarray(fwm_entry.get("x", sprs_entry.get("x", [])), dtype=np.float64),
                "x_kind": fwm_entry.get("x_kind", sprs_entry.get("x_kind", "")),
                "y_kind": fwm_entry.get("y_kind", sprs_entry.get("y_kind", "")),
            }
    return combined


def _power_scaling_factor(noise_type: str, power_dbm: float) -> float | None:
    """Return exact pump-power scaling from the 0 dBm precompute."""
    if noise_type == "sprs":
        return float(10.0 ** (power_dbm / 10.0))
    if noise_type == "fwm":
        return float(10.0 ** (3.0 * power_dbm / 10.0))
    return None


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
    half_span = (config.num_channels - 1) / 2.0 * config.channel_spacing
    center_freq = config.start_freq + half_span
    padding = FREQ_GRID_PADDING_FACTOR * config.channel_spacing
    f_min = center_freq - half_span - padding
    f_max = center_freq + half_span + padding
    n_points = int(np.ceil((f_max - f_min) / NOISE_GRID_RESOLUTION_HZ)) + 1
    return np.linspace(f_min, f_max, n_points)


def _build_wdm_config(quantum_indices: list[int]) -> WDMConfig:
    params = {k: v for k, v in WDM_PARAMS.items() if k != "P0"}
    return WDMConfig(**params, quantum_channel_indices=list(quantum_indices), P0=_get_P0())


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
            channel_powers_W=base_config.channel_powers_W,
            num_channels=int(base_config.num_channels),
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
            modulation_format="16QAM",
        )
    return build_wdm_grid(
        config=model_config,
        spectrum_type=spec["spectrum_type"],
        f_grid=f_grid,
        classical_channel_indices=CLASSICAL_INDICES,
        modulation_format="16QAM" if spec["spectrum_type"] == SpectrumType.RAISED_COSINE else "OOK",
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
        quantum_channel_indices=base_config.quantum_channel_indices,
        channel_powers_W=base_config.channel_powers_W,
        num_channels=int(base_config.num_channels),
    )
    # Do NOT pass classical_channel_indices — derive it as complement of quantum_channel_indices
    # (avoids explicit overlap validation)
    if spec["spectrum_type"] == SpectrumType.OSA_SAMPLED:
        return build_wdm_grid(
            config=model_config,
            spectrum_type=spec["spectrum_type"],
            f_grid=f_grid,
            osa_csv_path=osa_csv_path,
            osa_rbw=OSA_RBW_HZ,
            modulation_format="16QAM",
        )
    return build_wdm_grid(
        config=model_config,
        spectrum_type=spec["spectrum_type"],
        f_grid=f_grid,
        modulation_format="16QAM" if spec["spectrum_type"] == SpectrumType.RAISED_COSINE else "OOK",
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
    L_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回完整 PSD 数组，适用于连续模型。

    当 L_arr=None 时返回 (N_f,)，当 L_arr 给出时返回 (N_f, N_L) —
    每列对应一个光纤长度。

    fwm: compute_fwm_spectrum_conti
    sprs: compute_sprs_spectrum_conti
    both: 两者逐点相加

    Returns
    -------
    (fwd, bwd), each shape (N_f,) or (N_f, N_L)
    """
    from qkd_sim.physical.noise import DiscreteFWMSolver, DiscreteSPRSSolver

    n_f = len(f_grid)
    n_l = 1 if L_arr is None else int(L_arr.shape[0])
    fwd = np.zeros((n_f, n_l), dtype=np.float64)
    bwd = np.zeros((n_f, n_l), dtype=np.float64)

    if noise_type in ("fwm", "both"):
        solver = DiscreteFWMSolver()
        fwm_fwd = solver.compute_fwm_spectrum_conti(
            fiber, grid, f_grid, direction="forward", L_arr=L_arr
        )
        fwm_bwd = solver.compute_fwm_spectrum_conti(
            fiber, grid, f_grid, direction="backward", L_arr=L_arr
        )
        if L_arr is None:
            fwd[:, 0] += fwm_fwd
            bwd[:, 0] += fwm_bwd
        else:
            fwd += fwm_fwd
            bwd += fwm_bwd

    if noise_type in ("sprs", "both"):
        solver = DiscreteSPRSSolver()
        df_grid = float(np.mean(np.diff(f_grid)))
        if L_arr is None:
            sprs_fwd = solver.compute_sprs_spectrum_conti(fiber, grid, f_grid, direction="forward")
            sprs_bwd = solver.compute_sprs_spectrum_conti(fiber, grid, f_grid, direction="backward")
            fwd[:, 0] += sprs_fwd
            bwd[:, 0] += sprs_bwd
        else:
            sprs_fwd = solver.compute_sprs_spectrum_conti_l_array(
                fiber, grid, f_grid, L_arr=L_arr, direction="forward"
            )
            sprs_bwd = solver.compute_sprs_spectrum_conti_l_array(
                fiber, grid, f_grid, L_arr=L_arr, direction="backward"
            )
            fwd += sprs_fwd
            bwd += sprs_bwd

    # Return (N_f,) for backward compatibility
    if n_l == 1:
        return np.asarray(fwd.ravel(), dtype=np.float64), np.asarray(bwd.ravel(), dtype=np.float64)
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
    n_ch = int(base_config.num_channels)
    power_label = _POWER_OVERRIDE_DBM if _POWER_OVERRIDE_DBM is not None else 0.0
    if _PROFILE_ENABLED:
        print(
            "[profile] precompute_by_channel "
            f"type={noise_type}, power={power_label:+.1f} dBm, "
            f"lengths={n_l}, quantum_channels={n_q}, freq_points={len(noise_f_grid)}, models={len(model_keys)}"
        )

    if noise_type == "with_signal":
        # Continuous PSD: noise (FWM+SpRS) + signal, integrate to power per length
        from qkd_sim.physical.signal import SpectrumType

        df = float(np.mean(np.diff(noise_f_grid)))
        all_by_len = {
            ch_idx: {mk: {"fwd": np.zeros(n_l), "bwd": np.zeros(n_l)} for mk in model_keys}
            for ch_idx in range(n_ch)
        }
        # Classical set: only C39/C40/C41 (zero-based indices [38, 39, 40])
        classical_set = set(CLASSICAL_INDICES)
        L_arr = np.array(LENGTHS_KM, dtype=np.float64) * 1e3  # km → m
        fiber_base = _make_fiber(fiber_params, LENGTHS_KM[0])
        model_config_noise = WDMConfig(
            start_freq=base_config.start_freq,
            start_channel=base_config.start_channel,
            end_channel=base_config.end_channel,
            channel_spacing=base_config.channel_spacing,
            B_s=base_config.B_s,
            P0=_get_P0(),
            beta_rolloff=WDM_PARAMS["beta_rolloff"],
            quantum_channel_indices=list(base_config.quantum_channel_indices),
            channel_powers_W=base_config.channel_powers_W,
            num_channels=int(base_config.num_channels),
        )
        grid_noise = build_wdm_grid(
            config=model_config_noise,
            spectrum_type=SpectrumType.RAISED_COSINE,
            f_grid=noise_f_grid,
            classical_channel_indices=CLASSICAL_INDICES,
            modulation_format="16QAM",
        )
        # Vectorize over all lengths at once
        noise_fwd_psd_all, noise_bwd_psd_all = _compute_noise_spectrum_pair(
            "both", fiber_base, grid_noise, noise_f_grid, L_arr=L_arr
        )  # shape (N_f, N_L) or (N_f,) when n_l==1
        # Guard: if n_l==1, result is 1D; expand to 2D for [:, li] indexing
        if noise_fwd_psd_all.ndim == 1:
            noise_fwd_psd_all = noise_fwd_psd_all[:, np.newaxis]
            noise_bwd_psd_all = noise_bwd_psd_all[:, np.newaxis]
        # Signal PSD: sum ONLY over classical channels (C39/C40/C41) — computed once, shared by all model_keys
        signal_psd = np.zeros(len(noise_f_grid), dtype=np.float64)
        for idx, ch in enumerate(grid_noise.channels):
            if idx in classical_set:
                signal_psd += ch.get_psd(noise_f_grid)
        for model_key in model_keys:
            spec = specs[model_key]
            # noise_fwd_psd_all has shape (N_f, N_L); broadcast with signal_psd
            for li in range(n_l):
                total_fwd_psd = noise_fwd_psd_all[:, li] + signal_psd
                total_bwd_psd = noise_bwd_psd_all[:, li] + signal_psd
                total_fwd_power = float(np.sum(total_fwd_psd) * df)
                total_bwd_power = float(np.sum(total_bwd_psd) * df)
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

            # G_TX is independent of fiber length; compute once outside the length loop
            fwd = np.zeros(n_ch, dtype=np.float64)
            bwd = np.zeros(n_ch, dtype=np.float64)
            fwd[classical_mask] = signal[classical_mask]
            # bwd remains zeros (no backward-propagating signal at transmit side)

            for li in range(n_l):
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

    # --- CSV caching for precomputed data ---
    _PRECOMPUTED_DIR = _PROJECT_ROOT / "data" / "precomputed"


    all_by_len = {
        q_local_idx: {mk: {"fwd": np.zeros(n_l), "bwd": np.zeros(n_l)} for mk in model_keys}
        for q_local_idx in range(n_q)
    }
    # Pre-build all grids per (q_local_idx, model_key) to avoid repeated _build_model_grid calls
    grid_cache: dict[tuple[int, str], object] = {}
    for q_local_idx, q_idx in enumerate(quantum_indices):
        single_q_config = _build_wdm_config([q_idx])
        for model_key in model_keys:
            spec = specs[model_key]
            cache_key = (q_local_idx, model_key)
            if cache_key not in grid_cache:
                grid_cache[cache_key] = _build_model_grid(
                    model_key,
                    spec,
                    single_q_config,
                    noise_f_grid,
                    osa_csv_path,
                )
            grid = grid_cache[cache_key]
            # Fiber varies with length, not with q_idx or model_key; create once per length
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
    n_ch = int(base_config.num_channels)

    if noise_type == "with_signal":
        # Continuous PSD: noise (FWM+SpRS) + signal, output (N_f,) per length
        from qkd_sim.physical.signal import SpectrumType

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
        # Classical set: only C39/C40/C41 (zero-based indices [38, 39, 40])
        classical_set = set(CLASSICAL_INDICES)
        for li, length_km in enumerate(LENGTHS_KM):
            fiber = _make_fiber(fiber_params, length_km)
            # Noise spectrum: use REAL quantum/classical split (only C39/C40/C41 are pumps)
            model_config_noise = WDMConfig(
                start_freq=base_config.start_freq,
                start_channel=base_config.start_channel,
                end_channel=base_config.end_channel,
                channel_spacing=base_config.channel_spacing,
                B_s=base_config.B_s,
                P0=_get_P0(),
                beta_rolloff=WDM_PARAMS["beta_rolloff"],
                quantum_channel_indices=list(base_config.quantum_channel_indices),
                channel_powers_W=base_config.channel_powers_W,
                num_channels=int(base_config.num_channels),
            )
            grid_noise = build_wdm_grid(
                config=model_config_noise,
                spectrum_type=SpectrumType.RAISED_COSINE,
                f_grid=noise_f_grid,
                classical_channel_indices=CLASSICAL_INDICES,
                modulation_format="16QAM",
            )
            noise_fwd_psd, noise_bwd_psd = _compute_noise_spectrum_pair(
                "both", fiber, grid_noise, noise_f_grid
            )
            # Signal PSD: sum ONLY over classical channels (C39/C40/C41) — computed once, shared by all model_keys
            signal_psd = np.zeros_like(noise_f_grid, dtype=np.float64)
            for idx, ch in enumerate(grid_noise.channels):
                if idx in classical_set:
                    signal_psd += ch.get_psd(noise_f_grid)
            for model_key in model_keys:
                spec = specs[model_key]
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
        # G_TX is independent of fiber length; compute once per model, assign to all lengths
        for model_key in model_keys:
            spec = specs[model_key]
            grid_all = _build_all_classical_grid(spec, base_config, noise_f_grid, osa_csv_path)
            signal = _integrate_signal_per_channel(grid_all, grid_all.f_grid)
            classical_mask = np.array([ch.channel_type == "classical" for ch in grid_all.channels], dtype=bool)

            fwd = np.zeros(n_ch, dtype=np.float64)
            bwd = np.zeros(n_ch, dtype=np.float64)
            fwd[classical_mask] = signal[classical_mask]
            # bwd remains zeros (no backward-propagating signal at transmit side)

            fwd_arr = np.asarray(fwd, dtype=np.float64)
            bwd_arr = np.asarray(bwd, dtype=np.float64)
            for li in range(n_l):
                all_by_ch[li][model_key]["fwd"] = fwd_arr
                all_by_ch[li][model_key]["bwd"] = bwd_arr

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

    # --- Multiprocessing dispatch for default noise types ---
    if _MP_ENABLED and n_l > 1 and noise_type not in ("with_signal", "only_signal"):
        # Import worker from separate module (avoids Windows spawn path issues)
        from scripts.dash_workers import mp_worker_single_length

        # Serialize data for subprocess
        wdm_cfg_dict = dict(
            start_freq=base_config.start_freq,
            start_channel=base_config.start_channel,
            end_channel=base_config.end_channel,
            channel_spacing=base_config.channel_spacing,
            B_s=base_config.B_s,
            P0=_get_P0(),
            beta_rolloff=WDM_PARAMS["beta_rolloff"],
            quantum_channel_indices=list(base_config.quantum_channel_indices),
            num_channels=int(base_config.num_channels),
        )
        noise_f_list = noise_f_grid.tolist()
        power_dbm = _POWER_OVERRIDE_DBM

        futures = {}
        with ProcessPoolExecutor(max_workers=_get_mp_workers()) as executor:
            for li, length_km in enumerate(LENGTHS_KM):
                fut = executor.submit(
                    mp_worker_single_length,
                    li, length_km, noise_type,
                    model_keys, specs,
                    wdm_cfg_dict, noise_f_list,
                    fiber_params,
                    CLASSICAL_INDICES,
                    len(noise_f_list),
                    power_dbm,
                )
                futures[fut] = li

            for fut in as_completed(futures):
                li, results = fut.result()
                for mk, res in zip(model_keys, results):
                    all_by_ch[li][mk] = res

        valid_l_indices = [
            li for li in range(n_l)
            if any(
                np.any(all_by_ch[li][mk]["fwd"] > 0)
                or np.any(all_by_ch[li][mk]["bwd"] > 0)
                for mk in model_keys
            )
        ]
        return all_by_ch, valid_l_indices
    # --- End multiprocessing ---

    # Continuous model: iterate per length (original approach, avoids L_arr issues with sprs)
    # Cache: grid depends only on model_key (not on q_idx or li)
    grid_cache: dict[str, object] = {}

    for model_key in model_keys:
        spec = specs[model_key]
        if not spec["continuous"]:
            continue
        t_grid = 0.0
        t_fiber = 0.0
        t_solver = 0.0
        t_store = 0.0
        if model_key not in grid_cache:
            _t = time.perf_counter()
            grid_cache[model_key] = _build_model_grid(
                model_key,
                spec,
                base_config,
                noise_f_grid,
                osa_csv_path,
            )
            t_grid += time.perf_counter() - _t
        grid = grid_cache[model_key]
        df = float(np.mean(np.diff(noise_f_grid)))

        if noise_type in ("fwm", "sprs"):
            _t = time.perf_counter()
            fiber = _make_fiber(fiber_params, float(LENGTHS_KM[0]))
            L_arr = np.asarray(LENGTHS_KM, dtype=np.float64) * 1e3
            t_fiber += time.perf_counter() - _t
            _t = time.perf_counter()
            fwd_psd_all, bwd_psd_all = _compute_noise_spectrum_pair(
                noise_type, fiber, grid, noise_f_grid, L_arr=L_arr
            )  # shape (N_f, N_L)
            t_solver += time.perf_counter() - _t
            if fwd_psd_all.ndim == 1:
                fwd_psd_all = fwd_psd_all[:, np.newaxis]
                bwd_psd_all = bwd_psd_all[:, np.newaxis]
            for li in range(n_l):
                _t = time.perf_counter()
                all_by_ch[li][model_key] = {
                    "fwd": np.asarray(fwd_psd_all[:, li] * df, dtype=np.float64),
                    "bwd": np.asarray(bwd_psd_all[:, li] * df, dtype=np.float64),
                    "x": np.asarray(noise_f_grid, dtype=np.float64),
                    "x_kind": "frequency_grid",
                    "y_kind": "power_per_bin",
                }
                t_store += time.perf_counter() - _t
        else:
            for li, length_km in enumerate(LENGTHS_KM):
                _t = time.perf_counter()
                fiber = _make_fiber(fiber_params, length_km)
                t_fiber += time.perf_counter() - _t
                _t = time.perf_counter()
                fwd_psd, bwd_psd = _compute_noise_spectrum_pair(
                    noise_type, fiber, grid, noise_f_grid
                )  # shape (N_f,)
                t_solver += time.perf_counter() - _t
                _t = time.perf_counter()
                all_by_ch[li][model_key] = {
                    "fwd": np.asarray(fwd_psd * df, dtype=np.float64),
                    "bwd": np.asarray(bwd_psd * df, dtype=np.float64),
                    "x": np.asarray(noise_f_grid, dtype=np.float64),
                    "x_kind": "frequency_grid",
                    "y_kind": "power_per_bin",
                }
                t_store += time.perf_counter() - _t
        if _PROFILE_ENABLED:
            print(
                "[profile]   continuous "
                f"model={model_key}: grid={t_grid:.3f}s, fiber={t_fiber:.3f}s, "
                f"solver={t_solver:.3f}s, store={t_store:.3f}s"
            )

    # Discrete model: grid per (model_key, q_idx), reuse grid_cache for same model_key
    for model_key in model_keys:
        spec = specs[model_key]
        if spec["continuous"]:
            continue  # Already handled above
        t_grid = 0.0
        t_fiber = 0.0
        t_solver = 0.0
        t_store = 0.0
        solver_calls = 0
        # Build one grid per q_idx (quantum channel changes which channel is "quantum")
        q_idx_to_grid: dict[int, object] = {}
        for q_local_idx, q_idx in enumerate(quantum_indices):
            if q_idx not in q_idx_to_grid:
                _t = time.perf_counter()
                q_idx_to_grid[q_idx] = _build_model_grid(
                    model_key,
                    spec,
                    _build_wdm_config([q_idx]),
                    noise_f_grid,
                    osa_csv_path,
                )
                t_grid += time.perf_counter() - _t

        for li, length_km in enumerate(LENGTHS_KM):
            _t = time.perf_counter()
            fiber = _make_fiber(fiber_params, length_km)
            t_fiber += time.perf_counter() - _t
            fwd_arr = np.zeros(n_q, dtype=np.float64)
            bwd_arr = np.zeros(n_q, dtype=np.float64)
            for q_local_idx, q_idx in enumerate(quantum_indices):
                grid = q_idx_to_grid[q_idx]
                _t = time.perf_counter()
                fwd, bwd = _compute_noise_power_pair(
                    noise_type,
                    fiber,
                    grid,
                    continuous=False,
                )
                t_solver += time.perf_counter() - _t
                solver_calls += 1
                if len(fwd) > 0:
                    fwd_arr[q_local_idx] = float(fwd[0])
                    bwd_arr[q_local_idx] = float(bwd[0])
            _t = time.perf_counter()
            all_by_ch[li][model_key] = {
                "fwd": np.asarray(fwd_arr, dtype=np.float64),
                "bwd": np.asarray(bwd_arr, dtype=np.float64),
                "x": np.asarray(q_center_freqs, dtype=np.float64),
                "x_kind": "channel_center",
                "y_kind": "channel_power",
            }
            t_store += time.perf_counter() - _t
        if _PROFILE_ENABLED:
            print(
                "[profile]   discrete "
                f"model={model_key}: grid={t_grid:.3f}s, fiber={t_fiber:.3f}s, "
                f"solver={t_solver:.3f}s, store={t_store:.3f}s, solver_calls={solver_calls}"
            )

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


# --- Startup precomputation of ALL power levels (step=5 dBm, 7 values) ---
def precompute_by_channel_all_powers(
    noise_type: str,
    specs: dict,
    LENGTHS_KM: np.ndarray,
    base_config: WDMConfig,
    noise_f_grid: np.ndarray,
    osa_csv_path: Path,
    fiber_params: dict,
) -> tuple[dict, list]:
    """Precompute ALL power levels at startup, save to CSV and memory cache.

    Clears stale CSV cache first, then computes for each power level in
    PRECOMPUTE_POWER_LEVELS in parallel (if total work justifies spawn overhead),
    stores in both CSV (for Origin) and _POWER_CACHE (for instant slider response).

    Returns (all_by_ch at 0 dBm, valid_l_indices).
    """
    import multiprocessing

    with profile_scope("precompute_by_channel_all_powers: clear stale CSV cache"):
        _clear_precomputed_csv_files("ch")
        _POWER_CACHE.clear()

    if noise_type in ("sprs", "fwm"):
        print(
            "[profile] precompute_by_channel_all_powers: "
            f"using 0 dBm base result plus {'linear' if noise_type == 'sprs' else 'cubic'} power scaling"
        )
        set_power_override(0.0)
        with profile_scope("power +0 dBm: base precompute_by_channel"):
            base_all_by_idx, base_valid_l = precompute_by_channel(
                noise_type, specs, LENGTHS_KM, base_config,
                noise_f_grid, osa_csv_path, fiber_params,
            )
        results: dict[float, tuple[dict, list]] = {}
        for p in PRECOMPUTE_POWER_LEVELS:
            scale = _power_scaling_factor(noise_type, float(p))
            assert scale is not None
            with profile_scope(f"power {p:+.0f} dBm: scale from 0 dBm"):
                all_by_idx = (
                    base_all_by_idx
                    if float(p) == 0.0
                    else _scale_precomputed_result(base_all_by_idx, scale)
                )
            with profile_scope(f"power {p:+.0f} dBm: save CSV cache"):
                _cache_precomputed_result(all_by_idx, p, noise_type, list(specs.keys()), index_prefix="ch")
            _POWER_CACHE[("ch", float(p))] = all_by_idx
            results[float(p)] = (all_by_idx, base_valid_l)

        set_power_override(0.0)
        return results[0.0]

    if noise_type == "both":
        print(
            "[profile] precompute_by_channel_all_powers: "
            "using separate 0 dBm FWM/SpRS bases plus cubic/linear power scaling"
        )
        set_power_override(0.0)
        with profile_scope("power +0 dBm: base FWM precompute_by_channel"):
            fwm_base_by_idx, fwm_valid_l = precompute_by_channel(
                "fwm", specs, LENGTHS_KM, base_config,
                noise_f_grid, osa_csv_path, fiber_params,
            )
        with profile_scope("power +0 dBm: base SpRS precompute_by_channel"):
            sprs_base_by_idx, sprs_valid_l = precompute_by_channel(
                "sprs", specs, LENGTHS_KM, base_config,
                noise_f_grid, osa_csv_path, fiber_params,
            )
        base_valid_l = sorted(set(fwm_valid_l) & set(sprs_valid_l))
        results: dict[float, tuple[dict, list]] = {}
        for p in PRECOMPUTE_POWER_LEVELS:
            fwm_scale = _power_scaling_factor("fwm", float(p))
            sprs_scale = _power_scaling_factor("sprs", float(p))
            assert fwm_scale is not None and sprs_scale is not None
            with profile_scope(f"power {p:+.0f} dBm: combine FWM/SpRS bases"):
                all_by_idx = _combine_precomputed_results(
                    fwm_base_by_idx, sprs_base_by_idx, fwm_scale, sprs_scale
                )
            with profile_scope(f"power {p:+.0f} dBm: save CSV cache"):
                _cache_precomputed_result(all_by_idx, p, noise_type, list(specs.keys()), index_prefix="ch")
            _POWER_CACHE[("ch", float(p))] = all_by_idx
            results[float(p)] = (all_by_idx, base_valid_l)

        set_power_override(0.0)
        return results[0.0]

    # Sequential fast path: avoids spawn overhead for fast noise types (e.g. sprs ~1s total)
    if not _MP_ENABLED or len(PRECOMPUTE_POWER_LEVELS) <= 2:
        results: dict[float, tuple[dict, list]] = {}
        for p in PRECOMPUTE_POWER_LEVELS:
            set_power_override(float(p))
            with profile_scope(f"power {p:+.0f} dBm: precompute_by_channel"):
                all_by_idx, valid_l = precompute_by_channel(
                    noise_type, specs, LENGTHS_KM, base_config,
                    noise_f_grid, osa_csv_path, fiber_params,
                )
            with profile_scope(f"power {p:+.0f} dBm: save CSV cache"):
                _cache_precomputed_result(all_by_idx, p, noise_type, list(specs.keys()), index_prefix="ch")
            _POWER_CACHE[("ch", p)] = all_by_idx
            results[p] = (all_by_idx, valid_l)
    else:
        # Parallel: all power levels are independent
        n_workers = min(len(PRECOMPUTE_POWER_LEVELS), _get_mp_workers())
        ctx = multiprocessing.get_context()
        with profile_scope(f"all powers: parallel precompute ({n_workers} workers)"):
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        _precompute_channel_worker,
                        p, noise_type, specs, LENGTHS_KM, base_config,
                        noise_f_grid, osa_csv_path, fiber_params,
                    ): p
                    for p in PRECOMPUTE_POWER_LEVELS
                }
                results = {}
                for fut in futures:
                    p, all_by_idx, valid_l = fut.result()
                    # Cache is already written inside the worker
                    _POWER_CACHE[("ch", p)] = all_by_idx
                    results[p] = (all_by_idx, valid_l)

    # Extract 0 dBm result
    result_all, result_valid = results[0.0]

    # Restore 0 dBm in memory cache and power override
    set_power_override(0.0)
    _POWER_CACHE[("ch", 0.0)] = result_all
    return result_all, result_valid


def precompute_by_length_all_powers(
    noise_type: str,
    specs: dict,
    LENGTHS_KM: np.ndarray,
    base_config: WDMConfig,
    noise_f_grid: np.ndarray,
    osa_csv_path: Path,
    fiber_params: dict,
) -> tuple[dict, list]:
    """Precompute ALL power levels at startup, save to CSV and memory cache.

    Clears stale CSV cache first, then computes for each power level in
    PRECOMPUTE_POWER_LEVELS in parallel (if total work justifies spawn overhead),
    stores in both CSV (for Origin) and _POWER_CACHE (for instant slider response).

    Returns (all_by_len at 0 dBm, valid_ch_indices).
    """
    import multiprocessing

    _clear_precomputed_csv_files("len")
    _POWER_CACHE.clear()

    if noise_type == "both":
        print(
            "[profile] precompute_by_length_all_powers: "
            "using separate 0 dBm FWM/SpRS bases plus cubic/linear power scaling"
        )
        set_power_override(0.0)
        with profile_scope("power +0 dBm: base FWM precompute_by_length"):
            fwm_base_by_idx, fwm_valid_ch = precompute_by_length(
                "fwm", specs, LENGTHS_KM, base_config,
                noise_f_grid, osa_csv_path, fiber_params,
            )
        with profile_scope("power +0 dBm: base SpRS precompute_by_length"):
            sprs_base_by_idx, sprs_valid_ch = precompute_by_length(
                "sprs", specs, LENGTHS_KM, base_config,
                noise_f_grid, osa_csv_path, fiber_params,
            )
        base_valid_ch = sorted(set(fwm_valid_ch) & set(sprs_valid_ch))
        results: dict[float, tuple[dict, list]] = {}
        for p in PRECOMPUTE_POWER_LEVELS:
            fwm_scale = _power_scaling_factor("fwm", float(p))
            sprs_scale = _power_scaling_factor("sprs", float(p))
            assert fwm_scale is not None and sprs_scale is not None
            with profile_scope(f"power {p:+.0f} dBm: combine FWM/SpRS bases"):
                all_by_idx = _combine_precomputed_results(
                    fwm_base_by_idx, sprs_base_by_idx, fwm_scale, sprs_scale
                )
            with profile_scope(f"power {p:+.0f} dBm: save CSV cache"):
                _cache_precomputed_result(all_by_idx, p, noise_type, list(specs.keys()), index_prefix="len")
            _POWER_CACHE[("len", float(p))] = all_by_idx
            results[float(p)] = (all_by_idx, base_valid_ch)

        set_power_override(0.0)
        return results[0.0]

    # Sequential fast path: avoids spawn overhead for fast noise types (e.g. sprs ~1s total)
    if not _MP_ENABLED or len(PRECOMPUTE_POWER_LEVELS) <= 2:
        results: dict[float, tuple[dict, list]] = {}
        for p in PRECOMPUTE_POWER_LEVELS:
            set_power_override(float(p))
            all_by_idx, valid_ch = precompute_by_length(
                noise_type, specs, LENGTHS_KM, base_config,
                noise_f_grid, osa_csv_path, fiber_params,
            )
            _cache_precomputed_result(all_by_idx, p, noise_type, list(specs.keys()), index_prefix="len")
            _POWER_CACHE[("len", p)] = all_by_idx
            results[p] = (all_by_idx, valid_ch)
    else:
        # Parallel: all power levels are independent
        n_workers = min(len(PRECOMPUTE_POWER_LEVELS), _get_mp_workers())
        ctx = multiprocessing.get_context()
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _precompute_length_worker,
                    p, noise_type, specs, LENGTHS_KM, base_config,
                    noise_f_grid, osa_csv_path, fiber_params,
                ): p
                for p in PRECOMPUTE_POWER_LEVELS
            }
            results = {}
            for fut in futures:
                p, all_by_idx, valid_ch = fut.result()
                # CSV caching is done inside the worker via precompute_by_length
                _POWER_CACHE[("len", p)] = all_by_idx
                results[p] = (all_by_idx, valid_ch)

    # Extract 0 dBm result
    result_all, result_valid = results[0.0]

    # Restore 0 dBm in memory cache and power override
    set_power_override(0.0)
    _POWER_CACHE[("len", 0.0)] = result_all
    return result_all, result_valid
