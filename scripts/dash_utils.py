from __future__ import annotations

from pathlib import Path

import numpy as np

from qkd_sim.config.schema import WDMConfig
from qkd_sim.physical.signal import build_wdm_grid

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

WDM_PARAMS = dict(
    start_freq=190.1e12,
    start_channel=1,
    end_channel=61,
    channel_spacing=100e9,
    B_s=32e9,
    P0=1e-3,
    beta_rolloff=0.2,
)
CLASSICAL_INDICES = [38, 39, 40]
NOISE_GRID_RESOLUTION_HZ = 5e9
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
    return WDMConfig(**WDM_PARAMS, quantum_channel_indices=list(quantum_indices))


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


def _display_channel_label(channel_index: int) -> str:
    return f"C{channel_index + WDM_PARAMS['start_channel']}"


def adaptive_log_ticks(
    y_bot_log: float, y_top_log: float, max_ticks: int = 8
) -> dict:
    """自适应对数轴刻度: Plotly 自动管理密度, tickformat 保证干净标签."""
    return dict(
        tickmode="auto",
        nticks=max_ticks,
        tickformat="1e",
        exponentformat="power",
    )


def adaptive_linear_ticks(
    y_bot: float, y_top: float, max_ticks: int = 8
) -> dict:
    """自适应线性轴刻度: Plotly 自动管理密度, 保留两位小数."""
    return dict(
        tickmode="auto",
        nticks=max_ticks,
        tickformat=".2f",
    )
