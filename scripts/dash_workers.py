"""Worker functions for multiprocessing precomputation.

This module is imported by subprocess workers via ProcessPoolExecutor.
Must not trigger any app-level initialization at import time.
"""

from __future__ import annotations

import numpy as np


def mp_worker_single_length(
    li: int,
    length_km: float,
    noise_type: str,
    model_keys: list[str],
    specs: dict,
    wdm_cfg: dict,
    noise_f_grid: list[float],
    fiber_params: dict,
    classical_indices: list[int],
    n_f: int,
    power_override_dbm: float | None,
) -> tuple[int, list[dict]]:
    """Worker: one length × all models. Runs in subprocess via ProcessPoolExecutor.

    All modules re-imported inside for Windows compatibility (spawn start method).
    Returns (li, list of result dicts for each model_key in model_keys order).
    """
    import numpy as np
    from qkd_sim.config.schema import WDMConfig
    from qkd_sim.physical.fiber import Fiber
    from qkd_sim.physical.signal import build_wdm_grid, SpectrumType
    from qkd_sim.physical.noise import DiscreteFWMSolver, DiscreteSPRSSolver

    noise_f_grid = np.array(noise_f_grid, dtype=np.float64)

    def _make_fiber(fp: dict, L_km: float) -> Fiber:
        from qkd_sim.config.schema import FiberConfig
        params = dict(fp)
        params["L_km"] = float(L_km)
        return Fiber(FiberConfig(**params))

    def _P0(p_dbm: float | None) -> float:
        p = 1e-3  # 0 dBm default
        if p_dbm is not None:
            p = 1e-3 * (10 ** (p_dbm / 10))
        return p

    def _noise_pair(nt: str, fib: Fiber, grd, fgr: np.ndarray):
        fwd = np.zeros(len(fgr), dtype=np.float64)
        bwd = np.zeros(len(fgr), dtype=np.float64)
        if nt in ("fwm", "both"):
            slv = DiscreteFWMSolver()
            fwd += slv.compute_fwm_spectrum_conti(fib, grd, fgr, direction="forward")
            bwd += slv.compute_fwm_spectrum_conti(fib, grd, fgr, direction="backward")
        if nt in ("sprs", "both"):
            slv = DiscreteSPRSSolver()
            df = float(np.mean(np.diff(fgr)))
            fwd += slv.compute_sprs_spectrum_conti(fib, grd, fgr, direction="forward") / df
            bwd += slv.compute_sprs_spectrum_conti(fib, grd, fgr, direction="backward") / df
        return np.asarray(fwd, dtype=np.float64), np.asarray(bwd, dtype=np.float64)

    cfg = WDMConfig(**wdm_cfg)
    p0 = _P0(power_override_dbm)
    classical_set = set(classical_indices)
    df = float(np.mean(np.diff(noise_f_grid)))

    results: list[dict] = []

    if noise_type == "with_signal":
        noise_cfg = WDMConfig(
            start_freq=cfg.start_freq,
            start_channel=cfg.start_channel,
            end_channel=cfg.end_channel,
            channel_spacing=cfg.channel_spacing,
            B_s=cfg.B_s,
            P0=p0,
            beta_rolloff=0.0,
            quantum_channel_indices=list(cfg.quantum_channel_indices),
            channel_powers_W=cfg.channel_powers_W,
            num_channels=int(cfg.num_channels),
        )
        grid_noise = build_wdm_grid(
            config=noise_cfg,
            spectrum_type=SpectrumType.RAISED_COSINE,
            f_grid=noise_f_grid,
            classical_channel_indices=classical_indices,
            modulation_format="16QAM",
        )
        fiber = _make_fiber(fiber_params, length_km)
        n_fwd, n_bwd = _noise_pair("both", fiber, grid_noise, noise_f_grid)
        for mk in model_keys:
            sig_psd = np.zeros(n_f, dtype=np.float64)
            for idx, ch in enumerate(grid_noise.channels):
                if idx in classical_set:
                    sig_psd += ch.get_psd(noise_f_grid)
            fwd = (n_fwd + sig_psd) * df
            bwd = (n_bwd + sig_psd) * df
            results.append({
                "fwd": fwd, "bwd": bwd,
                "x": np.asarray(noise_f_grid, dtype=np.float64),
                "x_kind": "frequency_grid",
                "y_kind": "power_per_bin",
            })

    elif noise_type == "only_signal":
        classical_set = set(classical_indices)
        all_idx = list(range(int(cfg.num_channels)))
        # x-axis: all channel center frequencies
        all_ch_freqs = np.array(
            [
                cfg.start_freq + idx * cfg.channel_spacing
                for idx in all_idx
            ],
            dtype=np.float64,
        )
        allclassical_cfg = WDMConfig(
            start_freq=cfg.start_freq,
            start_channel=cfg.start_channel,
            end_channel=cfg.end_channel,
            channel_spacing=cfg.channel_spacing,
            B_s=cfg.B_s,
            P0=p0,
            beta_rolloff=0.0,
            quantum_channel_indices=[],
            channel_powers_W=cfg.channel_powers_W,
            num_channels=int(cfg.num_channels),
        )
        grid_all = build_wdm_grid(
            config=allclassical_cfg,
            spectrum_type=SpectrumType.RAISED_COSINE,
            f_grid=noise_f_grid,
            classical_channel_indices=all_idx,
            modulation_format="16QAM",
        )
        fiber = _make_fiber(fiber_params, length_km)
        for mk in model_keys:
            sig_psd = np.zeros(n_f, dtype=np.float64)
            for idx, ch in enumerate(grid_all.channels):
                if idx in classical_set:
                    sig_psd += ch.get_psd(noise_f_grid)
            fwd = sig_psd * df
            bwd = np.zeros(n_f, dtype=np.float64)
            results.append({
                "fwd": fwd, "bwd": bwd,
                "x": all_ch_freqs,
                "x_kind": "channel_center",
                "y_kind": "channel_power",
            })

    else:
        # fwm / sprs / both
        fiber = _make_fiber(fiber_params, length_km)
        for mk in model_keys:
            sp = specs[mk]
            if sp["continuous"]:
                mc = WDMConfig(
                    start_freq=cfg.start_freq,
                    start_channel=cfg.start_channel,
                    end_channel=cfg.end_channel,
                    channel_spacing=cfg.channel_spacing,
                    B_s=cfg.B_s,
                    P0=p0,
                    beta_rolloff=sp["beta_rolloff"] if sp["beta_rolloff"] is not None else 0.0,
                    quantum_channel_indices=list(cfg.quantum_channel_indices),
                    channel_powers_W=cfg.channel_powers_W,
                    num_channels=int(cfg.num_channels),
                )
                grid = build_wdm_grid(
                    config=mc,
                    spectrum_type=sp["spectrum_type"],
                    f_grid=noise_f_grid,
                    classical_channel_indices=classical_indices,
                    modulation_format="16QAM" if sp["spectrum_type"] == SpectrumType.RAISED_COSINE else "OOK",
                )
                fwd, bwd = _noise_pair(noise_type, fiber, grid, noise_f_grid)
                fwd = fwd * df
                bwd = bwd * df
                results.append({
                    "fwd": fwd, "bwd": bwd,
                    "x": np.asarray(noise_f_grid, dtype=np.float64),
                    "x_kind": "frequency_grid",
                    "y_kind": "power_per_bin",
                })
            else:
                n_q = len(cfg.quantum_channel_indices)
                # Quantum channel center frequencies for x-axis
                q_center_freqs = np.array(
                    [
                        cfg.start_freq + idx * cfg.channel_spacing
                        for idx in cfg.quantum_channel_indices
                    ],
                    dtype=np.float64,
                )
                fwd_arr = np.zeros(n_q, dtype=np.float64)
                bwd_arr = np.zeros(n_q, dtype=np.float64)
                for qi, q_idx in enumerate(cfg.quantum_channel_indices):
                    qc = WDMConfig(
                        start_freq=cfg.start_freq,
                        start_channel=cfg.start_channel,
                        end_channel=cfg.end_channel,
                        channel_spacing=cfg.channel_spacing,
                        B_s=cfg.B_s,
                        P0=p0,
                        beta_rolloff=0.0,
                        quantum_channel_indices=[q_idx],
                        channel_powers_W=cfg.channel_powers_W,
                        num_channels=int(cfg.num_channels),
                    )
                    grid = build_wdm_grid(
                        config=qc,
                        spectrum_type=sp["spectrum_type"],
                        f_grid=noise_f_grid,
                        classical_channel_indices=classical_indices,
                        modulation_format="16QAM" if sp["spectrum_type"] == SpectrumType.RAISED_COSINE else "OOK",
                    )
                    # Use compute_forward/compute_backward (return N_q power, not N_f PSD)
                    if noise_type in ("fwm", "both"):
                        slv = DiscreteFWMSolver()
                        fwd_arr[qi] = float(slv.compute_forward(fiber, grid)[0])
                        bwd_arr[qi] = float(slv.compute_backward(fiber, grid)[0])
                    if noise_type in ("sprs", "both"):
                        slv = DiscreteSPRSSolver()
                        f = float(slv.compute_forward(fiber, grid)[0])
                        b = float(slv.compute_backward(fiber, grid)[0])
                        if noise_type == "sprs":
                            fwd_arr[qi] = f
                            bwd_arr[qi] = b
                        else:  # "both": accumulate
                            fwd_arr[qi] += f
                            bwd_arr[qi] += b
                fwd = fwd_arr
                bwd = bwd_arr
                results.append({
                    "fwd": fwd, "bwd": bwd,
                    "x": q_center_freqs,
                    "x_kind": "channel_center",
                    "y_kind": "channel_power",
                })

    return li, results
