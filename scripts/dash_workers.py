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
        return Fiber(
            length=L_km * 1e3,
            alpha=fp["alpha_dB_per_km"],
            gamma=fp["gamma_per_W_km"],
            D_c=fp["D_c"],
            D_slope=fp["D_slope"],
            rayleigh_coeff=fp["rayleigh_coeff"],
            T_kelvin=fp["T_kelvin"],
        )

    def _P0(p_dbm: float | None) -> float:
        p = 1e-3  # 0 dBm default
        if p_dbm is not None:
            p = 1e-3 * (10 ** (p_dbm / 10))
        return p

    def _noise_pair(nt: str, fib: Fiber, grd, fgr: np.ndarray):
        fwd = np.zeros_like(fgr, dtype=np.float64)
        bwd = np.zeros_like(fgr, dtype=np.float64)
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
            num_channels=int(cfg.num_channels),
        )
        grid_noise = build_wdm_grid(
            config=noise_cfg,
            spectrum_type=SpectrumType.RAISED_COSINE,
            f_grid=noise_f_grid,
            classical_channel_indices=classical_indices,
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
            results.append({"fwd": fwd, "bwd": bwd})

    elif noise_type == "only_signal":
        all_idx = list(range(int(cfg.num_channels)))
        allclassical_cfg = WDMConfig(
            start_freq=cfg.start_freq,
            start_channel=cfg.start_channel,
            end_channel=cfg.end_channel,
            channel_spacing=cfg.channel_spacing,
            B_s=cfg.B_s,
            P0=p0,
            beta_rolloff=0.0,
            quantum_channel_indices=[],
            num_channels=int(cfg.num_channels),
        )
        grid_all = build_wdm_grid(
            config=allclassical_cfg,
            spectrum_type=SpectrumType.RAISED_COSINE,
            f_grid=noise_f_grid,
            classical_channel_indices=all_idx,
        )
        fiber = _make_fiber(fiber_params, length_km)
        nli_fwd, nli_bwd = _noise_pair("fwm", fiber, grid_all, noise_f_grid)
        for mk in model_keys:
            sig_psd = np.zeros(n_f, dtype=np.float64)
            for idx, ch in enumerate(grid_all.channels):
                if idx in classical_set:
                    sig_psd += ch.get_psd(noise_f_grid)
            fwd = (nli_fwd + sig_psd) * df
            bwd = (nli_bwd + sig_psd) * df
            results.append({"fwd": fwd, "bwd": bwd})

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
                    num_channels=int(cfg.num_channels),
                )
                grid = build_wdm_grid(
                    config=mc,
                    spectrum_type=sp["spectrum_type"],
                    f_grid=noise_f_grid,
                    classical_channel_indices=classical_indices,
                )
                fwd, bwd = _noise_pair(noise_type, fiber, grid, noise_f_grid)
                fwd = fwd * df
                bwd = bwd * df
            else:
                n_q = len(cfg.quantum_channel_indices)
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
                        num_channels=int(cfg.num_channels),
                    )
                    grid = build_wdm_grid(
                        config=qc,
                        spectrum_type=sp["spectrum_type"],
                        f_grid=noise_f_grid,
                        classical_channel_indices=classical_indices,
                    )
                    fwd, bwd = _noise_pair(noise_type, fiber, grid, noise_f_grid)
                    if len(fwd) > 0:
                        fwd_arr[qi] = float(fwd[0])
                        bwd_arr[qi] = float(bwd[0])
                fwd = fwd_arr
                bwd = bwd_arr
            results.append({"fwd": fwd, "bwd": bwd})

    return li, results
