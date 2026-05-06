"""Analyze C01-C60 noise vs SKR: -15 dBm, 5 km, strict_finite."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "src"))

from scripts import dash_utils as du
du.MODULATION_FORMAT = 'DP-16QAM'

from scripts.dash_utils import (
    _build_wdm_config, _build_noise_frequency_grid, WDM_PARAMS, CLASSICAL_INDICES,
    precompute_by_channel, get_noise_model_keys, _get_P0, set_power_override,
    compute_skr_point, noise_power_to_p_noise, _resolve_osa_csv,
    _init_skr_model_registry,
)
from qkd_sim.config.plot_config import load_model_specs
from qkd_sim.config.schema import load_skr_config, load_fiber_config

POWER_DBM = -15.0
LENGTH_KM = 5.0

set_power_override(POWER_DBM)
base_quantum = sorted(set(range(1, 62)) - set(CLASSICAL_INDICES))
base_config = _build_wdm_config(base_quantum)
noise_f_grid = _build_noise_frequency_grid(base_config)
df = float(np.mean(np.diff(noise_f_grid)))

LENGTHS = np.array([LENGTH_KM])
specs = load_model_specs("fwm_noise_dp-16qam")

fiber_cfg = load_fiber_config(_PROJECT / "src/qkd_sim/config/defaults/fiber_para/fiber_smf.yaml")
skr_cfg = load_skr_config(_PROJECT / "src/qkd_sim/config/defaults/skr_para/bb84_config.yaml", profile="custom")
osa_csv, osa_freq = _resolve_osa_csv("dp-16qam")

fiber_params = {
    "alpha_dB_per_km": 0.2, "gamma_per_W_km": 1.3,
    "D_ps_nm_km": 17.0, "D_slope_ps_nm2_km": 0.056,
    "A_eff": 8.0e-11, "rayleigh_coeff": 4.8e-8, "T_kelvin": 300.0,
}

print("Computing noise...")
all_by_ch, valid_l = precompute_by_channel(
    "with_signal", specs, LENGTHS, base_config,
    noise_f_grid, osa_csv, fiber_params, osa_freq,
)
li = valid_l[0]
mk = list(all_by_ch[li].keys())[0]
noise_fwd = all_by_ch[li][mk]["noise_only_fwd"]

q_centers = np.array(
    [base_config.start_freq + (itn - 1) * base_config.channel_spacing
     for itn in base_quantum],
    dtype=np.float64,
)

R_rep = skr_cfg.R_rep
skr_fns = _init_skr_model_registry()
skr_fn, skr_label = skr_fns["strict_finite"]

print(f"P0={POWER_DBM:+.0f}dBm  L={LENGTH_KM}km  R_rep={R_rep/1e6:.1f}MHz  B_q={base_config.B_q/1e9:.0f}GHz")
print(f"SKR: strict_finite  Classical: {CLASSICAL_INDICES}")
print(f"{'='*95}")
print(f"{'ITU':>4s} {'f_q':>9s} {'Noise':>10s} {'mu':>8s} {'p_noise':>10s} {'QBER':>8s} {'SKR':>14s}")
print(f"{'─'*4} {'─'*9} {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*14}")

stats = []
for qi, itn in enumerate(base_quantum):
    f_q = q_centers[qi]
    n_w = float(noise_fwd[qi])
    dist_m = float(LENGTH_KM * 1000.0)
    p_noise = noise_power_to_p_noise(n_w, f_q, R_rep)
    mu = n_w / (6.62607015e-34 * f_q * R_rep)  # h*f_q*R_rep
    try:
        bps, bpp, qber = skr_fn(dist_m, fiber_cfg, skr_cfg, p_noise, optimize_params=False)
    except:
        bps, bpp, qber = float('nan'), float('nan'), float('nan')
    noise_dbm = 10*np.log10(max(n_w, 1e-40)/1e-3)

    skr_str = f"{bps:.2e}" if not np.isnan(bps) and bps > 0 else ("NaN" if np.isnan(bps) else "0")
    qber_pct = qber * 100 if not np.isnan(qber) else float('nan')
    print(f"C{itn:02d} {f_q/1e12:9.4f} {noise_dbm:10.2f} {mu:8.4f} {p_noise:10.6f} {qber_pct:7.3f}% {skr_str:>14s}")
    stats.append({"itn": itn, "f_q": f_q, "noise_W": n_w, "noise_dBm": noise_dbm,
                   "mu": mu, "p_noise": p_noise, "qber": qber, "skr": bps})

# ── Analysis ──
noise_db = np.array([s["noise_dBm"] for s in stats])
mu_arr = np.array([s["mu"] for s in stats])
skr_arr = np.array([s["skr"] for s in stats if not np.isnan(s["skr"])])
skr_pos = [(s["itn"], s["skr"]) for s in stats if not np.isnan(s["skr"]) and s["skr"] > 0]

print(f"\n─ Summary ─")
print(f"Noise: {noise_db.min():.2f} ~ {noise_db.max():.2f} dBm (span {noise_db.max()-noise_db.min():.1f} dB)")
print(f"mu:    {mu_arr.min():.4f} ~ {mu_arr.max():.4f}")
print(f"SKR>0: {len(skr_pos)}/{len(base_quantum)} channels")
if skr_pos:
    skr_vals = [s[1] for s in skr_pos]
    print(f"SKR range: {min(skr_vals):.2e} ~ {max(skr_vals):.2e} bps")
    print(f"Channels with SKR>0: C{[s[0] for s in skr_pos[:5]]}...C{[s[0] for s in skr_pos[-5:]]}")

# Find the transition point
print(f"\n─ Transition analysis ─")
for s in stats:
    if not np.isnan(s["skr"]) and s["skr"] == 0 and s["mu"] < 0.01:
        print(f"  SKR=0 at C{s['itn']:02d}: mu={s['mu']:.6f}, p_noise={s['p_noise']:.8f}, noise={s['noise_dBm']:.1f} dBm — dominated by p_dark")
        break
    if not np.isnan(s["skr"]) and s["skr"] == 0 and s["mu"] > 0.01:
        print(f"  SKR=0 at C{s['itn']:02d}: mu={s['mu']:.4f}, p_noise={s['p_noise']:.6f}, noise={s['noise_dBm']:.1f} dBm — noise kills SKR")
        break

for s in stats:
    if not np.isnan(s["skr"]) and s["skr"] > 0:
        print(f"  SKR>0 at C{s['itn']:02d}: mu={s['mu']:.4f}, p_noise={s['p_noise']:.6f}, noise={s['noise_dBm']:.1f} dBm, SKR={s['skr']:.2e} bps")
        break
