"""Compare SpRS noise: sprs_solver.py vs reference/sprs.py.

Run from project root:
    python test_sprs_comparison.py
"""

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_REFERENCE_DIR := _SCRIPT_DIR / "reference"))

from reference.sprs import Channel as RefChannel, Fiber as RefFiber, RamanSolver as RefRamanSolver
from qkd_sim.physical.noise.sprs_solver import DiscreteSPRSSolver
from qkd_sim.physical.noise.raman_data import get_raman_gain
from qkd_sim.physical.signal import build_wdm_grid, SpectrumType
from qkd_sim.config.schema import FiberConfig, WDMConfig


# ---------------------------------------------------------------------------
# Shared test parameters (must match reference/sprs.py __main__ setup)
# ---------------------------------------------------------------------------

FREQ_MIN = 180.0e12
FREQ_MAX = 200.0e12
SPACING = 20e9          # reference uses 20 GHz spacing
CLASSICAL_FREQ_RANGE = (191.1e12, 191.2e12)  # C-band pump band
P0 = 1e-3               # total pump power per channel [W]
FIBER_ALPHA_DB_PER_KM = 0.2
FIBER_LENGTH_M = 10e3
FIBER_TEMP_K = 300.0


def _db_per_km_to_per_m(alpha_db_km: float) -> float:
    return alpha_db_km / 4.343 * 1e-3


def make_reference_setup() -> tuple:
    """Build reference objects with shared parameters."""
    channel_obj = RefChannel(f_min=FREQ_MIN, f_max=FREQ_MAX, spacing=SPACING)

    # Identify classical (pump) channels
    classic_freqs = []
    for f in channel_obj.channel_list:
        if CLASSICAL_FREQ_RANGE[0] < f < CLASSICAL_FREQ_RANGE[1]:
            classic_freqs.append(f)
    classic_channel = np.array(classic_freqs)

    # Set channel types and PSD
    for i, f in enumerate(channel_obj.channel_list):
        if f in classic_channel:
            channel_obj.channel_type[i, 0] = 2   # classical
            channel_obj.power_spectral_density[i, 0] = P0 / SPACING  # PSD = P / Δf
        else:
            channel_obj.channel_type[i, 0] = 3   # quantum

    fiber_obj = RefFiber(channel=channel_obj)
    return channel_obj, fiber_obj, classic_channel


# ---------------------------------------------------------------------------
# Test 1: g_R equivalence
# ---------------------------------------------------------------------------

def test_g_R_equivalence():
    print("\n" + "=" * 70)
    print("TEST 1: g_R (Raman gain coefficient) equivalence")
    print("=" * 70)

    channel_obj, fiber_obj, classic_channel = make_reference_setup()

    ref_raman = RefRamanSolver(raman=None, fiber=fiber_obj)

    f_q_test = 194.4e12
    f_c_test = 191.15e12
    delta_f_test = abs(f_q_test - f_c_test)

    ref_g_R = ref_raman._get_raman_gain_coefficient(
        np.array([[f_c_test]]), np.array([[f_q_test]])
    )[0, 0]

    new_g_R = get_raman_gain(
        delta_f=delta_f_test,
        f_pump=f_c_test,
        A_eff=fiber_obj.get_effective_area,
    )

    print(f"  f_q = {f_q_test/1e12:.3f} THz, f_c = {f_c_test/1e12:.3f} THz, Δf = {delta_f_test/1e9:.1f} GHz")
    print(f"  reference g_R = {ref_g_R:.6e}  [1/(W·m)]")
    print(f"  new g_R       = {new_g_R:.6e}  [1/(W·m)]")
    print(f"  relative error = {abs(ref_g_R - new_g_R) / max(abs(ref_g_R), 1e-30):.2e}")

    assert np.isclose(ref_g_R, new_g_R, rtol=1e-9), "g_R mismatch!"
    print("  PASSED: g_R equivalent")


# ---------------------------------------------------------------------------
# Test 2: Full g_R matrix
# ---------------------------------------------------------------------------

def test_g_R_matrix():
    print("\n" + "=" * 70)
    print("TEST 2: Full g_R matrix comparison")
    print("=" * 70)

    channel_obj, fiber_obj, classic_channel = make_reference_setup()
    q_chs = np.array([f for f in channel_obj.channel_list if f not in classic_channel])

    ref_raman = RefRamanSolver(raman=None, fiber=fiber_obj)
    ref_g_R = ref_raman._get_raman_gain_coefficient(
        classic_channel.reshape(1, -1), q_chs.reshape(-1, 1)
    )

    delta_f = np.abs(q_chs.reshape(-1, 1) - classic_channel.reshape(1, -1))
    new_g_R = get_raman_gain(
        delta_f=delta_f,
        f_pump=classic_channel.reshape(1, -1),
        A_eff=fiber_obj.get_effective_area,
    )

    print(f"  shape: ref={ref_g_R.shape}, new={new_g_R.shape}")
    print(f"  ref g_R sum:  {ref_g_R.sum():.6e}")
    print(f"  new g_R sum:  {new_g_R.sum():.6e}")
    print(f"  max rel err: {np.max(np.abs(ref_g_R - new_g_R) / np.maximum(np.abs(ref_g_R), 1e-30)):.2e}")

    assert np.allclose(ref_g_R, new_g_R, rtol=1e-9), "g_R matrix mismatch!"
    print("  PASSED: g_R matrix equivalent")


# ---------------------------------------------------------------------------
# Test 3: SpRS cross-section matrix
# ---------------------------------------------------------------------------

def test_sprs_cross_section():
    print("\n" + "=" * 70)
    print("TEST 3: SpRS cross-section sigma matrix")
    print("=" * 70)

    channel_obj, fiber_obj, classic_channel = make_reference_setup()

    # Select a small number of quantum channels for clarity
    q_chs = np.array([f for f in channel_obj.channel_list if f not in classic_channel])
    q_chs = q_chs[:8]   # first 8 quantum channels
    c_chs = classic_channel[:3]    # first 3 classical channels

    print(f"  {len(q_chs)} quantum channels, {len(c_chs)} classical channels")

    # Reference sigma
    ref_raman = RefRamanSolver(raman=None, fiber=fiber_obj)
    ref_sigma = ref_raman._get_sprs_coff(c_chs.reshape(1, -1), q_chs.reshape(-1, 1))
    print(f"  reference sigma shape: {ref_sigma.shape}")
    print(f"  reference sigma sum:   {ref_sigma.sum():.6e}")

    # New sigma
    from qkd_sim.physical.noise.sprs_solver import _phonon_occupation, _raman_cross_section

    f_q = q_chs.reshape(-1, 1)
    f_c = c_chs.reshape(1, -1)
    delta_f = np.abs(f_q - f_c)

    g_R = get_raman_gain(
        delta_f=delta_f,
        f_pump=f_c,
        A_eff=fiber_obj.get_effective_area,
    )
    n_th = _phonon_occupation(delta_f, fiber_obj.get_temperature)
    new_sigma = _raman_cross_section(f_q, f_c, g_R, n_th, delta_f, bandwidth=SPACING)

    print(f"  new sigma shape:       {new_sigma.shape}")
    print(f"  new sigma sum:         {new_sigma.sum():.6e}")
    print(f"  ratio (new/ref):       {new_sigma.sum() / ref_sigma.sum():.6f}")

    # Show per-element comparison
    print(f"\n  ref sigma[0,:] = {ref_sigma[0,:]}")
    print(f"  new sigma[0,:] = {new_sigma[0,:]}")

    rel_err = np.abs(new_sigma - ref_sigma) / np.maximum(np.abs(ref_sigma), 1e-30)
    max_err = np.max(rel_err)
    print(f"\n  Max relative error: {max_err:.2e}")
    if max_err > 0.1:
        print("  WARNING: Large difference detected!")
        idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
        print(f"  Max at q_idx={idx[0]}, c_idx={idx[1]}: ref={ref_sigma[idx]:.6e}, new={new_sigma[idx]:.6e}")
        # Show the key values for diagnosis
        print(f"\n  Diagnosis:")
        print(f"    f_q = {q_chs[idx[0]]/1e12:.4f} THz, f_c = {c_chs[idx[1]]/1e12:.4f} THz")
        print(f"    delta_f = {delta_f[idx]:.3e} Hz")
        print(f"    channel spacing = {SPACING:.3e} Hz")
        print(f"    ratio = {delta_f[idx]/SPACING:.6f}")


# ---------------------------------------------------------------------------
# Test 4: Full forward noise
# ---------------------------------------------------------------------------

def test_forward_noise():
    print("\n" + "=" * 70)
    print("TEST 4: Forward SpRS noise power")
    print("=" * 70)

    channel_obj, fiber_obj, classic_channel = make_reference_setup()

    ref_raman = RefRamanSolver(raman=None, fiber=fiber_obj)
    ref_noise = ref_raman.get_intra_core_raman_noise(
        pump_channel=classic_channel,
        signal_channel=channel_obj.channel_list,
        z=np.array([FIBER_LENGTH_M]),
        direction='forward',
    )
    ref_noise = ref_noise.flatten()

    # Get quantum channel frequencies from reference
    q_chs = np.array([f for f in channel_obj.channel_list if f not in classic_channel])
    n_q = len(q_chs)
    n_total = len(channel_obj.channel_list)

    # Build a map from frequency to channel index
    freq_to_idx = {float(f): i for i, f in enumerate(channel_obj.channel_list)}
    quantum_indices = [freq_to_idx[float(f)] for f in q_chs]
    classical_indices = [freq_to_idx[float(f)] for f in classic_channel]

    # Build WDMConfig with quantum_channel_indices = indices of quantum channels
    wdm_config = WDMConfig(
        start_freq=FREQ_MIN,
        start_channel=1,
        end_channel=n_total,
        channel_spacing=SPACING,
        B_s=SPACING,
        P0=P0,
        beta_rolloff=None,
        quantum_channel_indices=quantum_indices,
    )
    f_grid = channel_obj.channel_list.copy()

    # Build grid with discrete (single-frequency) spectrum
    grid = build_wdm_grid(
        config=wdm_config,
        spectrum_type=SpectrumType.SINGLE_FREQ,
        f_grid=f_grid,
        classical_channel_indices=classical_indices,
    )

    fiber_cfg = FiberConfig(
        alpha_dB_per_km=FIBER_ALPHA_DB_PER_KM,
        gamma_per_W_km=1.3,
        D_ps_nm_km=17.0,
        D_slope_ps_nm2_km=0.056,
        L_km=FIBER_LENGTH_M / 1e3,
        A_eff=80e-12,
        rayleigh_coeff=4.8e-8,
        T_kelvin=FIBER_TEMP_K,
    )
    from qkd_sim.physical.fiber import Fiber as NewFiber
    new_fiber = NewFiber(fiber_cfg)

    solver = DiscreteSPRSSolver()
    new_noise = solver.compute_forward(new_fiber, grid)

    print(f"  Reference: {len(ref_noise)} channels, noise range [{ref_noise.min():.3e}, {ref_noise.max():.3e}] W")
    print(f"  New:      {len(new_noise)} channels, noise range [{new_noise.min():.3e}, {new_noise.max():.3e}] W")

    # Match by frequency — show first few quantum channels
    q_freqs_new = np.array([ch.f_center for ch in grid.get_quantum_channels()])
    print(f"\n  First 10 quantum channels:")
    for i, f_q in enumerate(q_freqs_new[:10]):
        ref_mask = np.isclose(channel_obj.channel_list, f_q)
        if np.any(ref_mask):
            ref_n = ref_noise[ref_mask][0]
            ratio = new_noise[i] / ref_n if ref_n > 0 else float('nan')
            print(f"  f={f_q/1e12:.3f} THz: ref={ref_n:.3e} W, new={new_noise[i]:.3e} W, ratio={ratio:.4f}")


if __name__ == "__main__":
    test_g_R_equivalence()
    test_g_R_matrix()
    test_sprs_cross_section()
    test_forward_noise()
    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70)
