"""BB84 QKD 安全码率计算。

公式来源: docs/formulas_skr.md
支持三种模型：
  - infinite_key_rate      : 无限长密钥
  - approx_finite_key_rate : 近似有限长（3 态诱骗 + Gaussian 修正）
  - strict_finite_key_rate : 严格有限长（1-decoy BB84，X/Z 双基矢，Hoeffding 不等式）

p_noise 参数用于注入外部噪声（如 SpRS 散射）的光子计数概率，与 SKRConfig.noise_count_prob 累加。
"""

from __future__ import annotations

import math

import numpy as np

from qkd_sim.config.schema import FiberConfig, SKRConfig


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def H2(x: float) -> float:
    """二元香农熵 h(x) = -x·log₂x - (1-x)·log₂(1-x)。"""
    x = min(max(float(x), 1e-15), 1.0 - 1e-15)
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def skr_bps_to_bit_per_pulse(skr_bps: float, R_rep: float) -> float:
    """Convert SKR from bit/s to bit/pulse. Returns 0.0 if R_rep <= 0."""
    return skr_bps / R_rep if R_rep > 0.0 else 0.0


# ---------------------------------------------------------------------------
# 内部：信道中间量
# ---------------------------------------------------------------------------

def _channel_quantities(
    distance_m: float,
    fiber_cfg: FiberConfig,
    skr_cfg: SKRConfig,
    mu: float,
    p_noise: float = 0.0,
) -> tuple[float, float, float, float]:
    """计算给定发射强度 μ 的信道中间量。

    Returns
    -------
    eta : float
        信道透过率（含 SPD 效率和插入损耗）
    Y0 : float
        真空态产率（暗计数 + 噪声）
    Q_mu : float
        总增益（公式 8）
    E_mu : float
        总误码率 QBER（公式 9）
    """
    distance_m = float(distance_m)
    eta = skr_cfg.eta_spd * np.exp(-fiber_cfg.alpha * distance_m) * skr_cfg.IL
    p_noise_total = skr_cfg.noise_count_prob + p_noise
    Y0 = 1.0 - (1.0 - skr_cfg.dark_count_prob - p_noise_total) ** 2
    Y0 = max(Y0, 0.0)

    e0 = 0.5
    exp_term = np.exp(-mu * eta)
    Q_mu = float(Y0 + 1.0 - exp_term)
    E_mu = float((e0 * Y0 + skr_cfg.e_det * (1.0 - exp_term)) / max(Q_mu, 1e-30))
    return float(eta), float(Y0), Q_mu, E_mu


# ---------------------------------------------------------------------------
# 模型一：无限长密钥
# ---------------------------------------------------------------------------

def infinite_key_rate(
    distance_m: float,
    fiber_cfg: FiberConfig,
    skr_cfg: SKRConfig,
    p_noise: float = 0.0,
) -> tuple[float, float, float]:
    """无限长密钥 BB84 安全码率。

    参见 docs/formulas_skr.md §1。

    Parameters
    ----------
    distance_m : float
        光纤传输距离 [m]
    fiber_cfg : FiberConfig
        光纤配置（提供 alpha）
    skr_cfg : SKRConfig
        SKR 系统配置
    p_noise : float
        外部噪声光子计数概率（SpRS 等），默认 0

    Returns
    -------
    skr_bps : float
        安全码率 [bit/s]，≥ 0
    skr_bit_per_pulse : float
        安全码率 [bit/pulse]，≥ 0
    qber : float
        总误码率 E_μ
    """
    mu = skr_cfg.mu_signal
    eta, Y0, Q_mu, E_mu = _channel_quantities(distance_m, fiber_cfg, skr_cfg, mu, p_noise)

    e0 = 0.5
    Y1 = Y0 + eta
    Q1 = Y1 * mu * math.exp(-mu)
    e1 = (e0 * Y0 + skr_cfg.e_det * eta) / max(Y1, 1e-30)
    e1 = _clip(0.0, 0.5, e1)

    skr = skr_cfg.q_sifting * (-Q_mu * skr_cfg.f_ec * H2(E_mu) + Q1 * (1.0 - H2(e1)))
    skr_bps = float(max(0.0, skr)) * float(skr_cfg.R_rep)
    return skr_bps, skr_bps_to_bit_per_pulse(skr_bps, skr_cfg.R_rep), float(E_mu)


# ---------------------------------------------------------------------------
# 模型二：近似有限长（3 态诱骗 + Gaussian 修正）
# ---------------------------------------------------------------------------

def approx_finite_key_rate(
    distance_m: float,
    fiber_cfg: FiberConfig,
    skr_cfg: SKRConfig,
    p_noise: float = 0.0,
) -> tuple[float, float, float]:
    """近似有限长 BB84 安全码率（三态诱骗：信号 μ + 诱骗 ν + 真空 0）。

    参见 docs/formulas_skr.md §2（公式 1–9）。

    Gaussian 有限长修正施加在诱骗态增益 Q_ν 和误差积 E_ν·Q_ν 上。

    Returns
    -------
    skr_bps : float
        安全码率 [bit/s]，≥ 0
    skr_bit_per_pulse : float
        安全码率 [bit/pulse]，≥ 0
    qber : float
        信号态总误码率 E_μ
    """
    mu = skr_cfg.mu_signal
    nu = skr_cfg.mu_decoy
    p_mu = skr_cfg.p_signal
    p_nu = skr_cfg.p_decoy
    # approx_finite_key_rate 仅支持 mode=alice
    if skr_cfg.block_length.mode != "alice":
        raise ValueError("approx_finite_key_rate 仅支持 block_length.mode='alice'，请使用 strict_finite_key_rate")
    N = skr_cfg.block_length.N_alice
    gamma = skr_cfg.gamma_ks
    e0 = 0.5

    eta, Y0, Q_mu, E_mu = _channel_quantities(distance_m, fiber_cfg, skr_cfg, mu, p_noise)
    _, _, Q_nu, E_nu = _channel_quantities(distance_m, fiber_cfg, skr_cfg, nu, p_noise)

    # 公式 (6)(7)：对诱骗态增益和误差积施加 Gaussian 有限长修正
    denom_Q = max(p_nu * Q_nu * N / 2.0, 1e-30)
    Q_nu_L = Q_nu * (1.0 - gamma / math.sqrt(denom_Q))
    Q_nu_L = max(Q_nu_L, 0.0)

    EnuQnu = E_nu * Q_nu
    denom_E = max(p_nu * EnuQnu * N / 2.0, 1e-30)
    EnuQnu_U = EnuQnu * (1.0 + gamma / math.sqrt(denom_E))
    EnuQnu_U = max(EnuQnu_U, 0.0)

    # 公式 (5)：Y₁ 下界（从诱骗态分析推导）
    denom_Y1 = mu * nu - nu ** 2
    if abs(denom_Y1) < 1e-30:
        return 0.0, 0.0, float(E_mu)
    Y1_L = (mu / denom_Y1) * (
        Q_nu_L * math.exp(nu)
        - (nu ** 2 / mu ** 2) * Q_mu * math.exp(mu)
        - (mu ** 2 - nu ** 2) / mu ** 2 * Y0
    )
    Y1_L = max(Y1_L, 0.0)

    # 公式 (2)(3)：Q₁ 下界和 e₁ 上界
    Q1_L = Y1_L * mu * math.exp(-mu)

    if Y1_L < 1e-30:
        return 0.0, 0.0, float(E_mu)
    e1_U = (EnuQnu_U * math.exp(nu) - e0 * Y0) / (nu * Y1_L)
    e1_U = _clip(0.0, 0.5, e1_U)

    # 公式 (1)：安全码率
    skr = p_mu * skr_cfg.q_sifting * (
        -Q_mu * skr_cfg.f_ec * H2(E_mu) + Q1_L * (1.0 - H2(e1_U))
    )
    skr_bps = float(max(0.0, skr)) * float(skr_cfg.R_rep)
    return skr_bps, skr_bps_to_bit_per_pulse(skr_bps, skr_cfg.R_rep), float(E_mu)


# ---------------------------------------------------------------------------
# 模型三：严格有限长（1-decoy BB84，X/Z 双基矢分离）
# ---------------------------------------------------------------------------

def _hoeffding_delta(n: float, epsilon: float) -> float:
    """Hoeffding 浓度不等式偏差 δ(n,ε) = √(n·ln(1/ε)/2)。"""
    return math.sqrt(n * math.log(1.0 / max(epsilon, 1e-300)) / 2.0)


def _azuma_delta(n: float, epsilon: float) -> float:
    """Azuma 浓度不等式偏差 δ(n,ε) = √(2·n·ln(1/ε))。"""
    return math.sqrt(2.0 * n * math.log(1.0 / max(epsilon, 1e-300)))


def _concentration_delta(n: float, epsilon: float, method: str) -> float:
    if method == "Hoeffding":
        return _hoeffding_delta(n, epsilon)
    return _azuma_delta(n, epsilon)


def _tau(m: int, mu_1: float, mu_2: float, P_mu_1: float, P_mu_2: float) -> float:
    """Bayes 先验权重 τ_m = Σ_k p_k·exp(-k)·k^m/m!。"""
    fac_m = math.factorial(m)
    t1 = P_mu_1 * math.exp(-mu_1) * mu_1 ** m / fac_m
    t2 = P_mu_2 * math.exp(-mu_2) * mu_2 ** m / fac_m
    return t1 + t2


def _gamma_serfling(
    epsilon: float,
    lambda_z: float,
    s_X1: float,
    s_Z1: float,
    improved: bool = False,
) -> float:
    """Serfling 不等式修正 γ（从 Z 基误差估计 X 基相位误差）。"""
    c = s_X1
    d = s_Z1
    b = lambda_z
    if c <= 0 or d <= 0:
        return 0.5
    if improved:
        denom = c * d * math.log(2)
        arg = (c + d) / (c * d * (1.0 - b) * b * epsilon ** 2)
        return math.sqrt((c + d) * (1.0 - b) * b / denom * math.log2(arg))
    else:
        return math.sqrt(((d + c) / (d * c)) * ((c + 1.0) / c) * math.log(1.0 / max(epsilon, 1e-300)))


def strict_finite_key_rate(
    distance_m: float,
    fiber_cfg: FiberConfig,
    skr_cfg: SKRConfig,
    p_noise: float = 0.0,
    asymptotic: bool = False,
    with_vacuum: bool = False,
    optimize_params: bool = True,
) -> tuple[float, float, float]:
    """严格有限长 BB84 安全码率（1-decoy，X/Z 双基矢分离）。

    基于 Wiesemann et al. (arXiv:2405.16578)，参见 docs/formulas_skr.md §3。

    密钥在 X 基生成，相位误差在 Z 基估计。
    使用 Hoeffding 或 Azuma 浓度不等式（由 skr_cfg.concentration_method 控制）。

    Parameters
    ----------
    distance_m : float
        光纤传输距离 [m]
    fiber_cfg : FiberConfig
        光纤配置（提供 alpha）
    skr_cfg : SKRConfig
        SKR 系统配置（mu_signal 为 μ₁，mu_decoy 为 μ₂，mu_signal > mu_decoy 必须成立）
    p_noise : float
        外部噪声光子计数概率，默认 0
    asymptotic : bool
        True 时忽略有限长修正（用于验证收敛性），默认 False
    with_vacuum : bool
        True 时将真空事件 s_X0^- 计入密钥长度，默认 False
    optimize_params : bool
        True 时使用 Nelder-Mead 优化 4 个自由参数（μ₁, μ₂, p_signal, P_X_alice），
        默认 True。优化结果通过顺序热启动加速，优化失败时回退到固定参数。

    Returns
    -------
    skr_bps : float
        安全码率 [bit/s]，≥ 0
    skr_bit_per_pulse : float
        安全码率 [bit/pulse]，≥ 0
    qber : float
        X 基总误码率
    """
    # --- 参数优化路径 ---
    if optimize_params:
        from qkd_sim.physical.skr.skr_optimizer import SKROptimizer
        optimizer = SKROptimizer(fiber_cfg, skr_cfg, p_noise=p_noise)
        opt_result = optimizer.optimize_distance(distance_m)
        if opt_result.optimal_skr_bps > 0.0:
            return (
                opt_result.optimal_skr_bps,
                skr_bps_to_bit_per_pulse(opt_result.optimal_skr_bps, skr_cfg.R_rep),
                opt_result.qber,
            )
        # 优化失败（SKR <= 0），回退到固定参数计算
        # fall through to the fixed-param path below

    mu_1 = skr_cfg.mu_signal    # signal intensity
    mu_2 = skr_cfg.mu_decoy     # decoy intensity
    P_mu_1 = skr_cfg.p_signal
    P_mu_2 = skr_cfg.p_decoy

    P_X_A = skr_cfg.P_X_alice
    P_X_B = skr_cfg.P_X_bob
    P_Z_A = 1.0 - P_X_A
    P_Z_B = 1.0 - P_X_B

    R_0 = skr_cfg.R_0

    # --- 块长解析：mode=alice 或 mode=bob ---
    bl = skr_cfg.block_length
    if bl.mode == "alice":
        N_A = bl.N_alice
        integration_time = N_A / R_0
    else:
        # fixed-N_bob: n_bob = t * R_0 * P_XX * P_det_1
        # => t = n_bob / (R_0 * P_XX * P_det_1)
        integration_time = bl.N_bob / (R_0 * P_XX * P_det_1)
        N_A = integration_time * R_0  # 等效 Alice 发送脉冲数（用于统计涨落）

    epsilon_cor = skr_cfg.epsilon_cor
    epsilon_sec = skr_cfg.epsilon_sec
    epsilon_0 = epsilon_sec / 15.0
    epsilon_1 = epsilon_0
    epsilon_2 = epsilon_0

    method = skr_cfg.concentration_method

    # --- 信道透过率和暗计数 ---
    eta_sys = (
        skr_cfg.eta_spd
        * math.exp(-fiber_cfg.alpha * distance_m)
        * skr_cfg.IL
    )
    p_noise_total = skr_cfg.noise_count_prob + p_noise
    P_DC = skr_cfg.dark_count_prob + p_noise_total

    P_XX = P_X_A * P_X_B
    P_ZZ = P_Z_A * P_Z_B

    # 先验检测概率
    def _P_det(mu: float) -> float:
        return 1.0 - math.exp(-mu * eta_sys) * (1.0 - P_DC)

    P_det_1 = _P_det(mu_1)
    P_det_2 = _P_det(mu_2)

    # 误码概率（给定检测）
    def _P_err(mu: float) -> float:
        return (1.0 - math.exp(-mu * eta_sys)) * skr_cfg.e_det + P_DC / 2.0

    P_err_1 = _P_err(mu_1)
    P_err_2 = _P_err(mu_2)

    # --- 检测数（X 基、Z 基） ---
    def _counts(P_basis: float, P_det: float) -> float:
        return integration_time * R_0 * P_basis * P_det

    n_X_1 = _counts(P_XX * P_mu_1, P_det_1)
    n_X_2 = _counts(P_XX * P_mu_2, P_det_2)
    n_X = n_X_1 + n_X_2

    n_Z_1 = _counts(P_ZZ * P_mu_1, P_det_1)
    n_Z_2 = _counts(P_ZZ * P_mu_2, P_det_2)
    n_Z = n_Z_1 + n_Z_2

    # --- 误码数（X 基、Z 基） ---
    def _errors(P_basis: float, P_mu: float, P_err_mu: float) -> float:
        return integration_time * R_0 * P_basis * P_mu * P_err_mu

    c_X_1 = _errors(P_XX, P_mu_1, P_err_1)
    c_X_2 = _errors(P_XX, P_mu_2, P_err_2)
    c_X = c_X_1 + c_X_2

    c_Z_1 = _errors(P_ZZ, P_mu_1, P_err_1)
    c_Z_2 = _errors(P_ZZ, P_mu_2, P_err_2)

    # QBER for reporting
    qber = c_X / max(n_X, 1e-30)

    if asymptotic:
        def _delta(_n: float, _eps: float) -> float:
            return 0.0
    else:
        def _delta(n: float, eps: float) -> float:  # type: ignore[misc]
            return _concentration_delta(n, eps, method)

    # --- 统计边界（Hoeffding / Azuma 浓度不等式） ---
    n_X_1_plus  = _clip(0.0, n_X, n_X_1 + _delta(n_X, epsilon_1))
    n_X_2_minus = _clip(0.0, n_X, n_X_2 - _delta(n_X, epsilon_1))

    c_X_1_plus  = _clip(0.0, c_X, c_X_1 + _delta(c_X, epsilon_2))
    c_X_2_plus  = _clip(0.0, c_X, c_X_2 + _delta(c_X, epsilon_2))
    c_X_2_minus = _clip(0.0, c_X, c_X_2 - _delta(c_X, epsilon_2))

    n_Z_1_plus  = _clip(0.0, n_Z, n_Z_1 + _delta(n_Z, epsilon_1))
    n_Z_2_minus = _clip(0.0, n_Z, n_Z_2 - _delta(n_Z, epsilon_1))

    c_Z_1_plus  = _clip(0.0, c_Z_1 + c_Z_2, c_Z_1 + _delta(c_Z_1 + c_Z_2, epsilon_2))
    c_Z_2_plus  = _clip(0.0, c_Z_1 + c_Z_2, c_Z_2 + _delta(c_Z_1 + c_Z_2, epsilon_2))
    c_Z_2_minus = _clip(0.0, c_Z_1 + c_Z_2, c_Z_2 - _delta(c_Z_1 + c_Z_2, epsilon_2))

    tau0 = _tau(0, mu_1, mu_2, P_mu_1, P_mu_2)
    tau1 = _tau(1, mu_1, mu_2, P_mu_1, P_mu_2)

    # --- 真空事件边界 ---
    s_X_0_minus_raw = (tau0 / (mu_1 - mu_2)) * (
        mu_1 * math.exp(mu_2) * n_X_2_minus / P_mu_2
        - mu_2 * math.exp(mu_1) * n_X_1_plus / P_mu_1
    )
    s_X_0_minus = _clip(0.0, n_X, s_X_0_minus_raw)

    def _s0_plus(n_xi_plus: float, c_xi_plus: float, n_tot: float, mu_i: float, p_i: float) -> float:
        return 2.0 * (tau0 * (math.exp(mu_i) / p_i) * c_xi_plus + _delta(n_tot, epsilon_1))

    s_X_0_plus = _clip(0.0, n_X, min(
        _s0_plus(n_X_1_plus, c_X_1_plus, n_X, mu_1, P_mu_1),
        _s0_plus(n_X_2_plus if (n_X_2_plus := _clip(0.0, n_X, n_X_2 + _delta(n_X, epsilon_1))) else 0.0,
                 c_X_2_plus, n_X, mu_2, P_mu_2),
    ))
    s_Z_0_plus = _clip(0.0, n_Z, min(
        _s0_plus(n_Z_1_plus, c_Z_1_plus, n_Z, mu_1, P_mu_1),
        _s0_plus(_clip(0.0, n_Z, n_Z_2 + _delta(n_Z, epsilon_1)), c_Z_2_plus, n_Z, mu_2, P_mu_2),
    ))

    # --- 单光子事件边界 ---
    def _s1_minus(n_i2_minus: float, n_i1_plus: float, s0_plus: float, n_tot: float) -> float:
        val = (mu_1 * tau1 / (mu_2 * (mu_1 - mu_2))) * (
            math.exp(mu_2) * n_i2_minus / P_mu_2
            - (mu_2 ** 2 / mu_1 ** 2) * math.exp(mu_1) * n_i1_plus / P_mu_1
            - (mu_1 ** 2 - mu_2 ** 2) / (mu_1 ** 2 * tau0) * s0_plus
        )
        return _clip(0.0, n_tot, val)

    s_X_1_minus = _s1_minus(n_X_2_minus, n_X_1_plus, s_X_0_plus, n_X)
    s_Z_1_minus = _s1_minus(n_Z_2_minus, n_Z_1_plus, s_Z_0_plus, n_Z)

    # --- 单光子误码上界（Z 基） ---
    v_Z_1_plus_raw = (tau1 / (mu_1 - mu_2)) * (
        math.exp(mu_1) * c_Z_1_plus / P_mu_1
        - math.exp(mu_2) * c_Z_2_minus / P_mu_2
    )
    v_Z_1_plus = _clip(0.0, n_Z, v_Z_1_plus_raw)

    # --- QBER 上界（X 基相位误差） ---
    if s_Z_1_minus > 0.0:
        lambda_Z_plus = _clip(0.0, 0.5, v_Z_1_plus / s_Z_1_minus)
    else:
        lambda_Z_plus = 0.5

    if s_X_1_minus > 0.0 and s_Z_1_minus > 0.0:
        gamma_val = _gamma_serfling(epsilon_0, lambda_Z_plus, s_X_1_minus, s_Z_1_minus)
        lambda_X_plus = _clip(0.0, 0.5, lambda_Z_plus + gamma_val)
    else:
        lambda_X_plus = 0.5

    # --- 纠错泄漏 ---
    qber_obs = c_X / max(n_X, 1e-30)
    leak_EC = n_X * skr_cfg.f_ec * H2(_clip(1e-15, 1.0 - 1e-15, qber_obs))

    # --- 最终密钥长度 ---
    s_vac = s_X_0_minus if with_vacuum else 0.0

    if not asymptotic:
        l_max = (
            s_vac
            + s_X_1_minus * (1.0 - H2(lambda_X_plus))
            - leak_EC
            - math.log2(2.0 / max(epsilon_cor, 1e-300))
            - 4.0 * math.log2(15.0 / (max(epsilon_sec, 1e-300) * 2.0 ** 0.25))
        )
    else:
        l_max = s_vac + s_X_1_minus * (1.0 - H2(lambda_X_plus)) - leak_EC

    l_max = max(l_max, 0.0)
    skr_bps = l_max / integration_time
    return float(skr_bps), skr_bps_to_bit_per_pulse(float(skr_bps), skr_cfg.R_0), float(qber)
