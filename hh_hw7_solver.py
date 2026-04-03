import math
import numpy as np

# ----------------------
# Parameters (default)
# ----------------------
params = {
    "C": 1.0,          # (uA*ms)/mV
    "g": 1.0,          # (uA/mV)/cm^2
    "r_over_rho": 1.0, # r/ρ
    "I_Na": 150.0,
    "I_K": 30.0,
    "v_star": 20.0,
    "tau": 10.0,
}

# ----------------------
# Utility
# ----------------------

def safe_log(x):
    if x <= 0:
        return None
    return math.log(x)


def newton_solve(func, x0, max_iter=50, tol=1e-10, step_tol=1e-12, verbose=False, valid_fn=None):
    x = np.array(x0, dtype=float)
    for it in range(max_iter):
        f = np.array(func(x))
        norm = np.linalg.norm(f, ord=2)
        if norm < tol:
            return x, True
        # numerical Jacobian
        J = np.zeros((len(x), len(x)), dtype=float)
        eps = 1e-6
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = eps * max(1.0, abs(x[i]))
            f1 = np.array(func(x + dx))
            J[:, i] = (f1 - f) / dx[i]
        try:
            delta = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            if verbose:
                print("Jacobian singular at iter", it)
            return x, False
        # damping
        step = 1.0
        improved = False
        for _ in range(12):
            x_new = x + step * delta
            if valid_fn is not None and (not valid_fn(x_new)):
                step *= 0.5
                continue
            f_new = np.array(func(x_new))
            if np.linalg.norm(f_new, ord=2) < norm:
                x = x_new
                improved = True
                break
            step *= 0.5
        if not improved:
            if np.linalg.norm(step * delta, ord=2) < step_tol:
                return x, False
            x = x + step * delta
    return x, False


# ----------------------
# Space-independent periodic solutions
# ----------------------

def T0_T1_from_n(n0, n1, tau):
    if n0 <= 0 or n1 <= 0 or n0 >= 1 or n1 >= 1 or n1 <= n0:
        return None, None
    t0 = tau * math.log(n1 / n0)
    t1 = tau * math.log((1 - n0) / (1 - n1))
    return t0, t1


def v0_to_vstar_residual(n0, n1, I0, p):
    # Interval [-T0, 0], m=0. Enforce v(0)=v*
    C = p["C"]; g = p["g"]; I_K = p["I_K"]; tau = p["tau"]; v_star = p["v_star"]
    t0, t1 = T0_T1_from_n(n0, n1, tau)
    if t0 is None:
        return 1e6
    alpha = g / C
    # v(0), with n(t)=n0*exp(-t/tau) on [-T0,0]
    term1 = v_star * math.exp(-alpha * t0)
    term2 = (I0 / (C * alpha)) * (1.0 - math.exp(-alpha * t0))
    denom = (alpha - 1.0 / tau)
    term3 = -(I_K * n0 / C) * (1.0 - math.exp(-denom * t0)) / denom
    v0 = term1 + term2 + term3
    return v0 - v_star


def vT1_to_vstar_residual(n0, n1, I0, p):
    # Interval (0, T1), m=1. Enforce v(T1)=v*
    C = p["C"]; g = p["g"]; I_K = p["I_K"]; I_Na = p["I_Na"]; tau = p["tau"]; v_star = p["v_star"]
    t0, t1 = T0_T1_from_n(n0, n1, tau)
    if t1 is None:
        return 1e6
    alpha = g / C
    const = (I0 - I_K) / C
    exp_coeff = (I_Na + I_K) * (1.0 - n0) / C
    term1 = v_star
    term2 = const * (math.exp(alpha * t1) - 1.0) / alpha
    denom = (alpha - 1.0 / tau)
    term3 = exp_coeff * (math.exp(denom * t1) - 1.0) / denom
    vT1 = math.exp(-alpha * t1) * (term1 + term2 + term3)
    return vT1 - v_star


def solve_space_independent(I0, p, x0=None):
    # Solve for n0, n1
    g = p["g"]; v_star = p["v_star"]; I_K = p["I_K"]; I_Na = p["I_Na"]
    # initial guess from C->0
    n0_guess = (I0 - g * v_star) / I_K
    n1_guess = (I0 + I_Na - g * v_star) / (I_Na + I_K)
    if x0 is None:
        x0 = [n0_guess, n1_guess]

    def func(x):
        n0, n1 = x
        return [
            v0_to_vstar_residual(n0, n1, I0, p),
            vT1_to_vstar_residual(n0, n1, I0, p),
        ]

    def valid(x):
        n0, n1 = x
        return (0 < n0 < n1 < 1)

    sol, ok = newton_solve(func, x0, valid_fn=valid)
    if not ok:
        return None
    n0, n1 = sol
    t0, t1 = T0_T1_from_n(n0, n1, p["tau"])
    return {"n0": n0, "n1": n1, "T0": t0, "T1": t1}


# ----------------------
# Traveling wave exact solution
# ----------------------

def lambdas(C, g, a):
    # Solve a λ^2 - C λ - g = 0
    disc = C * C + 4.0 * a * g
    sqrt_disc = math.sqrt(disc)
    lam_plus = (C + sqrt_disc) / (2.0 * a)
    lam_minus = (C - sqrt_disc) / (2.0 * a)
    return lam_minus, lam_plus


def solve_traveling_wave(I0, w, p, x0):
    C = p["C"]; g = p["g"]; r_over_rho = p["r_over_rho"]
    I_K = p["I_K"]; I_Na = p["I_Na"]; tau = p["tau"]; v_star = p["v_star"]

    if w <= 0:
        return None

    a = r_over_rho / (2.0 * w * w)
    lam_minus, lam_plus = lambdas(C, g, a)

    def func(x):
        N0, N1 = x
        # validity
        if not (0 < N0 < N1 < 1):
            return [1e6, 1e6]
        T0, T1 = T0_T1_from_n(N0, N1, tau)
        if T0 is None:
            return [1e6, 1e6]

        # Interval A: [-T0, 0], M=0
        A0 = I0 / g
        denomA = (g - C / tau - a / (tau * tau))
        A1 = -I_K * N0 / denomA

        E_p = math.exp(-lam_plus * T0)
        E_m = math.exp(lam_minus * T0)
        # Solve for c1, c2
        b1 = v_star - (A0 + A1 * math.exp(T0 / tau))
        b2 = v_star - (A0 + A1)
        # system: c1 + c2 E_p = b1; c1 E_m + c2 = b2
        det = 1.0 - E_p * E_m
        if abs(det) < 1e-12:
            return [1e6, 1e6]
        c1 = (b1 - b2 * E_p) / det
        c2 = (b2 - b1 * E_m) / det

        # Interval B: (0, T1), M=1
        B0 = (I0 - I_K) / g
        denomB = (g - C / tau - a / (tau * tau))
        B1 = (I_Na + I_K) * (1.0 - N0) / denomB

        E_p1 = math.exp(-lam_plus * T1)
        E_m1 = math.exp(lam_minus * T1)
        b3 = v_star - (B0 + B1)
        b4 = v_star - (B0 + B1 * math.exp(-T1 / tau))
        # system: d1 + d2 E_p1 = b3; d1 E_m1 + d2 = b4
        det2 = 1.0 - E_p1 * E_m1
        if abs(det2) < 1e-12:
            return [1e6, 1e6]
        d1 = (b3 - b4 * E_p1) / det2
        d2 = (b4 - b3 * E_m1) / det2

        # Slopes
        VA_0 = (-A1 / tau) + c1 * lam_minus * E_m + c2 * lam_plus
        VB_0 = (-B1 / tau) + d1 * lam_minus + d2 * lam_plus * E_p1
        VA_mT0 = (-A1 / tau) * math.exp(T0 / tau) + c1 * lam_minus + c2 * lam_plus * E_p
        VB_T1 = (-B1 / tau) * math.exp(-T1 / tau) + d1 * lam_minus * E_m1 + d2 * lam_plus

        return [VA_0 - VB_0, VA_mT0 - VB_T1]

    def valid(x):
        N0, N1 = x
        return 0 < N0 < N1 < 1

    sol, ok = newton_solve(func, x0, valid_fn=valid)
    if not ok:
        return None
    N0, N1 = sol
    T0, T1 = T0_T1_from_n(N0, N1, tau)
    return {"N0": N0, "N1": N1, "T0": T0, "T1": T1, "lam_minus": lam_minus, "lam_plus": lam_plus}


# ----------------------
# Approximate traveling wave (epsilon -> 0)
# ----------------------

def gamma_from_theta(theta):
    # gamma = (sqrt(1+2/theta^2)+1)/(sqrt(1+2/theta^2)-1)
    s = math.sqrt(1.0 + 2.0 / (theta * theta))
    return (s + 1.0) / (s - 1.0)


def N_from_gamma(gamma, I0, p):
    g = p["g"]; v_star = p["v_star"]; I_K = p["I_K"]; I_Na = p["I_Na"]
    num = (I0 + I_Na - g * v_star) - gamma * (g * v_star - I0)
    den = (I_Na + I_K) + gamma * I_K
    return num / den


# ----------------------
# Plot helpers
# ----------------------

def compute_space_independent_curves(I0_values, p):
    exact = []
    approx = []
    for I0 in I0_values:
        # approx
        g = p["g"]; v_star = p["v_star"]; I_K = p["I_K"]; I_Na = p["I_Na"]
        n0 = (I0 - g * v_star) / I_K
        n1 = (I0 + I_Na - g * v_star) / (I_Na + I_K)
        t0, t1 = T0_T1_from_n(n0, n1, p["tau"])
        if t0 is not None:
            approx.append((I0, t0 + t1))
        # exact
        sol = solve_space_independent(I0, p)
        if sol is not None:
            exact.append((I0, sol["T0"] + sol["T1"]))
    return np.array(approx), np.array(exact)


def compute_traveling_wave_curves(I0, theta_values, p):
    C = p["C"]; g = p["g"]; r_over_rho = p["r_over_rho"]
    L = math.sqrt(r_over_rho / g)
    T = C / g

    approx_pts = []
    exact_pts = []

    # use continuation for exact
    prev_sol = None

    for theta in theta_values:
        if theta <= 0:
            continue
        w = theta * L / T
        gamma = gamma_from_theta(theta)
        # approximate
        N0 = N_from_gamma(gamma, I0, p)
        N1 = N_from_gamma(1.0 / gamma, I0, p)
        T0, T1 = T0_T1_from_n(N0, N1, p["tau"])
        if T0 is not None:
            approx_pts.append((T0 + T1, w))
        # exact solver
        if T0 is None:
            prev_sol = None
            continue
        x0 = [N0, N1] if prev_sol is None else [prev_sol["N0"], prev_sol["N1"]]
        sol = solve_traveling_wave(I0, w, p, x0)
        if sol is None:
            prev_sol = None
            continue
        prev_sol = sol
        exact_pts.append((sol["T0"] + sol["T1"], w))

    return np.array(approx_pts), np.array(exact_pts)


if __name__ == "__main__":
    import os
    # avoid matplotlib cache issue
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = params

    # 1) Space-independent periodic solutions: period vs I0
    I0_vals = np.linspace(21, 49, 50)
    approx_si, exact_si = compute_space_independent_curves(I0_vals, p)
    plt.figure(figsize=(7,5))
    if len(approx_si) > 0:
        plt.plot(approx_si[:,0], approx_si[:,1], "--", label="C → 0 approx")
    if len(exact_si) > 0:
        plt.plot(exact_si[:,0], exact_si[:,1], "-", label="Exact C=1")
    plt.xlabel(r"$I_0$ ($\mu A/cm^2$)")
    plt.ylabel("Period T (ms)")
    plt.title("Space-independent periodic solutions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/dzwlalala/Documents/New project/space_independent_period.png", dpi=150)

    # 2) Traveling wave: w vs period for several I0
    I0_list = [10, 16, 20, 30, 35]
    theta_vals = np.linspace(0.2, 4.0, 60)
    plt.figure(figsize=(7,5))
    for I0 in I0_list:
        approx_tw, exact_tw = compute_traveling_wave_curves(I0, theta_vals, p)
        if len(approx_tw) > 0:
            plt.plot(approx_tw[:,0], approx_tw[:,1], "--", label=f"approx I0={I0}")
        if len(exact_tw) > 0:
            plt.plot(exact_tw[:,0], exact_tw[:,1], "-", label=f"exact I0={I0}")
    plt.xlabel("Period T0 + T1 (ms)")
    plt.ylabel("Wave speed w (cm/ms)")
    plt.title("Traveling wave: w vs period (exact vs C→0)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("/Users/dzwlalala/Documents/New project/traveling_wave_w_vs_period.png", dpi=150)

    print("Wrote plots:")
    print("  /Users/dzwlalala/Documents/New project/space_independent_period.png")
    print("  /Users/dzwlalala/Documents/New project/traveling_wave_w_vs_period.png")
