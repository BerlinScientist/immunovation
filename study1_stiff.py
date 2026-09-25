"""Study 1: A stiff cytokine/Treg toy model, explicit vs. implicit Euler.

Illustrative model (not fitted to data). Time in days, concentrations in arbitrary units.
    dC/dt = s - k*C - c*R*C          cytokine (fast: short half-life, consumed by Tregs)
    dR/dt = r*R*C/(K + C) - d*R      Treg population (slow: days)
The methods are the standard explicit and implicit Euler schemes, applied to a nonlinear two-variable system.
"""
import json
import os
import numpy as np
from scipy.integrate import solve_ivp

P = dict(s=100.0, k=50.0, c=0.5, K=1.0, r=1.2, d=0.5)


def f(y, p=P):
    C, R = y
    return np.array([p["s"] - p["k"] * C - p["c"] * R * C,
                     p["r"] * R * C / (p["K"] + C) - p["d"] * R])


def jac(y, p=P):
    C, R = y
    return np.array([[-p["k"] - p["c"] * R, -p["c"] * C],
                     [p["r"] * R * p["K"] / (p["K"] + C) ** 2,
                      p["r"] * C / (p["K"] + C) - p["d"]]])


def explicit_euler(f, y0, T, n):
    tau = T / n
    y = np.empty((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + tau * f(y[i])
    return y


def implicit_euler(f, jac, y0, T, n, tol=1e-12, maxit=50):
    """y_{k+1} = y_k + tau f(y_{k+1}), solved with Newton's method."""
    tau = T / n
    y = np.empty((n + 1, len(y0)))
    y[0] = y0
    total_newton = 0
    for i in range(n):
        z = y[i].copy()
        for it in range(maxit):
            g = z - y[i] - tau * f(z)
            if np.linalg.norm(g) < tol:
                break
            z = z - np.linalg.solve(np.eye(len(z)) - tau * jac(z), g)
            total_newton += 1
        y[i + 1] = z
    return y, total_newton


if __name__ == "__main__":
    y0 = np.array([1.0, 10.0])
    T = 30.0

    # steady state and stiffness (Jacobian eigenvalues at the steady state)
    Cs = P["K"] * P["d"] / (P["r"] - P["d"])
    Rs = (P["s"] / Cs - P["k"]) / P["c"]
    lam_fast, lam_slow = sorted(np.linalg.eigvals(jac(np.array([Cs, Rs]))).real)
    tau_max = 2 / abs(lam_fast)
    print(f"steady state C*={Cs:.3f}, R*={Rs:.1f}; eigenvalues {lam_fast:.1f}, {lam_slow:.3f}; "
          f"stiffness ratio {abs(lam_fast/lam_slow):.0f}; explicit needs tau < {tau_max:.4f} d")

    # reference solution: implicit Radau with tight tolerances
    ref = solve_ivp(lambda t, y: f(y), (0, T), y0, method="Radau", rtol=1e-10, atol=1e-12, dense_output=True)

    def clean(a, lim=1e6):
        a = np.asarray(a, float)
        bad = ~np.isfinite(a) | (np.abs(a) > lim)
        return [None if b else round(float(v), 5) for v, b in zip(a, bad)]

    def run(method, tau):
        n = int(round(T / tau))
        newton = 0
        if method == "explicit":
            with np.errstate(all="ignore"):
                y = explicit_euler(f, y0, T, n)
        else:
            y, newton = implicit_euler(f, jac, y0, T, n)
        t = np.linspace(0, T, n + 1)
        bad = ~np.isfinite(y).all(axis=1) | (np.abs(y).max(axis=1) > 1e6)
        div_day = float(t[np.argmax(bad)]) if bad.any() else None
        good = ~bad
        yr = ref.sol(t[good]).T
        err = float(np.max(np.abs(y[good, 1] - yr[:, 1]))) if good.any() else None
        minC = float(np.min(y[good, 0])) if good.any() else None
        return dict(method=method, tau=tau, n=n, newton=newton, diverged_day=div_day,
                    min_C=minC, max_err_R=err, R_end=None if div_day else float(y[-1, 1]),
                    t=t, y=y)

    table_cfg = [("explicit", 0.01), ("explicit", 0.014), ("explicit", 0.016), ("explicit", 0.02),
                 ("explicit", 0.05), ("implicit", 0.02), ("implicit", 0.1), ("implicit", 0.5),
                 ("implicit", 1.0)]
    runs = {(m, tau): run(m, tau) for m, tau in table_cfg}
    out = dict(params=P, steady=[Cs, Rs], eig=[float(lam_fast), float(lam_slow)], tau_max=float(tau_max),
               stiffness_ratio=float(abs(lam_fast / lam_slow)), T=T, R_end_ref=float(ref.sol(T)[1]))

    # Panel A: cytokine C over the first 20 days, same step tau = 0.02 d (above the explicit limit)
    tf = np.linspace(0, 20, 801)
    ea, ia = runs[("explicit", 0.02)], runs[("implicit", 0.02)]
    m20 = ea["t"] <= 20 + 1e-9
    out["A"] = dict(tau=0.02, tref=tf.tolist(), ref=ref.sol(tf)[0].tolist(),
                    t=ea["t"][m20].tolist(), explicit=clean(ea["y"][m20, 0]),
                    implicit=clean(ia["y"][m20, 0]))

    # Panel B: Treg population R over 30 days
    tt = np.linspace(0, T, 601)
    b = {"tref": tt.tolist(), "ref": ref.sol(tt)[1].tolist()}
    for key, cfg, sub in [("implicit_1", ("implicit", 1.0), 1), ("implicit_05", ("implicit", 0.5), 1),
                          ("explicit_014", ("explicit", 0.014), 20)]:
        r = runs[cfg]
        b[key] = dict(tau=cfg[1], n=r["n"], t=r["t"][::sub].tolist(), R=clean(r["y"][::sub, 1]))
    out["B"] = b

    out["table"] = [{k: v for k, v in r.items() if k not in ("t", "y")} for r in runs.values()]
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/study1.json", "w"))
    for r in out["table"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("R(30) reference:", ref.sol(T)[1])
