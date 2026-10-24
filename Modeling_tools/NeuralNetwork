"""Study 3: A small neural network as a fast stand-in (surrogate) for the Treg simulation.

Question: how many Tregs are there after 30 days, as a function of cytokine supply s and
proliferation rate r?  Every answer normally needs an ODE solve (Study 1 model).  The network
learns the map (s, r) -> R(30 d) from a limited number of simulations.

The feed-forward network is trained with L-BFGS-B on the mean-squared error, with all weights and
biases packed into one parameter vector.
Illustrative model; all numbers are synthetic.
"""
import json
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

from study1_stiff import P

T_END = 30.0
Y0 = [1.0, 10.0]
S_FULL, R_FULL = (40.0, 160.0), (0.6, 1.8)      # region we want predictions for
S_TRAIN, R_TRAIN = (60.0, 140.0), (0.7, 1.5)    # region where simulations were run


def simulate(s, r):
    p = dict(P, s=s, r=r)

    def rhs(t, y):
        C, R = y
        return [p["s"] - p["k"] * C - p["c"] * R * C, p["r"] * R * C / (p["K"] + C) - p["d"] * R]

    sol = solve_ivp(rhs, (0, T_END), Y0, method="Radau", rtol=1e-7, atol=1e-9)
    return sol.y[1, -1]


# ---- feed-forward network -----------------------------------------------------------------
def get_pdim(w):
    return sum(w[i + 1] * (w[i] + 1) for i in range(len(w) - 1))


def params_to_dof(p, w):
    Ws, bs, idx = [], [], 0
    for i in range(len(w) - 1):
        n_in, n_out = w[i], w[i + 1]
        Ws.append(p[idx:idx + n_out * n_in].reshape(n_out, n_in)); idx += n_out * n_in
        bs.append(p[idx:idx + n_out].reshape(n_out, 1)); idx += n_out
    return Ws, bs


def feedforward(X, Ws, bs):
    a = X.T
    for W, b in zip(Ws[:-1], bs[:-1]):
        a = np.tanh(W @ a + b)
    return (Ws[-1] @ a + bs[-1]).T


def scale_in(S, Rr, lo_s, hi_s, lo_r, hi_r):
    return np.column_stack([2 * (S - lo_s) / (hi_s - lo_s) - 1, 2 * (Rr - lo_r) / (hi_r - lo_r) - 1])


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    N = 150
    lhs = qmc.LatinHypercube(d=2, seed=3).random(N)
    s_tr = S_TRAIN[0] + lhs[:, 0] * (S_TRAIN[1] - S_TRAIN[0])
    r_tr = R_TRAIN[0] + lhs[:, 1] * (R_TRAIN[1] - R_TRAIN[0])
    t0 = time.perf_counter()
    y_tr = np.array([simulate(s, r) for s, r in zip(s_tr, r_tr)])
    t_sim = (time.perf_counter() - t0) / N
    print(f"{N} training simulations, {t_sim*1e3:.1f} ms each; R30 range {y_tr.min():.1f}..{y_tr.max():.1f}")

    Y_SCALE = 200.0
    X_train = scale_in(s_tr, r_tr, *S_FULL, *R_FULL)  # scale to the FULL region so maps are comparable
    Y_train = (y_tr / Y_SCALE).reshape(-1, 1)

    widths = [2, 20, 20, 1]
    best = None
    for seed in range(4):
        p0 = np.random.default_rng(seed).standard_normal(get_pdim(widths)) * 0.5

        def loss(p):
            Ws, bs = params_to_dof(p, widths)
            return np.mean((feedforward(X_train, Ws, bs) - Y_train) ** 2)

        res = minimize(loss, p0, method="L-BFGS-B", options=dict(maxiter=1500, maxfun=10**6))
        print("restart", seed, "train MSE", res.fun, "iterations", res.nit)
        if best is None or res.fun < best.fun:
            best = res
    Ws, bs = params_to_dof(best.x, widths)

    def predict(s, r):
        return feedforward(scale_in(np.atleast_1d(s), np.atleast_1d(r), *S_FULL, *R_FULL), Ws, bs).ravel() * Y_SCALE

    # ---- held-out test inside the training region -------------------------------------------
    m = 500
    s_te = rng.uniform(*S_TRAIN, m); r_te = rng.uniform(*R_TRAIN, m)
    y_te = np.array([simulate(s, r) for s, r in zip(s_te, r_te)])
    err_in = predict(s_te, r_te) - y_te
    print("inside training region: RMSE", np.sqrt(np.mean(err_in ** 2)), "max", np.abs(err_in).max())

    # ---- map over the full region -------------------------------------------------------------
    G = 41
    sv = np.linspace(*S_FULL, G); rv = np.linspace(*R_FULL, G)
    SS, RR = np.meshgrid(sv, rv)        # rows = r, columns = s
    truth = np.array([simulate(s, r) for s, r in zip(SS.ravel(), RR.ravel())]).reshape(G, G)
    pred = predict(SS.ravel(), RR.ravel()).reshape(G, G)
    err = np.abs(pred - truth)
    inside = (SS >= S_TRAIN[0]) & (SS <= S_TRAIN[1]) & (RR >= R_TRAIN[0]) & (RR <= R_TRAIN[1])
    print("map RMSE inside", np.sqrt(np.mean((pred - truth)[inside] ** 2)),
          " outside", np.sqrt(np.mean((pred - truth)[~inside] ** 2)),
          " max outside", err[~inside].max())

    # timing of the surrogate
    Xq = scale_in(SS.ravel(), RR.ravel(), *S_FULL, *R_FULL)
    t0 = time.perf_counter()
    for _ in range(200):
        feedforward(Xq, Ws, bs)
    t_nn = (time.perf_counter() - t0) / 200 / Xq.shape[0]
    print(f"NN: {t_nn*1e6:.2f} us per point vs simulation {t_sim*1e3:.1f} ms  -> x{t_sim/t_nn:.0f}")

    out = dict(S_FULL=S_FULL, R_FULL=R_FULL, S_TRAIN=S_TRAIN, R_TRAIN=R_TRAIN, G=G, N_train=N,
               sv=sv.tolist(), rv=rv.tolist(), truth=np.round(truth, 3).tolist(),
               pred=np.round(pred, 3).tolist(), err=np.round(err, 3).tolist(),
               train_pts=np.column_stack([s_tr, r_tr, y_tr]).round(3).tolist(),
               widths=widths, n_params=int(get_pdim(widths)), train_mse=float(best.fun),
               rmse_in=float(np.sqrt(np.mean(err_in ** 2))), maxerr_in=float(np.abs(err_in).max()),
               rmse_map_in=float(np.sqrt(np.mean((pred - truth)[inside] ** 2))),
               rmse_map_out=float(np.sqrt(np.mean((pred - truth)[~inside] ** 2))),
               maxerr_out=float(err[~inside].max()), t_sim_ms=t_sim * 1e3, t_nn_us=t_nn * 1e6,
               speedup=float(t_sim / t_nn), Y_SCALE=Y_SCALE, T=T_END)
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/study3.json", "w"))
