"""Study 2: Interpolating a dose-response curve from few measurements.

Illustrative, noise-free synthetic curve (Hill/logistic in log10 dose):
    y(x) = 1 / (1 + 10**(h*(logEC50 - x))),  x = log10(dose), y = fraction of maximal response.
Compares polynomial interpolation on equidistant nodes (Lagrange form),
Chebyshev nodes, and a monotone piecewise cubic (PCHIP).
"""
import json
import os
import numpy as np
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator

H, LOGEC50 = 2.0, 0.0
A, B = -3.0, 3.0


def truth(x):
    return 1.0 / (1.0 + 10.0 ** (H * (LOGEC50 - x)))


class LagrangeInterpolation:
    """Lagrange form of the interpolation polynomial."""

    def __init__(self, nodes, function_values):
        if len(nodes) != len(set(nodes)):
            raise ValueError("Nodes must be pairwise distinct.")
        self.nodes = np.asarray(nodes, float)
        self.values = np.asarray(function_values, float)

    def _basis(self, x, k):
        r = np.ones_like(x, dtype=float)
        for i, xi in enumerate(self.nodes):
            if i != k:
                r *= (x - xi) / (self.nodes[k] - xi)
        return r

    def __call__(self, x):
        x = np.asarray(x, float)
        return sum(self.values[k] * self._basis(x, k) for k in range(len(self.nodes)))


def equidistant(m):
    return np.linspace(A, B, m)


def chebyshev(m):
    k = np.arange(m)
    return 0.5 * (A + B) + 0.5 * (B - A) * np.cos((2 * k + 1) * np.pi / (2 * m))[::-1]


xf = np.linspace(A, B, 2001)
yf = truth(xf)


def evaluate(kind, m):
    if kind == "poly_equi":
        n = equidistant(m)
        return LagrangeInterpolation(n, truth(n))(xf), n
    if kind == "poly_cheb":
        n = chebyshev(m)
        return BarycentricInterpolator(n, truth(n))(xf), n
    n = equidistant(m)
    return PchipInterpolator(n, truth(n))(xf), n


if __name__ == "__main__":
    out = dict(h=H, logEC50=LOGEC50, A=A, B=B)
    M_SHOW = 13
    out["curves"] = {"x": xf[::5].tolist(), "truth": yf[::5].tolist()}
    out["m_show"] = M_SHOW
    for kind in ("poly_equi", "poly_cheb", "pchip"):
        y, n = evaluate(kind, M_SHOW)
        out["curves"][kind] = dict(y=np.round(y[::5], 5).tolist(), nodes=n.tolist(),
                                   vals=truth(n).tolist(), ymin=float(y.min()), ymax=float(y.max()),
                                   maxerr=float(np.max(np.abs(y - yf))))
        print(kind, "m =", M_SHOW, "range", round(y.min(), 3), round(y.max(), 3), "max err", round(np.max(np.abs(y - yf)), 4))

    ms = list(range(3, 41))
    err = {k: [] for k in ("poly_equi", "poly_cheb", "pchip")}
    for m in ms:
        for kind in err:
            y, _ = evaluate(kind, m)
            err[kind].append(float(np.max(np.abs(y - yf))))
    out["err"] = dict(m=ms, **err)
    # smallest error per method
    for kind in err:
        i = int(np.argmin(err[kind]))
        print(kind, "best m =", ms[i], "err", err[kind][i], " err at m=40:", err[kind][-1])
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/study2.json", "w"))
