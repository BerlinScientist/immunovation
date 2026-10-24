"""Study 4: Weekly capacity planning for an assay core with linear optimization.

Illustrative numbers (not from a real lab). Decision: how many batches of each assay per week
to maximise the number of samples analysed, given limited staff, instruments and budget.

Solver: if simplex.py (primal simplex, tableau form) is present next to this file it is used and
checked against SciPy. Without it the script falls back to SciPy's linprog and gives the same answer.
"""
import json
import os
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

try:
    from simplex import primal_simplex
    HAVE_SIMPLEX = True
except ImportError:
    HAVE_SIMPLEX = False

assays = ["Flow cytometry run", "Cytokine ELISA plate", "qPCR plate", "RNA-seq library batch"]
samples = np.array([24, 40, 32, 24], float)                 # samples analysed per batch
resources = ["Technician hours", "Cytometer hours", "Plate-reader hours",
             "Thermocycler hours", "Reagent budget (EUR)", "Sequencing slots"]
units = ["h", "h", "h", "h", "EUR", "slots"]
A = np.array([[6, 4, 3, 8],
              [4, 0, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 1.5, 0],
              [900, 350, 250, 2200],
              [0, 0, 0, 1]], float)
b = np.array([60, 20, 10, 12, 5000, 2], float)
m, n = A.shape


def solve_lp(b_vec):
    """Maximise samples @ x subject to A x <= b_vec, x >= 0.
    Returns x, shadow prices (per resource), reduced costs (per assay)."""
    if HAVE_SIMPLEX:
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:m, :n] = A
        tableau[:m, n:n + m] = np.eye(m)
        tableau[:m, -1] = b_vec
        tableau[-1, :n] = -samples
        sol, final, _ = primal_simplex(tableau, return_details=True)
        # z-row of the final tableau: slack columns = shadow prices, assay columns = reduced costs
        return sol[:n], final[-1, n:n + m], final[-1, :n]
    res = linprog(-samples, A_ub=A, b_ub=b_vec, method="highs")
    return res.x, -res.ineqlin.marginals, res.lower.marginals


if __name__ == "__main__":
    x, shadow, reduced = solve_lp(b)
    value = float(samples @ x)
    used = A @ x
    slack = b - used

    # check against SciPy
    ref = linprog(-samples, A_ub=A, b_ub=b, method="highs")
    assert np.allclose(x, ref.x, atol=1e-8) and abs(value + ref.fun) < 1e-8, "solver disagrees with SciPy"
    assert np.allclose(shadow, -ref.ineqlin.marginals, atol=1e-8)
    print("solver:", "primal simplex (simplex.py)" if HAVE_SIMPLEX else "SciPy linprog (simplex.py not found)")

    # integer solution (whole batches only)
    ilp = milp(-samples, constraints=LinearConstraint(A, ub=b), integrality=np.ones(n), bounds=Bounds(0, np.inf))
    x_int = np.round(ilp.x).astype(int)

    print("LP optimum x =", np.round(x, 3), "samples/week =", round(value, 2))
    print("integer optimum x =", x_int, "samples/week =", float(samples @ x_int))
    for i in range(m):
        print(f"{resources[i]:22s} used {used[i]:8.2f} / {b[i]:7.1f}  slack {slack[i]:7.2f}  shadow price {shadow[i]:.4f}")
    print("reduced costs (samples):", np.round(reduced, 3))

    # sensitivity: value of the plan when one resource grows by 50%
    sens = []
    for i in range(m):
        b2 = b.copy(); b2[i] *= 1.5
        sens.append(float(samples @ solve_lp(b2)[0]) - value)
    print("gain from +50% of each resource:", np.round(sens, 1))

    os.makedirs("data", exist_ok=True)
    json.dump(dict(assays=assays, samples=samples.tolist(), resources=resources, units=units,
                   A=A.tolist(), b=b.tolist(), x=x.tolist(), value=value, x_int=x_int.tolist(),
                   value_int=float(samples @ x_int), used=used.tolist(), slack=slack.tolist(),
                   shadow=shadow.tolist(), reduced=reduced.tolist(), gain_plus50=sens,
                   verified_against_scipy=True), open("data/study4.json", "w"))
