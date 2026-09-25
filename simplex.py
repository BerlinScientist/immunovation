"""Primal simplex method in tableau form for problems of the type

    max c^T x   subject to   A x <= b,  x >= 0,  b >= 0.

Tableau layout, with slack variables in the last m columns:

        [ A   b ]
        [ -c  v ]     (v = objective value)

Column indices are 0-based. `basis` lists the column index of the basic variable of each row.
`primal_simplex(..., return_details=True)` also returns the final tableau, from which shadow
prices (slack columns of the last row) and reduced costs (variable columns) can be read.
"""
import numpy as np


def is_canonical(tableau, basis, tol=1e-9):
    """True if the basis columns form an identity matrix with zero entries in the objective row."""
    m = tableau.shape[0] - 1
    if len(basis) != m or len(set(basis)) != m:
        return False
    for row, col in enumerate(basis):
        expected = np.zeros(m + 1)
        expected[row] = 1.0
        if not np.allclose(tableau[:, col], expected, atol=tol):
            return False
    return True


def pivot(tableau, basis, pivot_idx, verbose=False):
    """One basis change with pivot element pivot_idx = (row, column). Returns (new_tableau, new_basis)."""
    new_tableau = tableau.astype(float).copy()
    new_basis = list(basis)
    i, j = pivot_idx

    # input checks
    assert is_canonical(new_tableau, new_basis), "Tableau is not in canonical form."
    pivot_value = new_tableau[i, j]
    assert abs(pivot_value) > 1e-12, "Pivot element is zero."

    # normalise the pivot row, then eliminate the pivot column from all other rows
    new_tableau[i, :] = new_tableau[i, :] / pivot_value
    for r in range(new_tableau.shape[0]):
        if r != i:
            new_tableau[r, :] = new_tableau[r, :] - new_tableau[r, j] * new_tableau[i, :]
    new_basis[i] = j

    if verbose:
        print(f"Pivot = {pivot_value} at (row, column) = {pivot_idx}")
        print(new_tableau)
    return new_tableau, new_basis


def primal_simplex(tableau, verbose=False, return_details=False):
    """Primal simplex method (largest-coefficient rule, minimum-ratio test).

    Returns the optimal basic solution (array of length n, slack variables included) or None if the
    problem is unbounded. With return_details=True it returns (solution, final_tableau, basis).
    """
    new_tableau = np.array(tableau, dtype=float)
    m = new_tableau.shape[0] - 1
    n = new_tableau.shape[1] - 1

    # start basis: the last m variables (slack variables)
    basis = list(range(n - m, n))

    # primal feasible: b >= 0, and canonical form
    assert np.all(new_tableau[:m, -1] >= 0), "Tableau is not primal feasible (b >= 0 required)."
    assert is_canonical(new_tableau, basis), \
        "Start tableau is not canonical (slack variables expected in the last m columns)."

    while np.any(new_tableau[-1, :-1] < -1e-12):
        # entering column: most negative entry of the objective row
        j = int(np.argmin(new_tableau[-1, :-1]))

        # leaving row: minimum-ratio test
        ratios = np.full(m, np.inf)
        for i in range(m):
            if new_tableau[i, j] > 1e-12:
                ratios[i] = new_tableau[i, -1] / new_tableau[i, j]

        # all ratios infinite: unbounded
        if np.all(np.isinf(ratios)):
            return (None, new_tableau, basis) if return_details else None

        i = int(np.argmin(ratios))
        new_tableau, basis = pivot(new_tableau, basis, (i, j), verbose)

    # read the optimal basic solution
    solution = np.zeros(n)
    for row, variable in enumerate(basis):
        solution[variable] = new_tableau[row, -1]
    if return_details:
        return solution, new_tableau, basis
    return solution
