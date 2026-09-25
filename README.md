# ImmunoVation modeling notes

Four small case studies that show how numerical methods behave on immunology-style models:
when a simulation breaks, when a curve fit cannot be trusted, where a fast surrogate stops
being reliable, and how to plan limited lab capacity.

**All models are illustrative.** Parameters were chosen to demonstrate the methods. They are not
fitted to patient, client or published data.

| Script | Question | Methods |
|---|---|---|
| `study1_stiff.py` | Can I trust my time step? (cytokine/Treg model with fast and slow time scales) | explicit and implicit Euler (Newton), stiff reference solver (Radau) |
| `study2_doseresponse.py` | Why can more dose levels make a fit worse? | Lagrange polynomials, Chebyshev nodes, PCHIP |
| `study3_surrogate.py` | Where does a neural-network stand-in for the simulation stop being reliable? | feed-forward network trained with L-BFGS-B, Latin hypercube sampling |
| `study4_lab_planning.py` | Which assay mix maximizes samples per week, and where is the bottleneck? | primal simplex, shadow prices, integer check |

## Run

```bash
pip install -r requirements.txt
python study1_stiff.py
python study2_doseresponse.py
python study3_surrogate.py      # about 2 minutes: trains the network with four restarts
python study4_lab_planning.py
```

Each script prints its key numbers and writes a JSON file to `data/` (created automatically).
Random seeds are fixed. Small differences across platforms and library versions are possible.

`study3_surrogate.py` imports the model from `study1_stiff.py`, so keep the files in one folder.

## Solver note

`study4_lab_planning.py` uses `simplex.py` (a tableau-based primal simplex method that also
returns the final tableau for shadow prices) if it is present, and checks the result against SciPy.
If `simplex.py` is missing, the script falls back to SciPy's `linprog` and gives the same answer.

The solver, interpolation, network and simplex code are implementations of standard algorithms
(Euler schemes, Lagrange interpolation, feed-forward networks, tableau simplex) applied to new
biological examples.

## License

MIT, see `LICENSE`.

Andrea Loewendorf, PhD, ImmunoVation, LLC
