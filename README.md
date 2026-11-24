# seamm_optimizer

Robust, economical geometry optimization for SEAMM: quasi-Newton minimization with an
RFO / trust-region step, Lindh's model Hessian or a computed (cheap-level) one, BFGS
updates, and a check that the end point is a minimum, with an escape from saddle points.

Design: `OPTIMIZER_DESIGN_PROPOSAL_20261124.md` (Paul's §10 decisions, 2026-11-24).
This is phase 1 (Cartesian coordinates). Next: internal coordinates (geomeTRIC's
primitives with a checked back-transformation), cheap-level Hessians and the
Δ-gradient / Δ-Hessian surrogates.

```python
from seamm_optimizer import Optimizer

result = Optimizer(engine, symbols, xyz, hessian_engine=hessian).run()
# engine(xyz) -> (energy eV, gradient N x 3 eV/Å); hessian(xyz) -> 3N x 3N eV/Å^2
```

`benchmark/mmff_proxy.py` compares it with ASE's BFGS on MMFF94 surfaces from the
benchmark's starting structures, at no cost.
