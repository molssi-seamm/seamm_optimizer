=======
History
=======
(unreleased) -- Phase 2: internal coordinates
    * Steps, Hessian and BFGS updates in geomeTRIC's internal coordinates (TRIC by
      default; DLC, HDLC, primitives, or Cartesians), with every back-transformation
      checked and a Cartesian fallback.
    * On MMFF94 proxies of the benchmark: perturbed starts 20.0 gradients mean (geomeTRIC
      21.4, ASE BFGS 82.4); PM7 CHNO starts 22.3 (geomeTRIC 17.4, ASE 121.7).

(unreleased) -- Phase 1: Cartesian quasi-Newton optimizer
    * RFO / trust-region steps, Lindh's model Hessian, BFGS updates, the shared
      0.0015 eV/Å convergence criterion, and a minimum check with a two-way escape
      from saddle points.
