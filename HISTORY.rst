=======
History
=======
2026.11.24.1 -- Bugfix: Lindh's model Hessian for near-linear arrangements
    * Torsions through near-linear angles are left out of Lindh's model Hessian, as
      geomeTRIC leaves out such dihedrals. The model's torsions include non-bonded
      pairs, and where four atoms line up (e.g. across cumene's ring) their terms made
      the model hundreds of times too stiff, so the optimization crawled.
    * A near-linear bend gets curvature in both bending directions (a linear-bend
      pair), not one; linear molecules such as HC#CI took many times more steps.
    * The trust radius follows geomeTRIC's rules: a step is rejected only when much
      worse than predicted, and poor steps halve the radius.
    * On a GFN2-xTB proxy of the benchmark (101 structures), with GFN1-xTB as the cheap
      level: median 4 gradients, against 6-9 for geomeTRIC and 15-22 for ASE's BFGS.

2026.11.24 -- First release: phases 1-3
    * Phase 3: cheap-level Hessians (at the start, refreshed when the model keeps
      failing, and for the minimum check), with their eigenvalues clamped for
      minimization.

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
