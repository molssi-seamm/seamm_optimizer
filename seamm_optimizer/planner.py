# -*- coding: utf-8 -*-

"""Choosing how to optimize by the cost to the user (Paul 11-24: the figure of merit
is the total time to solution, all levels included).

Three strategies, all ending at the expensive level's minimum:

- "lindh": quasi-Newton on the expensive level with Lindh's model Hessian;
- "cheap-hessian": the same, starting from (and refreshing with) the cheap level's
  Hessian;
- "surrogate": the cheap level carries the steps (SurrogateOptimizer).

The expected cost of each, in expensive gradients, comes from counts measured on the
GFN2-xTB proxy (125 molecules, 3-62 atoms, Design 2026-11-24), as functions of the
atom count N and the ratio r = c_cheap / c_expensive:

    lindh          4.6 + 0.52 N
    cheap-hessian  (-0.9 + 0.70 N) + (0.9 + 0.025 N) * h
                   (h = the cheap Hessian's cost / c_expensive)
    surrogate      (0.6 + 0.51 N) + (17.5 + 2.1 N) * r

On
that data the surrogate was cheapest up to r ~ 0.05, Lindh from r ~ 0.1-0.2, and a
cheap Hessian by finite differences (6N cheap gradients) paid only when nearly free.
Scored leaving each set out in turn, choosing between Lindh and the surrogate by
these models came within 0-13 % of the best choice per molecule, and adding the
number of rotatable bonds did not help reliably (Design 2026-11-24).
"""

from dataclasses import dataclass
import time

from .optimizer import Optimizer
from .surrogate import SurrogateOptimizer

#: (a, b) of the counts above, a + b N
MODEL = {
    "lindh": (4.6, 0.52),
    "cheap-hessian": (-0.9, 0.70),
    "cheap-hessian hessians": (0.9, 0.025),
    "surrogate": (0.6, 0.51),
    "surrogate cheap": (17.5, 2.1),
}


@dataclass
class Plan:
    strategy: str
    expected: dict  # strategy -> expected cost in expensive gradients
    costs: dict


def expected_costs(n_atoms, costs, cheap=True, cheap_hessian=True, cheap_budget=1.0):
    """The expected cost of each available strategy, in expensive gradients."""

    def lin(key):
        a, b = MODEL[key]
        return max(1.0, a + b * n_atoms)

    out = {"lindh": lin("lindh")}
    c_h = costs["expensive"]
    if cheap and costs.get("cheap"):
        r = costs["cheap"] / c_h
        # capping the micro-iterations does not lower the cost (fewer micro-steps,
        # more outer ones: fragments at r = 0.3, 999 against 808 uncapped), so the
        # uncapped cheap work is expected; at high r Lindh is chosen instead
        out["surrogate"] = lin("surrogate") + lin("surrogate cheap") * r
    # a cheap Hessian by finite differences (6N cheap gradients) almost never paid
    # when a cheap gradient was there for a surrogate, and choosing it on small
    # predicted differences cost more than it saved (held-out: r = 0.001, 1301
    # against 1090 for the surrogate alone). So with a cheap gradient the choice is
    # Lindh or the surrogate; the cheap-Hessian path is a candidate only without one,
    # or when the cheap Hessian's own cost is given (an analytic Hessian).
    if cheap_hessian and (
        not (cheap and costs.get("cheap")) or "cheap_hessian" in costs
    ):
        h_cost = costs.get("cheap_hessian")
        if h_cost is None and costs.get("cheap"):
            h_cost = (
                6 * n_atoms * costs["cheap"]
            )  # finite differences of cheap gradients
        if h_cost is not None:
            out["cheap-hessian"] = (
                lin("cheap-hessian") + lin("cheap-hessian hessians") * h_cost / c_h
            )
    return out


def measure(engine, xyz):
    """The wall time of one call."""
    t = time.perf_counter()
    engine(xyz)
    return time.perf_counter() - t


def plan(n_atoms, costs, cheap=True, cheap_hessian=True, cheap_budget=1.0):
    expected = expected_costs(n_atoms, costs, cheap, cheap_hessian, cheap_budget)
    strategy = min(expected, key=expected.get)
    return Plan(strategy, expected, dict(costs))


def optimize(
    engine,
    symbols,
    xyz,
    cheap=None,
    cheap_hessian=None,
    costs=None,
    fmax=0.0015,
    cheap_budget=1.0,
    **kwargs,
):
    """Optimize at the expensive level by the strategy with the lowest expected total
    cost. `costs` = {"expensive": c_H, "cheap": c_L, "cheap_hessian": c_LH} in any
    units; missing ones are measured from one call each (the measuring calls are not
    wasted: the expensive one is at the start). Returns the result, with .plan."""
    costs = dict(costs or {})
    if "expensive" not in costs:
        costs["expensive"] = measure(engine, xyz)
    if cheap is not None and "cheap" not in costs:
        costs["cheap"] = measure(cheap, xyz)
    p = plan(
        len(symbols), costs, cheap is not None, cheap_hessian is not None, cheap_budget
    )
    # a cheap Hessian is used for the minimum check (and fallback) only if affordable
    affordable = (
        cheap_hessian is not None
        and (costs.get("cheap_hessian") or 6 * len(symbols) * costs.get("cheap", 0.0))
        <= 5.0 * costs["expensive"]
    )
    hessian = cheap_hessian if affordable else None
    if p.strategy == "surrogate":
        r = SurrogateOptimizer(
            engine,
            cheap,
            symbols,
            xyz,
            fmax=fmax,
            cheap_hessian_engine=hessian,
            **kwargs,
        ).run()
    elif p.strategy == "cheap-hessian":
        r = Optimizer(
            engine,
            symbols,
            xyz,
            fmax=fmax,
            hessian="cheap",
            cheap_hessian_engine=cheap_hessian,
            **kwargs,
        ).run()
    else:
        r = Optimizer(
            engine,
            symbols,
            xyz,
            fmax=fmax,
            hessian="lindh",
            cheap_hessian_engine=hessian,
            refresh="never",
            **kwargs,
        ).run()
    r.plan = p
    return r
