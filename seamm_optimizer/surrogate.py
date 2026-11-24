# -*- coding: utf-8 -*-

"""Phase 4: a cheap level carries the expensive one (design proposal §6).

At each expensive point x_k both levels are evaluated, and the surrogate

    S(x) = E_L(x) + ΔE_k + Δg_k·(x − x_k) + ½ (x − x_k)ᵀ ΔB (x − x_k),   Δ = H − L,

which agrees with the expensive level H at x_k, is minimized on the cheap level L in
micro-iterations (our own optimizer, within a trust radius). The expensive level is
evaluated at the result, and the step accepted or rejected on the ratio of its actual
to the predicted change. ΔB, the Hessian of the difference, is learned from the
differences of Δg between expensive points with symmetric-rank-one updates (a
difference of Hessians need not be positive definite). ΔB = 0 is MOPAC's
Δ-gradient method (variant "delta-gradient"); without a cheap level it would be
quasi-Newton on H.

Every evaluation on both levels is counted, so that the cost -- the user's time to
solution -- can be judged at any ratio of the levels' costs (Paul 11-24).
"""

from dataclasses import dataclass, field
import logging

import numpy as np

from .geometry import rigid_modes
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class SurrogateResult:
    xyz: np.ndarray
    energy: float
    gradient: np.ndarray
    converged: bool
    n_gradients: int = 0  # expensive
    n_cheap_gradients: int = 0
    n_steps: int = 0
    n_rejected: int = 0
    minimum: bool | None = None
    lowest_frequency: float | None = None
    confirmations: list = field(default_factory=list)
    escapes: list = field(default_factory=list)
    fallback: str | None = None
    history: list = field(default_factory=list)

    @property
    def max_force(self):
        return float(np.max(np.linalg.norm(self.gradient, axis=1)))

    def cost(self, ratio, cheap_hessians=0, hessian_cost=None):
        """The cost in expensive gradients, with a cheap gradient costing `ratio`."""
        return self.n_gradients + ratio * self.n_cheap_gradients


def _aligned(x, reference):
    """x moved rigidly (Kabsch) onto reference."""
    cx, cr = x.mean(axis=0), reference.mean(axis=0)
    a, b = x - cx, reference - cr
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1.0, 1.0, d]) @ vt
    return a @ R + cr


class SurrogateOptimizer:
    """Minimize the expensive `engine` with the cheap `surrogate` carrying the steps.

    Parameters
    ----------
    engine, surrogate : callable
        ``f(xyz) -> (energy eV, gradient eV/Å (N, 3))`` at the expensive and the cheap
        level.
    symbols, xyz : list, array
        The atoms and the start (Å).
    fmax : float
        Convergence: the largest force on an atom at the expensive level (eV/Å).
    variant : str
        "delta-hessian" (ΔB learned) or "delta-gradient" (ΔB = 0, as MOPAC).
    trust, max_trust, min_trust : float
        The trust radius: the largest atomic displacement of a step (Å).
    inner_fmax_factor, inner_relative, inner_max_steps : float, float, int
        The micro-iterations converge S to the larger of inner_fmax_factor × fmax and
        inner_relative × the current largest expensive force, in at most
        inner_max_steps cheap gradients.
    hessian_engine, cheap_hessian_engine : callable, optional
        For checking the end point (the cheap level's modes confirmed at the
        expensive level, as Optimizer does); no check without either. The cheap
        Hessian also serves the quasi-Newton fallback.
    costs : dict, optional
        The cost of one call at each level, {"expensive": c_H, "cheap": c_L} (any
        units, e.g. seconds): the micro-iterations per step are capped at
        cheap_budget * c_H / c_L. Without it, inner_max_steps.
    cheap_budget : float
        The cheap work allowed per step, in expensive gradients.
    wander : float
        The micro-iterations stop beyond this many trust radii.
    fallback_after : int
        After this many poor surrogate steps in a row (ratio < 0.25, or no step
        found), the rest is quasi-Newton on the expensive level alone; also when the
        cheap level fails at an expensive point.
    escape_displacement, escape_rounds : float, int
        As for Optimizer: the escape from a confirmed saddle point.
    """

    def __init__(
        self,
        engine,
        surrogate,
        symbols,
        xyz,
        fmax=0.0015,
        variant="delta-hessian",
        max_steps=None,
        trust=0.2,
        max_trust=0.5,
        min_trust=0.002,
        inner_fmax_factor=0.3,
        inner_relative=0.1,
        inner_max_steps=50,
        costs=None,
        cheap_budget=1.0,
        hessian_engine=None,
        cheap_hessian_engine=None,
        check_minimum=True,
        wander=2.0,
        fallback_after=3,
        escape_displacement=0.3,
        escape_rounds=2,
    ):
        if variant not in ("delta-hessian", "delta-gradient"):
            raise ValueError(f"Unknown surrogate variant {variant!r}")
        self.engine = engine
        self.surrogate = surrogate
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        self.xyz0 = np.asarray(xyz, float).reshape(self.n, 3)
        self.fmax = fmax
        self.variant = variant
        self.max_steps = max_steps if max_steps is not None else 18 * self.n
        self.trust0, self.max_trust, self.min_trust = trust, max_trust, min_trust
        self.inner_fmax = inner_fmax_factor * fmax
        self.inner_relative = inner_relative
        self.inner_max_steps = inner_max_steps
        # the cost of a call at each level (any units), for capping the cheap work
        # per step at cheap_budget expensive gradients' worth (Paul 11-24: the figure
        # of merit is the total time to solution)
        self.costs = dict(costs) if costs else None
        self.cheap_budget = cheap_budget
        self.hessian_engine = hessian_engine
        self.cheap_hessian_engine = cheap_hessian_engine
        self.check_minimum = check_minimum
        self.wander = wander
        self.fallback_after = fallback_after
        self.escape_displacement = escape_displacement
        self.escape_rounds = escape_rounds
        self.n_gradients = 0
        self.n_cheap_gradients = 0

    # ------------------------------------------------------------------ levels
    def _expensive(self, xyz):
        e, g = self.engine(xyz)
        self.n_gradients += 1
        return float(e), np.asarray(g, float).reshape(self.n, 3)

    def _cheap(self, xyz):
        e, g = self.surrogate(xyz)
        self.n_cheap_gradients += 1
        return float(e), np.asarray(g, float).reshape(self.n, 3)

    def _converged(self, g):
        return float(np.max(np.linalg.norm(g, axis=1))) <= self.fmax

    def _micro_cap(self):
        """The most micro-iterations per step: inner_max_steps, or fewer when the
        cheap level is not cheap, so a step's cheap work costs at most cheap_budget
        expensive gradients (at least 2)."""
        if not self.costs or not self.costs.get("cheap"):
            return self.inner_max_steps
        ratio = self.costs["cheap"] / self.costs["expensive"]
        return int(max(2, min(self.inner_max_steps, self.cheap_budget / ratio)))

    # ------------------------------------------------------------------ descent
    def _plain(self, x, why, history):
        """The rest of the way by quasi-Newton on the expensive level alone, the
        "no surrogate" candidate (proposal §6), with the cheap Hessian if there is
        one."""
        history.append({"fallback": why})
        kw = (
            {"hessian": "cheap", "cheap_hessian_engine": self.cheap_hessian_engine}
            if self.cheap_hessian_engine is not None
            else {"hessian": "lindh"}
        )
        r = Optimizer(
            self._expensive,
            self.symbols,
            x,
            fmax=self.fmax,
            max_steps=self.max_steps,
            check_minimum=False,
            **kw,
        ).run()
        return r.xyz, r.energy, r.gradient, r.n_steps, r.n_rejected

    def _descend(self, x):
        """Minimize from x; (xyz, energy, gradient, steps, rejected, history,
        fallback reason or None)."""
        x = np.asarray(x, float).reshape(self.n, 3).copy()
        history = []
        eH, gH = self._expensive(x)
        try:
            eL, gL = self._cheap(x)
        except Exception as e:  # noqa: BLE001 - any failure of the cheap level
            why = f"the cheap level failed at an expensive point: {e}"
            return (*self._plain(x, why, history), history, why)
        dB = np.zeros((3 * self.n, 3 * self.n))
        trust = self.trust0
        history.append({"energy": eH, "max_force": _max_force(gH), "trust": trust})
        n_steps = n_rejected = poor = 0
        while not self._converged(gH) and n_steps < self.max_steps:
            if poor >= self.fallback_after:
                why = f"{poor} poor surrogate steps in a row"
                x, eH, gH, k, j = self._plain(x, why, history)
                return x, eH, gH, n_steps + k, n_rejected + j, history, why
            n_steps += 1
            dE, dg = eH - eL, (gH - gL).ravel()
            x0 = x.copy()
            bound = self.wander * trust
            visited = []  # (S, point aligned onto x0) for every micro-iteration
            attempted = []  # points the micro-iterations tried beyond the bound

            def S(
                y,
                x0=x0,
                dE=dE,
                dg=dg,
                dB=dB,
                visited=visited,
                attempted=attempted,
                bound=bound,
            ):
                y = np.asarray(y, float).reshape(-1, 3)
                ya = _aligned(y, x0)
                if float(np.max(np.linalg.norm(ya - x0, axis=1))) > bound:
                    attempted.append(ya)
                    # the micro-iterations stay near the trust region: far beyond
                    # it the cheap level's SCF failed (GFN1, 7 runs, 2026-11-24)
                    raise _Leave()
                try:
                    el, gl = self._cheap(y)
                except _Leave:
                    raise
                except Exception as e:  # noqa: BLE001 - the cheap level failed here
                    raise _Leave() from e
                d = (y - x0).ravel()
                Bd = dB @ d
                value = el + dE + float(dg @ d) + 0.5 * float(d @ Bd)
                visited.append((value, ya))
                return value, gl + (dg + Bd).reshape(-1, 3)

            # converge S only as far as the expensive gradient warrants (an inexact
            # Newton step): tighter than the shared criterion only near the end
            converged_inner = False
            try:
                inner = Optimizer(
                    S,
                    self.symbols,
                    x0,
                    fmax=max(self.inner_fmax, self.inner_relative * _max_force(gH)),
                    max_steps=self._micro_cap(),
                    check_minimum=False,
                    # its own trust radius (internal coordinates) no larger than
                    # ours: its first step otherwise left the region at once and
                    # the surrogate never moved (n-decane, Design 2026-11-24)
                    trust=trust,
                    max_trust=self.wander * trust,
                ).run()
                converged_inner = inner.converged
            except _Leave:
                pass
            micro = len(visited)
            # the step: the lowest point of S on the micro-iterations' path inside
            # the trust radius. Scaling an end point back onto the trust sphere could
            # land uphill on S, and such a step was accepted with the expensive
            # energy far higher (Design 2026-11-24).
            inside = [
                (value, y)
                for value, y in visited
                if float(np.max(np.linalg.norm(y - x0, axis=1))) <= trust * (1 + 1e-9)
            ]
            value, y = min(inside, key=lambda pair: pair[0]) if inside else (eH, x0)
            if value - eH >= -1.0e-9 and (len(visited) > 1 or attempted):
                # the path left the trust radius at once: backtrack along the line
                # toward the lowest point it found (or the first point it tried
                # beyond the bound), one cheap evaluation per trial
                if len(visited) > 1:
                    best = min(visited, key=lambda pair: pair[0])[1]
                else:
                    best = attempted[0]
                d = best - x0
                longest = float(np.max(np.linalg.norm(d, axis=1)))
                alpha = min(1.0, trust / longest) if longest > 0.0 else 0.0
                for _ in range(6):
                    if alpha <= 0.0:
                        break
                    try:
                        v, _ = S(x0 + alpha * d)
                    except _Leave:
                        v = np.inf
                    if v - eH < -1.0e-9:
                        value, y = v, x0 + alpha * d
                        break
                    alpha *= 0.5
            predicted = value - eH
            kind = (
                "surrogate"
                if converged_inner and len(inside) == len(visited)
                else "restricted"
            )
            if predicted >= -1.0e-9:
                # nothing inside the trust radius improves S: shrink it, without
                # spending an expensive gradient
                poor += 1
                trust = max(self.min_trust, 0.5 * trust)
                history.append({"no step": True, "trust": trust, "micro": micro})
                continue
            eH_new, gH_new = self._expensive(y)
            try:
                eL_new, gL_new = self._cheap(y)
            except Exception as e:  # noqa: BLE001 - the cheap level failed here
                if eH_new < eH:
                    x = y
                why = f"the cheap level failed at an expensive point: {e}"
                x, eH, gH, k, j = self._plain(x, why, history)
                return x, eH, gH, n_steps + k, n_rejected + j, history, why
            actual = eH_new - eH
            ratio = actual / predicted
            step = (y - x0).ravel()
            # learn the Hessian of the difference from this pair, kept or not
            if self.variant == "delta-hessian":
                dg_new = (gH_new - gL_new).ravel()
                dB = _sr1(dB, step, dg_new - dg, rigid_modes(x0))
            length = float(np.max(np.linalg.norm(step.reshape(-1, 3), axis=1)))
            poor = poor + 1 if ratio < 0.25 else 0
            if actual > 1.0e-6 and ratio < -1.0:  # predicted is a decrease here
                n_rejected += 1
                trust = max(self.min_trust, 0.5 * min(trust, length))
                history.append({"rejected": True, "energy": eH_new, "trust": trust})
                continue
            x, eH, gH, eL, gL = y, eH_new, gH_new, eL_new, gL_new
            if ratio > 0.75:
                trust = min(self.max_trust, np.sqrt(2.0) * trust)
            elif ratio < 0.25:
                trust = max(self.min_trust, 0.5 * min(trust, length))
            history.append(
                {
                    "energy": eH,
                    "max_force": _max_force(gH),
                    "trust": trust,
                    "ratio": ratio,
                    "step": length,
                    "kind": kind,
                    "micro": micro,
                }
            )
        return x, eH, gH, n_steps, n_rejected, history, None

    # ------------------------------------------------------------------ run
    def run(self):
        """Minimize, then check the end point (the cheap level's imaginary modes
        confirmed at the expensive level, as Optimizer does) and escape from a
        confirmed saddle: displace along the mode both ways, minimize each, keep the
        lower; up to escape_rounds rounds."""
        x, eH, gH, n_steps, n_rejected, history, why = self._descend(self.xyz0)
        result = SurrogateResult(
            xyz=x,
            energy=eH,
            gradient=gH,
            converged=self._converged(gH),
            n_steps=n_steps,
            n_rejected=n_rejected,
            history=history,
            fallback=why,
        )
        checker = self.hessian_engine or self.cheap_hessian_engine
        if self.check_minimum and checker is not None and result.converged:
            check = Optimizer(
                self._expensive,
                self.symbols,
                x,
                fmax=self.fmax,
                hessian_engine=self.hessian_engine,
                cheap_hessian_engine=self.cheap_hessian_engine,
            )
            for round_ in range(self.escape_rounds + 1):
                nu, mode = check._minimum_check(result.xyz)
                result.lowest_frequency, result.minimum = nu, mode is None
                if result.minimum or round_ == self.escape_rounds:
                    break
                d = mode * (
                    self.escape_displacement / np.max(np.linalg.norm(mode, axis=1))
                )
                attempts = [
                    self._descend(result.xyz + sign * d) for sign in (1.0, -1.0)
                ]
                best = min(attempts, key=lambda a: (not self._converged(a[2]), a[1]))
                result.escapes.append(
                    {
                        "round": round_ + 1,
                        "frequency": nu,
                        "energies": [a[1] for a in attempts],
                        "saddle energy": result.energy,
                    }
                )
                result.xyz, result.energy, result.gradient = best[0], best[1], best[2]
                result.converged = self._converged(best[2])
                result.n_steps += best[3]
                result.n_rejected += best[4]
                result.history += best[5]
                result.fallback = result.fallback or best[6]
                if not result.converged:
                    break
            result.confirmations = check.confirmations
        result.n_gradients = self.n_gradients
        result.n_cheap_gradients = self.n_cheap_gradients
        return result


class _Leave(Exception):
    """The micro-iterations left the region around the trust radius, or the cheap
    level failed there."""


#: the largest |eigenvalue| (eV/Å^2) of the learned Hessian difference: differences
#: between levels are much smaller than the Hessians themselves (stretches ~30-60)
DB_CAP = 10.0


def _sr1(B, s, y, rigid=None, cap=DB_CAP):
    """The symmetric-rank-one update of B with step s and gradient change y, in the
    space without overall translation and rotation; skipped when ill-conditioned, and
    the result's eigenvalues held within +-cap (an ill-conditioned update once
    overflowed)."""
    if rigid is not None and rigid.shape[1] > 0:
        P = np.eye(len(s)) - rigid @ rigid.T
        s, y = P @ s, P @ y
    r = y - B @ s
    denominator = float(r @ s)
    if not np.isfinite(denominator) or abs(denominator) < 1.0e-6 * float(
        np.linalg.norm(r) * np.linalg.norm(s)
    ):
        return B
    if float(r @ r) > 1.0e6 * abs(denominator):  # an update far beyond the cap
        return B
    B = B + np.outer(r, r) / denominator
    w, v = np.linalg.eigh(0.5 * (B + B.T))
    if not np.all(np.isfinite(w)):
        return np.zeros_like(B)
    return (v * np.clip(w, -cap, cap)) @ v.T


def _max_force(g):
    return float(np.max(np.linalg.norm(g, axis=1)))
