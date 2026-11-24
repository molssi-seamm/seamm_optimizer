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
        expensive level, as Optimizer does); no check without either.
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
        hessian_engine=None,
        cheap_hessian_engine=None,
        check_minimum=True,
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
        self.hessian_engine = hessian_engine
        self.cheap_hessian_engine = cheap_hessian_engine
        self.check_minimum = check_minimum
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

    # ------------------------------------------------------------------ run
    def run(self):
        x = self.xyz0.copy()
        eH, gH = self._expensive(x)
        eL, gL = self._cheap(x)
        dB = np.zeros((3 * self.n, 3 * self.n))
        trust = self.trust0
        history = [{"energy": eH, "max_force": _max_force(gH), "trust": trust}]
        n_steps = n_rejected = 0
        while not self._converged(gH) and n_steps < self.max_steps:
            n_steps += 1
            dE, dg = eH - eL, (gH - gL).ravel()
            x0 = x.copy()

            visited = []  # (S, point aligned onto x0) for every micro-iteration

            def S(y, x0=x0, dE=dE, dg=dg, dB=dB, visited=visited):
                y = np.asarray(y, float).reshape(-1, 3)
                el, gl = self._cheap(y)
                d = (y - x0).ravel()
                Bd = dB @ d
                value = el + dE + float(dg @ d) + 0.5 * float(d @ Bd)
                visited.append((value, _aligned(y, x0)))
                return value, gl + (dg + Bd).reshape(-1, 3)

            # converge S only as far as the expensive gradient warrants (an inexact
            # Newton step): tighter than the shared criterion only near the end
            inner = Optimizer(
                S,
                self.symbols,
                x0,
                fmax=max(self.inner_fmax, self.inner_relative * _max_force(gH)),
                max_steps=self.inner_max_steps,
                check_minimum=False,
            ).run()
            # the step: the lowest point of S on the micro-iterations' path inside
            # the trust radius. Scaling an end point back onto the trust sphere could
            # land uphill on S (an indefinite ΔB lets S run away), and such a step
            # was accepted with the expensive energy far higher (Design 2026-11-24).
            inside = [
                (value, y)
                for value, y in visited
                if float(np.max(np.linalg.norm(y - x0, axis=1))) <= trust * (1 + 1e-9)
            ]
            value, y = min(inside, key=lambda pair: pair[0])
            if value - eH >= -1.0e-9:
                # the path left the trust radius at once (the micro-iterations take
                # their own steps): backtrack along the line toward the lowest point
                # they found, at one cheap evaluation per trial
                best = min(visited, key=lambda pair: pair[0])[1]
                d = best - x0
                longest = float(np.max(np.linalg.norm(d, axis=1)))
                alpha = min(1.0, trust / longest) if longest > 0.0 else 0.0
                for _ in range(6):
                    if alpha <= 0.0:
                        break
                    trial = x0 + alpha * d
                    v, _ = S(trial)
                    if v - eH < -1.0e-9:
                        value, y = v, trial
                        break
                    alpha *= 0.5
            predicted = value - eH
            kind = (
                "surrogate"
                if inner.converged and len(inside) == len(visited)
                else ("restricted")
            )
            if predicted >= -1.0e-9:
                # nothing inside the trust radius improves S: shrink it, without
                # spending an expensive gradient
                trust = max(self.min_trust, 0.5 * trust)
                history.append(
                    {"no step": True, "trust": trust, "micro": inner.n_gradients}
                )
                if trust <= self.min_trust:
                    break
                continue
            eH_new, gH_new = self._expensive(y)
            eL_new, gL_new = self._cheap(y)
            actual = eH_new - eH
            ratio = actual / predicted
            step = (y - x0).ravel()
            # learn the Hessian of the difference from this pair, kept or not
            if self.variant == "delta-hessian":
                dg_new = (gH_new - gL_new).ravel()
                dB = _sr1(dB, step, dg_new - dg, rigid_modes(x0))
            length = float(np.max(np.linalg.norm(step.reshape(-1, 3), axis=1)))
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
                    "micro": inner.n_gradients,
                }
            )
        result = SurrogateResult(
            xyz=x,
            energy=eH,
            gradient=gH,
            converged=self._converged(gH),
            n_steps=n_steps,
            n_rejected=n_rejected,
            history=history,
        )
        checker = self.hessian_engine or self.cheap_hessian_engine
        if self.check_minimum and checker is not None and result.converged:
            # the same check as Optimizer's, with the cheap level's modes confirmed
            # at the expensive level (no escape here yet)
            check = Optimizer(
                self._expensive,
                self.symbols,
                x,
                fmax=self.fmax,
                hessian_engine=self.hessian_engine,
                cheap_hessian_engine=self.cheap_hessian_engine,
            )
            nu, mode = check._minimum_check(x)
            result.lowest_frequency, result.minimum = nu, mode is None
            result.confirmations = check.confirmations
        result.n_gradients = self.n_gradients
        result.n_cheap_gradients = self.n_cheap_gradients
        return result


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
