# -*- coding: utf-8 -*-

"""The optimizer: quasi-Newton minimization with an RFO / trust-region step, in
Cartesians or internal coordinates (geomeTRIC's TRIC, DLC, HDLC or primitives), a
model or computed starting Hessian, BFGS updates in the coordinates used, a checked
back-transformation with a Cartesian fallback, and a check that the end point is a
minimum, with an escape from a saddle point.

The engine is any callable ``engine(xyz) -> (energy, gradient)`` with xyz an N x 3
array in Å, the energy in eV and the gradient N x 3 in eV/Å. Two Hessian engines,
each ``(xyz) -> 3N x 3N`` in eV/Å^2, are optional:

- ``hessian_engine``, at the expensive level: the starting Hessian with
  hessian="computed", and the minimum check;
- ``cheap_hessian_engine``, at a cheap level (PM7, GFN2-xTB, an MLFF, a force field):
  the starting Hessian with hessian="cheap", refreshed when the quadratic model keeps
  predicting the energy change badly (`refresh`), and the minimum check when there is
  no expensive one (proposal §5).

Convergence is the shared criterion of the benchmark: the largest force on any atom
at most `fmax` (default 0.0015 eV/Å, the CBH criterion).
"""

from dataclasses import dataclass, field
import logging

import numpy as np

from . import coordinates as coordinate_systems
from .geometry import _EV_A2_AMU_TO_CM2, frequencies
from .model_hessian import HARTREE, lindh_hessian
from .steps import bfgs_update, positive_hessian, rfo_step

logger = logging.getLogger(__name__)

#: how many recent steps judge a refreshed Hessian
REFRESH_PAIRS = 5

#: ASE BFGS's starting Hessian, a scaled identity, kept for comparisons
ASE_IDENTITY = 70.0  # eV/Å^2


@dataclass
class Result:
    xyz: np.ndarray
    energy: float
    gradient: np.ndarray
    converged: bool
    n_gradients: int = 0
    n_hessians: int = 0
    n_cheap_hessians: int = 0
    n_refreshes: int = 0
    n_refreshes_declined: int = 0
    n_steps: int = 0
    n_rejected: int = 0
    n_fallbacks: int = 0
    n_rebuilds: int = 0
    coordinates: str = ""
    last_step: float = 0.0
    minimum: bool | None = None
    lowest_frequency: float | None = None
    escapes: list = field(default_factory=list)
    confirmations: list = field(default_factory=list)
    history: list = field(default_factory=list)

    @property
    def max_force(self):
        return float(np.max(np.linalg.norm(self.gradient, axis=1)))


class Optimizer:
    """Minimize the energy from the engine, starting at xyz.

    Parameters
    ----------
    engine : callable
        ``engine(xyz) -> (energy eV, gradient N x 3 eV/Å)``.
    symbols : list of str
    xyz : N x 3 array (Å)
    fmax : float
        Converged when the largest atomic force is at most this (eV/Å).
    max_steps : int
        Default 50 + 12 N.
    hessian : "lindh" | "geometric" | "cheap" | "identity" | "computed" | array |
        callable
        The starting Hessian: Lindh's model, geomeTRIC's guess (internal coordinates
        only), the cheap Hessian engine's, ASE's scaled identity, the expensive
        Hessian engine's, a given 3N x 3N array, or ``callable(symbols, xyz)``.
    hessian_engine : callable, optional
        ``hessian_engine(xyz) -> 3N x 3N (eV/Å^2)`` at the expensive level, for
        "computed" and the minimum check.
    cheap_hessian_engine : callable, optional
        The same at a cheap level, for "cheap", refreshes and (without an expensive
        one) the minimum check.
    refresh : "adaptive" | "never" | int
        With a cheap Hessian engine: recompute the Hessian after `refresh_after` steps
        in a row whose actual energy change disagreed with the prediction (ratio
        outside 0.5-1.5), or every `refresh` steps when an int.
    refresh_after : int
    coordinates : str
        "TRIC" (default), "DLC", "HDLC", "primitive" or "cartesian".
    trust, max_trust, min_trust : float
        The trust radius and its limits, as the length of the step in the
        coordinates used (bohr and radians; bohr for Cartesians).
    check_minimum : bool
        With a Hessian engine, check the end point and escape from a saddle point.
    imaginary_threshold : float
        Imaginary frequencies (cm^-1) beyond this make a saddle point (30).
    escape_displacement : float
        The largest atomic displacement (Å) along the imaginary mode.
    escape_rounds : int
        How many times to escape from a saddle point at most.
    confirm_displacement : float
        The largest atomic displacement (Å) for confirming, from expensive gradients,
        an imaginary mode that only the cheap level's Hessian shows (two gradients
        per mode), before escaping along it.
    """

    def __init__(
        self,
        engine,
        symbols,
        xyz,
        fmax=0.0015,
        max_steps=None,
        hessian="lindh",
        hessian_engine=None,
        cheap_hessian_engine=None,
        refresh="adaptive",
        refresh_after=3,
        coordinates="TRIC",
        trust=0.5,
        max_trust=1.0,
        min_trust=0.002,
        check_minimum=True,
        imaginary_threshold=30.0,
        escape_displacement=0.3,
        escape_rounds=2,
        confirm_displacement=0.02,
    ):
        self.engine = engine
        self.symbols = list(symbols)
        self.xyz0 = np.asarray(xyz, float).copy()
        self.n = len(self.symbols)
        self.fmax = fmax
        self.max_steps = max_steps if max_steps is not None else 50 + 12 * self.n
        self.hessian = hessian
        self.hessian_engine = hessian_engine
        self.cheap_hessian_engine = cheap_hessian_engine
        self.refresh = refresh
        self.refresh_after = refresh_after
        self.coordinates = coordinates
        self.trust0 = trust
        self.max_trust = max_trust
        self.min_trust = min_trust
        self.check_minimum = check_minimum
        self.imaginary_threshold = imaginary_threshold
        self.escape_displacement = escape_displacement
        self.escape_rounds = escape_rounds
        self.confirm_displacement = confirm_displacement
        self.confirmations = []
        self.n_gradients = 0
        self.n_hessians = 0
        self.n_cheap_hessians = 0
        self.n_refreshes = 0
        self.n_refreshes_declined = 0

    # ------------------------------------------------------------------ engines
    def _evaluate(self, xyz):
        energy, gradient = self.engine(xyz)
        self.n_gradients += 1
        return float(energy), np.asarray(gradient, float).reshape(self.n, 3)

    def _computed_hessian(self, xyz):
        if self.hessian_engine is None:
            raise ValueError("hessian='computed' needs a hessian_engine")
        self.n_hessians += 1
        H = np.asarray(self.hessian_engine(xyz), float)
        return 0.5 * (H + H.T)

    def _cheap_hessian(self, xyz):
        if self.cheap_hessian_engine is None:
            raise ValueError("hessian='cheap' needs a cheap_hessian_engine")
        self.n_cheap_hessians += 1
        H = np.asarray(self.cheap_hessian_engine(xyz), float)
        return 0.5 * (H + H.T)

    def _starting_hessian(self, xyz):
        choice = self.hessian
        if isinstance(choice, np.ndarray):
            return choice.copy()
        if callable(choice):
            return np.asarray(choice(self.symbols, xyz), float)
        if choice == "lindh":
            return lindh_hessian(self.symbols, xyz)
        if choice == "identity":
            return ASE_IDENTITY * np.eye(3 * self.n)
        if choice == "computed":
            return self._computed_hessian(xyz)
        if choice == "cheap":
            return self._cheap_hessian(xyz)
        raise ValueError(f"Unknown starting Hessian {choice!r}")

    def _converged(self, gradient):
        return float(np.max(np.linalg.norm(gradient, axis=1))) <= self.fmax

    # ------------------------------------------------------------------ minimize
    def _minimize(self, xyz):
        """One quasi-Newton minimization from xyz, without the minimum check.

        Works in atomic units in the chosen coordinates; energies are reported in eV."""
        energy, gx = self._evaluate(xyz)
        coords = coordinate_systems.make(self.coordinates, self.symbols, xyz)
        gq = coords.gradient(xyz, gx)
        named = self.hessian if isinstance(self.hessian, str) else None
        if named == "geometric":
            if not hasattr(coords, "guess_hessian"):
                raise ValueError("hessian='geometric' needs internal coordinates")
            hq = coords.guess_hessian(xyz)
        else:
            hq = coords.hessian(xyz, gx, self._starting_hessian(xyz))
            # a computed Hessian -- the cheap level's, the expensive one's, or one
            # given as an array -- may be indefinite away from a minimum: clamp it
            # (an unclamped array Hessian crept with a tiny trust radius, Design
            # 2026-11-25); the model Hessians are positive already
            if named in ("cheap", "computed") or named is None:
                hq = positive_hessian(hq)
        trust = self.trust0
        history = [{"energy": energy, "max_force": _max_force(gx), "trust": trust}]
        n_steps = n_rejected = n_fallbacks = n_rebuilds = poor = 0
        # the recent steps and gradient changes in the current coordinates, for
        # judging a refreshed Hessian (cleared when the coordinates are rebuilt)
        pairs = []
        last_step = 0.0
        while not self._converged(gx) and n_steps < self.max_steps:
            n_steps += 1
            null = coords.rigid(xyz)
            dq, predicted, kind = rfo_step(gq, hq, trust, null=null)
            xyz_new, ok = coords.to_cartesian(xyz, dq)
            shrink = 0
            while not ok and shrink < 2:
                shrink += 1
                dq, predicted, kind = rfo_step(
                    gq, hq, 0.5 * np.linalg.norm(dq), null=null
                )
                xyz_new, ok = coords.to_cartesian(xyz, dq)
            if not ok:
                # the internal step cannot be realized: a Cartesian step from the same
                # model, then new internal coordinates at the new structure
                n_fallbacks += 1
                kind = "Cartesian fallback"
                cart = coordinate_systems.Cartesian(self.symbols, xyz)
                hc = cart.hessian(xyz, gx, coords.to_cartesian_hessian(xyz, hq))
                dc, predicted, _ = rfo_step(
                    cart.gradient(xyz, gx), hc, min(trust, 0.3), null=cart.rigid(xyz)
                )
                xyz_new, _ = cart.to_cartesian(xyz, dc)
            energy_new, gx_new = self._evaluate(xyz_new)
            actual = (energy_new - energy) / HARTREE
            ratio = actual / predicted if predicted < 0.0 else -1.0
            if kind == "Cartesian fallback":
                hx = coords.to_cartesian_hessian(xyz, hq)
                coords = coordinate_systems.make(
                    self.coordinates, self.symbols, xyz_new
                )
                n_rebuilds += 1
                pairs = []
                hq = coords.hessian(xyz_new, gx_new, hx)
                gq_new = coords.gradient(xyz_new, gx_new)
                gq_old = coords.gradient(xyz, gx)
                step = coords.difference(xyz_new, xyz)
                hq, _ = bfgs_update(
                    hq, step, gq_new - gq_old, null=coords.rigid(xyz_new)
                )
            else:
                gq_new = coords.gradient(xyz_new, gx_new)
                step = coords.difference(xyz_new, xyz)
                # the curvature information is valid whether or not the step is kept
                hq, _ = bfgs_update(hq, step, gq_new - gq, null=null)
                pairs = (pairs + [(step, gq_new - gq)])[-REFRESH_PAIRS:]
            length = float(np.linalg.norm(step))
            # Only a step much worse than predicted (ratio < -1) is rejected, as in
            # geomeTRIC: rejecting every small rise cost a gradient each time and,
            # with the trust cut to the step, made the optimizer crawl (Design
            # 2026-11-24, MMFF proxy: an exact Hessian took 20 gradients against 15)
            if energy_new - energy > 1.0e-6 and ratio < -1.0:
                n_rejected += 1
                trust = max(self.min_trust, 0.5 * min(trust, length))
                if kind == "Cartesian fallback":
                    # the coordinates were rebuilt at the rejected structure: rebuild
                    # them where we stay
                    hx = coords.to_cartesian_hessian(xyz_new, hq)
                    coords = coordinate_systems.make(
                        self.coordinates, self.symbols, xyz
                    )
                    hq = coords.hessian(xyz, gx, hx)
                    gq = coords.gradient(xyz, gx)
                    pairs = []
                history.append(
                    {
                        "rejected": True,
                        "energy": energy_new,
                        "trust": trust,
                        "kind": kind,
                    }
                )
                continue
            last_step = float(np.max(np.linalg.norm(xyz_new - xyz, axis=1)))
            xyz, energy, gx, gq = xyz_new, energy_new, gx_new, gq_new
            # refresh the Hessian from the cheap level when the model keeps failing
            if self.cheap_hessian_engine is not None and self.refresh != "never":
                poor = poor + 1 if not 0.5 <= ratio <= 1.5 else 0
                due = (
                    n_steps % self.refresh == 0
                    if isinstance(self.refresh, int)
                    else poor >= self.refresh_after
                )
                if due and not self._converged(gx):
                    hq = self._refreshed(coords, xyz, gx, hq, pairs)
                    poor = 0
            # good steps grow the trust radius, poor ones halve it (from the smaller
            # of the radius and the step), okay ones leave it
            if ratio > 0.75:
                trust = min(self.max_trust, np.sqrt(2.0) * trust)
            elif ratio < 0.25:
                trust = max(self.min_trust, 0.5 * min(trust, length))
            history.append(
                {
                    "energy": energy,
                    "max_force": _max_force(gx),
                    "trust": trust,
                    "ratio": ratio,
                    "step": length,
                    "kind": kind,
                }
            )
        return Result(
            xyz=xyz,
            energy=energy,
            gradient=gx,
            converged=self._converged(gx),
            n_steps=n_steps,
            n_rejected=n_rejected,
            n_fallbacks=n_fallbacks,
            n_rebuilds=n_rebuilds,
            coordinates=coords.name,
            last_step=last_step,
            history=history,
        )

    # ------------------------------------------------------------------ minimum check
    def _refreshed(self, coords, xyz, gx, hq, pairs):
        """The Hessian after a refresh from the cheap level: the cheap Hessian with
        the recent BFGS updates replayed on it, but only if it reproduces those
        steps' gradient changes better than the current model; otherwise the current
        model. A cheap level that disagrees with the expensive one (MMFF under
        GFN2-xTB) was refreshed 4 times and threw away what BFGS had learned: 26
        gradients against 20 with Lindh's model alone (Design 2026-11-24)."""
        null = coords.rigid(xyz)
        candidate = positive_hessian(coords.hessian(xyz, gx, self._cheap_hessian(xyz)))
        for s, y in pairs:
            candidate, _ = bfgs_update(candidate, s, y, null=null)
        if pairs:
            P = np.eye(len(hq)) - null @ null.T if null.shape[1] else np.eye(len(hq))

            def misfit(H):
                return sum(
                    float(np.linalg.norm(P @ (H @ s - y)) ** 2) for s, y in pairs
                )

            if misfit(candidate) >= misfit(hq):
                self.n_refreshes_declined += 1
                return hq
        self.n_refreshes += 1
        return candidate

    def _minimum_check(self, xyz):
        """(lowest frequency in cm^-1, the Cartesian mode to escape along or None).

        With a Hessian at the expensive level the frequencies are its own. With only
        the cheap level's, an imaginary mode there may not be one on the expensive
        surface: PM7's minimum of an amine is a ~180i saddle on GFN2-xTB, and escaping
        along it twice both ways cost 6x the gradients and ended "not a minimum"
        (Design 2026-11-24). So each such mode is confirmed first, from the expensive
        gradients displaced along it both ways: the curvature d.(g(+) - g(-))/2h, with
        d the mode as frequencies() gives it (M^-1/2 times the mass-weighted mode)."""
        if self.hessian_engine is not None:
            H, cheap = self._computed_hessian(xyz), False
        else:
            H, cheap = self._cheap_hessian(xyz), True
        nu, modes = frequencies(self.symbols, xyz, H)
        order = np.argsort(nu)
        suspects = [k for k in order if nu[k] < -self.imaginary_threshold]
        if not cheap or not suspects:
            k = int(order[0])
            return float(nu[k]), (modes[:, k].reshape(self.n, 3) if suspects else None)
        others = [nu[k] for k in order if nu[k] >= -self.imaginary_threshold]
        lowest = float(min(others)) if others else np.inf
        for k in suspects:
            d = modes[:, k].reshape(self.n, 3)
            h = self.confirm_displacement / float(np.max(np.linalg.norm(d, axis=1)))
            _, gp = self._evaluate(xyz + h * d)
            _, gm = self._evaluate(xyz - h * d)
            curvature = float(np.sum(d * (gp - gm))) / (2.0 * h)
            nu_x = float(
                np.sign(curvature) * np.sqrt(abs(curvature) * _EV_A2_AMU_TO_CM2)
            )
            self.confirmations.append({"cheap": float(nu[k]), "expensive": nu_x})
            if nu_x < -self.imaginary_threshold:
                return nu_x, d
            lowest = min(lowest, nu_x)
        return lowest, None

    def run(self):
        """Minimize, then (with a Hessian engine) check the end point and escape from
        a saddle point: displace along the imaginary mode both ways, re-minimize each,
        keep the lower; up to escape_rounds rounds."""
        result = self._minimize(self.xyz0)
        checker = self.hessian_engine or self.cheap_hessian_engine
        if self.check_minimum and checker is not None and result.converged:
            for round_ in range(self.escape_rounds + 1):
                nu, mode = self._minimum_check(result.xyz)
                result.lowest_frequency = nu
                result.minimum = mode is None
                if result.minimum or round_ == self.escape_rounds:
                    break
                d = mode * (
                    self.escape_displacement / np.max(np.linalg.norm(mode, axis=1))
                )
                attempts = [
                    self._minimize(result.xyz + sign * d) for sign in (1.0, -1.0)
                ]
                best = min(attempts, key=lambda r: (not r.converged, r.energy))
                result.escapes.append(
                    {
                        "round": round_ + 1,
                        "frequency": nu,
                        "energies": [a.energy for a in attempts],
                        "saddle energy": result.energy,
                    }
                )
                best.escapes, best.history = (
                    result.escapes,
                    result.history + best.history,
                )
                best.n_steps += result.n_steps
                best.n_rejected += result.n_rejected
                best.n_fallbacks += result.n_fallbacks
                best.n_rebuilds += result.n_rebuilds
                result = best
        result.n_gradients = self.n_gradients
        result.n_hessians = self.n_hessians
        result.n_cheap_hessians = self.n_cheap_hessians
        result.n_refreshes = self.n_refreshes
        result.n_refreshes_declined = self.n_refreshes_declined
        result.confirmations = self.confirmations
        return result


def _max_force(gradient):
    return float(np.max(np.linalg.norm(gradient, axis=1)))
