# -*- coding: utf-8 -*-

"""The optimizer: quasi-Newton minimization in Cartesian coordinates with an RFO /
trust-region step, a model or computed starting Hessian, BFGS updates, and a check
that the end point is a minimum, with an escape from a saddle point.

The engine is any callable ``engine(xyz) -> (energy, gradient)`` with xyz an N x 3
array in Å, the energy in eV and the gradient N x 3 in eV/Å. A Hessian engine,
``hessian_engine(xyz) -> 3N x 3N`` in eV/Å^2, is optional: it is used for the starting
Hessian (hessian="computed") and for the minimum check.

Convergence is the shared criterion of the benchmark: the largest force on any atom
at most `fmax` (default 0.0015 eV/Å, the CBH criterion).
"""

from dataclasses import dataclass, field
import logging

import numpy as np

from .geometry import frequencies, rigid_modes
from .model_hessian import lindh_hessian
from .steps import bfgs_update, rfo_step

logger = logging.getLogger(__name__)

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
    n_steps: int = 0
    n_rejected: int = 0
    minimum: bool | None = None
    lowest_frequency: float | None = None
    escapes: list = field(default_factory=list)
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
    hessian : "lindh" | "identity" | "computed" | array | callable
        The starting Hessian: Lindh's model, ASE's scaled identity, the Hessian
        engine's, a given 3N x 3N array, or ``callable(symbols, xyz)``.
    hessian_engine : callable, optional
        ``hessian_engine(xyz) -> 3N x 3N (eV/Å^2)``, for "computed" and the minimum
        check.
    trust, max_trust, min_trust : float
        The trust radius (Å, the length of the whole step) and its limits.
    check_minimum : bool
        With a Hessian engine, check the end point and escape from a saddle point.
    imaginary_threshold : float
        Imaginary frequencies (cm^-1) beyond this make a saddle point (30, Paul 11-24).
    escape_displacement : float
        The largest atomic displacement (Å) along the imaginary mode.
    escape_rounds : int
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
        trust=0.3,
        max_trust=0.5,
        min_trust=0.001,
        check_minimum=True,
        imaginary_threshold=30.0,
        escape_displacement=0.3,
        escape_rounds=2,
    ):
        self.engine = engine
        self.symbols = list(symbols)
        self.xyz0 = np.asarray(xyz, float).copy()
        self.n = len(self.symbols)
        self.fmax = fmax
        self.max_steps = max_steps if max_steps is not None else 50 + 12 * self.n
        self.hessian = hessian
        self.hessian_engine = hessian_engine
        self.trust0 = trust
        self.max_trust = max_trust
        self.min_trust = min_trust
        self.check_minimum = check_minimum
        self.imaginary_threshold = imaginary_threshold
        self.escape_displacement = escape_displacement
        self.escape_rounds = escape_rounds
        self.n_gradients = 0
        self.n_hessians = 0

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
        raise ValueError(f"Unknown starting Hessian {choice!r}")

    def _converged(self, gradient):
        return float(np.max(np.linalg.norm(gradient, axis=1))) <= self.fmax

    # ------------------------------------------------------------------ minimize
    def _minimize(self, xyz):
        """One quasi-Newton minimization from xyz, without the minimum check."""
        energy, gradient = self._evaluate(xyz)
        H = self._starting_hessian(xyz)
        trust = self.trust0
        history = [
            {"energy": energy, "max_force": _max_force(gradient), "trust": trust}
        ]
        n_steps = n_rejected = 0
        while not self._converged(gradient) and n_steps < self.max_steps:
            n_steps += 1
            null = rigid_modes(xyz)
            g = gradient.ravel()
            s, predicted, kind = rfo_step(g, H, trust, null=null)
            xyz_new = xyz + s.reshape(self.n, 3)
            energy_new, gradient_new = self._evaluate(xyz_new)
            actual = energy_new - energy
            ratio = actual / predicted if predicted < 0.0 else -1.0
            # the curvature information is valid whether or not the step is kept
            H, _ = bfgs_update(H, s, gradient_new.ravel() - g, null=null)
            length = float(np.linalg.norm(s))
            # a rise beyond numerical noise is rejected; tiny rises near convergence
            # are accepted (the gradient is what is converged)
            if actual > 1.0e-6 and ratio < 0.0:
                n_rejected += 1
                trust = max(self.min_trust, 0.25 * length)
                history.append({"rejected": True, "energy": energy_new, "trust": trust})
                continue
            xyz, energy, gradient = xyz_new, energy_new, gradient_new
            if ratio > 0.75 and length > 0.8 * trust:
                trust = min(self.max_trust, 2.0 * trust)
            elif ratio < 0.25:
                trust = max(self.min_trust, 0.25 * length)
            history.append(
                {
                    "energy": energy,
                    "max_force": _max_force(gradient),
                    "trust": trust,
                    "ratio": ratio,
                    "step": length,
                    "kind": kind,
                }
            )
        return Result(
            xyz=xyz,
            energy=energy,
            gradient=gradient,
            converged=self._converged(gradient),
            n_steps=n_steps,
            n_rejected=n_rejected,
            history=history,
        )

    # ------------------------------------------------------------------ minimum check
    def _lowest_mode(self, xyz):
        H = self._computed_hessian(xyz)
        nu, modes = frequencies(self.symbols, xyz, H)
        k = int(np.argmin(nu))
        return float(nu[k]), modes[:, k].reshape(self.n, 3)

    def run(self):
        """Minimize, then (with a Hessian engine) check the end point and escape from
        a saddle point: displace along the imaginary mode both ways, re-minimize each,
        keep the lower; up to escape_rounds rounds."""
        result = self._minimize(self.xyz0)
        if self.check_minimum and self.hessian_engine is not None and result.converged:
            for round_ in range(self.escape_rounds + 1):
                nu, mode = self._lowest_mode(result.xyz)
                result.lowest_frequency = nu
                result.minimum = nu > -self.imaginary_threshold
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
                result = best
        result.n_gradients = self.n_gradients
        result.n_hessians = self.n_hessians
        return result


def _max_force(gradient):
    return float(np.max(np.linalg.norm(gradient, axis=1)))
