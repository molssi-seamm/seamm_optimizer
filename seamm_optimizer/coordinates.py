# -*- coding: utf-8 -*-

"""Coordinate systems for the optimizer: Cartesians, and internal coordinates built
with geomeTRIC's code (TRIC, delocalized internals, redundant primitives), per the
design (proposal §4 and §10).

Every system works in atomic units internally (bohr, hartree) and offers:

- ``q(xyz)``: the coordinates of a Cartesian structure (Å in, a vector out);
- ``gradient(xyz, gx)``: the gradient in these coordinates (gx in eV/Å);
- ``hessian(xyz, gx, hx)``: a Cartesian Hessian (eV/Å^2) in these coordinates;
- ``to_cartesian(xyz, dq)``: the Cartesian structure for a step dq, and whether the
  back-transformation reproduced dq (the check of §4: a failure lets the optimizer
  shrink the step or fall back to a Cartesian step);
- ``difference(xyz1, xyz0)``: q(xyz1) - q(xyz0), with angles wrapped;
- ``rigid(xyz)``: an orthonormal basis of the coordinate changes of an overall
  translation or rotation, removed from every step.
"""

import numpy as np

from .geometry import rigid_modes
from .model_hessian import BOHR, HARTREE

#: the back-transformation is accepted when |dq_achieved - dq| <= this x |dq| (+1e-6)
BACK_TRANSFORM_TOLERANCE = 0.1


def _to_au_gradient(gx):
    return np.asarray(gx, float).ravel() * BOHR / HARTREE


def _to_au_hessian(hx):
    return np.asarray(hx, float) * BOHR**2 / HARTREE


class Cartesian:
    """Cartesian coordinates (bohr)."""

    name = "Cartesian"

    def __init__(self, symbols, xyz):
        self.symbols = list(symbols)

    def q(self, xyz):
        return np.asarray(xyz, float).ravel() / BOHR

    def gradient(self, xyz, gx):
        return _to_au_gradient(gx)

    def hessian(self, xyz, gx, hx):
        return _to_au_hessian(hx)

    def to_cartesian(self, xyz, dq):
        return np.asarray(xyz, float) + dq.reshape(-1, 3) * BOHR, True

    def difference(self, xyz1, xyz0):
        return (np.asarray(xyz1, float) - np.asarray(xyz0, float)).ravel() / BOHR

    def rigid(self, xyz):
        return rigid_modes(np.asarray(xyz, float))

    def to_cartesian_hessian(self, xyz, hq):
        """A q-space Hessian (atomic units) as a Cartesian one (eV/Å^2)."""
        return np.asarray(hq) * HARTREE / BOHR**2


class Internal:
    """geomeTRIC's internal coordinates.

    kind: "TRIC" (translation-rotation internal coordinates: each fragment's
    translations and rotations are coordinates, so clusters need no bonds between
    fragments), "DLC" (delocalized internals, fragments connected), "HDLC" (TRIC plus
    Cartesians) or "primitive" (TRIC's redundant primitives)."""

    def __init__(self, symbols, xyz, kind="TRIC"):
        from geometric.internal import (
            DelocalizedInternalCoordinates,
            PrimitiveInternalCoordinates,
        )
        from geometric.molecule import Molecule

        self.symbols = list(symbols)
        self.kind = kind
        self.name = kind
        molecule = Molecule()
        molecule.elem = list(symbols)
        molecule.xyzs = [np.asarray(xyz, float)]
        molecule.build_topology()
        if kind == "TRIC":
            self.ic = DelocalizedInternalCoordinates(
                molecule, build=True, connect=False, addcart=False
            )
        elif kind == "DLC":
            self.ic = DelocalizedInternalCoordinates(
                molecule, build=True, connect=True, addcart=False
            )
        elif kind == "HDLC":
            self.ic = DelocalizedInternalCoordinates(
                molecule, build=True, connect=False, addcart=True
            )
        elif kind == "primitive":
            self.ic = PrimitiveInternalCoordinates(
                molecule, connect=False, addcart=False
            )
        else:
            raise ValueError(f"Unknown internal coordinates {kind!r}")

    @staticmethod
    def _bohr(xyz):
        return np.asarray(xyz, float).ravel() / BOHR

    def q(self, xyz):
        return np.asarray(self.ic.calculate(self._bohr(xyz)), float)

    def gradient(self, xyz, gx):
        return np.asarray(self.ic.calcGrad(self._bohr(xyz), _to_au_gradient(gx)), float)

    def hessian(self, xyz, gx, hx):
        """A Cartesian Hessian in these coordinates, without the gradient (K) term.

        The K term is right for an exact Hessian but, applied to a model Hessian away
        from a minimum, it made the model worse: on MMFF from perturbed starts it
        took 64 gradients for PEG4 against 17 without it (Design, 2026-11-24)."""
        zero = np.zeros(3 * len(self.symbols))
        hq = self.ic.calcHess(self._bohr(xyz), zero, _to_au_hessian(hx))
        return 0.5 * (hq + hq.T)

    def guess_hessian(self, xyz, only=None):
        """geomeTRIC's own guess (a diagonal Schlegel-type model in the primitives);
        with `only` (primitive class names, e.g. ("LinearAngle", "OutOfPlane")), just
        those primitives' terms."""
        if only is None:
            hq = np.asarray(self.ic.guess_hessian(self._bohr(xyz)), float)
            return 0.5 * (hq + hq.T)
        prims = self.ic.Prims if hasattr(self.ic, "Prims") else self.ic
        hprim = np.asarray(prims.guess_hessian(self._bohr(xyz)), float)
        keep = np.array([type(ic).__name__ in only for ic in prims.Internals], float)
        hprim = hprim * keep[:, None] * keep[None, :]
        if hasattr(self.ic, "Vecs"):
            V = np.asarray(self.ic.Vecs)
            hprim = V.T @ hprim @ V
        return 0.5 * (hprim + hprim.T)

    def B(self, xyz):
        return np.asarray(self.ic.wilsonB(self._bohr(xyz)), float)

    def difference(self, xyz1, xyz0):
        return np.asarray(self.ic.calcDiff(self._bohr(xyz1), self._bohr(xyz0)), float)

    def to_cartesian(self, xyz, dq):
        new = self.ic.newCartesian(
            self._bohr(xyz), np.asarray(dq, float), verbose=False
        )
        xyz_new = np.asarray(new, float).reshape(-1, 3) * BOHR
        achieved = self.difference(xyz_new, xyz)
        error = np.linalg.norm(achieved - dq)
        ok = not getattr(self.ic, "bork", False) and error <= (
            BACK_TRANSFORM_TOLERANCE * np.linalg.norm(dq) + 1.0e-6
        )
        return xyz_new, ok

    def rigid(self, xyz):
        """The q-space directions of an overall translation or rotation."""
        Q = rigid_modes(np.asarray(xyz, float))  # Cartesian, orthonormal
        R = self.B(xyz) @ (Q * BOHR)  # dq for unit rigid displacements (bohr)
        u, s, _ = np.linalg.svd(R, full_matrices=False)
        keep = s > 1.0e-8 * max(1.0, s.max())
        return u[:, keep]

    def to_cartesian_hessian(self, xyz, hq):
        """B^T Hq B (no gradient term) as a Cartesian Hessian in eV/Å^2."""
        B = self.B(xyz)
        return (B.T @ np.asarray(hq) @ B) * HARTREE / BOHR**2


def make(kind, symbols, xyz):
    """A coordinate system: "cartesian", "TRIC", "DLC", "HDLC" or "primitive"."""
    if str(kind).lower() in ("cartesian", "cart"):
        return Cartesian(symbols, xyz)
    return Internal(symbols, xyz, kind=kind)
