# -*- coding: utf-8 -*-

"""Model Hessians: cheap starting guesses for the optimizer.

lindh_hessian: R. Lindh, A. Bernhardsson, G. Karlström, P.-Å. Malmqvist,
Chem. Phys. Lett. 241, 423 (1995). Force constants for every stretch, bend and
torsion, each weighted by distance-dependent factors

    rho_ij = exp(alpha_ij (r_ref,ij^2 - r_ij^2))

so that no bonding pattern is needed: bonded atoms get rho ~ 1, distant ones ~ 0.
Parameters by periodic-table row (1: H-He, 2: Li-Ne, 3: Na and beyond), in atomic
units. The Hessian is returned in Cartesians, eV/Å^2.
"""

import itertools

import numpy as np

from .geometry import b_bend, b_stretch, b_torsion

BOHR = 0.529177210903  # Å
HARTREE = 27.211386245988  # eV

_ALPHA = np.array(
    [[1.0000, 0.3949, 0.3949], [0.3949, 0.2800, 0.2800], [0.3949, 0.2800, 0.2800]]
)
_RREF = np.array([[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]])
K_STRETCH, K_BEND, K_TORSION = 0.45, 0.15, 0.005
#: products of rho smaller than this are left out
RHO_CUTOFF = 1.0e-4

_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


def _row(symbol):
    z = _Z.get(symbol, 19)
    return 0 if z <= 2 else (1 if z <= 10 else 2)


def rho_matrix(symbols, xyz):
    """The Lindh rho_ij factors (dimensionless), N x N, zero on the diagonal."""
    rows = np.array([_row(s) for s in symbols])
    r = np.linalg.norm(xyz[:, None] - xyz[None], axis=2) / BOHR
    alpha = _ALPHA[rows[:, None], rows[None, :]]
    rref = _RREF[rows[:, None], rows[None, :]]
    rho = np.exp(alpha * (rref**2 - r**2))
    np.fill_diagonal(rho, 0.0)
    return rho


def lindh_hessian(symbols, xyz, cutoff=RHO_CUTOFF):
    """Lindh's model Hessian in Cartesians (3N x 3N, eV/Å^2)."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    rho = rho_matrix(symbols, xyz)
    xb = xyz / BOHR  # B vectors in atomic units: the force constants are
    H = np.zeros((3 * n, 3 * n))
    for i, j in itertools.combinations(range(n), 2):
        if rho[i, j] < cutoff:
            continue
        _, b = b_stretch(xb, i, j)
        H += K_STRETCH * rho[i, j] * np.outer(b, b)
    # bends i-j-k (apex j) and torsions i-j-k-m: only over pairs that are "bonded"
    neighbours = [np.nonzero(rho[a] >= cutoff)[0] for a in range(n)]
    for j in range(n):
        for i, k in itertools.combinations(neighbours[j], 2):
            weight = rho[i, j] * rho[j, k]
            if weight < cutoff:
                continue
            bend = b_bend(xb, i, j, k)
            if bend is None:
                continue
            H += K_BEND * weight * np.outer(bend[1], bend[1])
    for j, k in itertools.combinations(range(n), 2):
        if rho[j, k] < cutoff:
            continue
        for i in neighbours[j]:
            if i == k:
                continue
            for m in neighbours[k]:
                if m in (i, j):
                    continue
                weight = rho[i, j] * rho[j, k] * rho[k, m]
                if weight < cutoff:
                    continue
                torsion = b_torsion(xb, i, j, k, m)
                if torsion is None:
                    continue
                H += K_TORSION * weight * np.outer(torsion[1], torsion[1])
    return H * HARTREE / BOHR**2
