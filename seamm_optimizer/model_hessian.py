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

from .geometry import b_bend, b_stretch, b_torsion, rigid_modes

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


#: torsions with a bend this close to linear (|cos| above it, about 162 degrees) are
#: left out, as geomeTRIC leaves out such dihedrals: their B vectors grow as
#: 1/sin(angle), squared in the Hessian. The rho-weighted torsions include non-bonded
#: pairs, e.g. across a benzene ring (isopropyl C - ipso C ... para C - H), which made
#: cumene's model Hessian ~300x too stiff there (Design 2026-11-24).
LINEAR_COS = 0.95


def _near_linear(xyz, i, j, k):
    u, v = xyz[i] - xyz[j], xyz[k] - xyz[j]
    return abs(float(u @ v)) > LINEAR_COS * float(np.linalg.norm(u) * np.linalg.norm(v))


def _linear_bend_pair(xyz, i, j, k):
    """The B vectors of the two bends of a near-linear i-j-k, in the directions p
    perpendicular to the i-k axis: the bend toward p is p.(x_i - x_j)/r_ij +
    p.(x_k - x_j)/r_kj."""
    n = len(xyz)
    axis = xyz[k] - xyz[i]
    axis = axis / np.linalg.norm(axis)
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    p1 = reference - (reference @ axis) * axis
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(axis, p1)
    rij = np.linalg.norm(xyz[i] - xyz[j])
    rkj = np.linalg.norm(xyz[k] - xyz[j])
    pair = []
    for p in (p1, p2):
        b = np.zeros((n, 3))
        b[i], b[k] = p / rij, p / rkj
        b[j] = -b[i] - b[k]
        pair.append(b.ravel())
    return pair


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
    linear = False
    for j in range(n):
        for i, k in itertools.combinations(neighbours[j], 2):
            weight = rho[i, j] * rho[j, k]
            if weight < cutoff:
                continue
            if _near_linear(xb, i, j, k):
                # a near-linear bend has two bending directions, and an ordinary
                # bend describes only one: a linear-bend pair, as geomeTRIC uses, so
                # that the perpendicular bend is not left without curvature
                for b in _linear_bend_pair(xb, i, j, k):
                    H += K_BEND * weight * np.outer(b, b)
                linear = True
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
                if _near_linear(xb, i, j, k) or _near_linear(xb, j, k, m):
                    continue
                torsion = b_torsion(xb, i, j, k, m)
                if torsion is None:
                    continue
                H += K_TORSION * weight * np.outer(torsion[1], torsion[1])
    if linear:
        # the pair's directions are fixed in space, so for a bend that is only nearly
        # linear it has a little overall rotation in it: project that out
        Q = rigid_modes(xyz)
        P = np.eye(3 * n) - Q @ Q.T
        H = P @ H @ P
    return H * HARTREE / BOHR**2
