# -*- coding: utf-8 -*-

"""Geometry utilities: internal coordinates and their Wilson B vectors, the
projector that removes overall translation and rotation, and harmonic frequencies.

Cartesian coordinates are N x 3 arrays in Å; B vectors are flattened 3N arrays.
"""

import math

import numpy as np

#: Atomic masses (amu) of the elements, by symbol, for frequencies
_MASSES = {
    "H": 1.00784,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Br": 79.904,
    "I": 126.90447,
    "Se": 78.971,
    "As": 74.921595,
    "Ge": 72.630,
    "Sn": 118.710,
    "Ga": 69.723,
    "In": 114.818,
    "Te": 127.60,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Ti": 47.867,
    "V": 50.9415,
    "Sc": 44.955908,
}

#: 1 eV/Å^2/amu in (cm^-1)^2: nu = sqrt(k/m) / (2 pi c)
_EV_A2_AMU_TO_CM2 = (
    1.602176634e-19 / 1e-20 / 1.66053906660e-27 / (2 * math.pi * 2.99792458e10) ** 2
)


def masses(symbols):
    """The masses (amu) of the atoms."""
    try:
        return np.array([_MASSES[s] for s in symbols])
    except KeyError as e:
        raise ValueError(f"No mass for element {e}") from None


def distance(xyz, i, j):
    return float(np.linalg.norm(xyz[i] - xyz[j]))


def b_stretch(xyz, i, j):
    """(r, B vector) of the distance i-j."""
    n = len(xyz)
    d = xyz[i] - xyz[j]
    r = float(np.linalg.norm(d))
    u = d / r
    b = np.zeros((n, 3))
    b[i], b[j] = u, -u
    return r, b.ravel()


def b_bend(xyz, i, j, k):
    """(angle i-j-k in radians, B vector), j the apex; None near 0 or 180 degrees."""
    n = len(xyz)
    u, v = xyz[i] - xyz[j], xyz[k] - xyz[j]
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    u, v = u / ru, v / rv
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    s = math.sqrt(max(0.0, 1.0 - c * c))
    if s < 1.0e-3:
        return None
    phi = math.acos(c)
    bi = (c * u - v) / (ru * s)
    bk = (c * v - u) / (rv * s)
    b = np.zeros((n, 3))
    b[i], b[k], b[j] = bi, bk, -bi - bk
    return phi, b.ravel()


def b_torsion(xyz, i, j, k, m):
    """(dihedral i-j-k-m in radians, B vector); None if a bend in it is near linear."""
    n = len(xyz)
    b1, b2, b3 = xyz[j] - xyz[i], xyz[k] - xyz[j], xyz[m] - xyz[k]
    c1, c2 = np.cross(b1, b2), np.cross(b2, b3)
    n1, n2 = float(np.dot(c1, c1)), float(np.dot(c2, c2))
    r2 = float(np.linalg.norm(b2))
    if r2 < 1.0e-6 or n1 < 1.0e-8 * r2 * r2 or n2 < 1.0e-8 * r2 * r2:
        return None
    # Blondel & Karplus (1996) gradient of the dihedral
    tau = math.atan2(r2 * float(np.dot(b1, c2)), float(np.dot(c1, c2)))
    gi = -r2 / n1 * c1
    gm = r2 / n2 * c2
    # with r_ij = x_i - x_j = -b1, r_kj = b2, r_kl = x_k - x_m = -b3
    p1 = np.dot(b1, b2) / (r2 * r2)
    p3 = np.dot(b3, b2) / (r2 * r2)
    gj = (-p1 - 1.0) * gi + p3 * gm
    gk = (-p3 - 1.0) * gm + p1 * gi
    b = np.zeros((n, 3))
    b[i], b[j], b[k], b[m] = gi, gj, gk, gm
    return tau, b.ravel()


def rigid_modes(xyz, weights=None):
    """An orthonormal basis (3N x k) of the overall translations and rotations
    (k = 6, or 5 for a linear molecule, 3 for an atom), optionally mass-weighted."""
    n = len(xyz)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    center = (w[:, None] * xyz).sum(0) / w.sum()
    r = xyz - center
    vectors = []
    for axis in np.eye(3):
        vectors.append((np.tile(axis, (n, 1)) * sw[:, None]).ravel())
    for axis in np.eye(3):
        vectors.append((np.cross(axis, r) * sw[:, None]).ravel())
    V = np.array(vectors).T
    u, s, _ = np.linalg.svd(V, full_matrices=False)
    keep = s > 1.0e-6 * s.max()
    return u[:, keep]


def projector(xyz):
    """The projector (3N x 3N) onto the internal motions: translations and
    rotations removed."""
    Q = rigid_modes(xyz)
    return np.eye(3 * len(xyz)) - Q @ Q.T


def frequencies(symbols, xyz, hessian):
    """Harmonic frequencies (cm^-1, imaginary as negative) and the Cartesian normal
    modes (3N x f), translations and rotations projected out. `hessian` in eV/Å^2."""
    m = masses(symbols)
    m3 = np.repeat(m, 3)
    inv = 1.0 / np.sqrt(m3)
    Hm = hessian * inv[:, None] * inv[None, :]
    Q = rigid_modes(xyz, weights=m)
    P = np.eye(len(m3)) - Q @ Q.T
    Hm = P @ Hm @ P
    w, v = np.linalg.eigh(0.5 * (Hm + Hm.T))
    # drop the rigid-body modes: the eigenvectors with the largest overlap with Q
    overlap = np.linalg.norm(Q.T @ v, axis=0)
    order = np.argsort(-overlap)
    internal = np.sort(order[Q.shape[1] :])
    w, v = w[internal], v[:, internal]
    nu = np.sign(w) * np.sqrt(np.abs(w) * _EV_A2_AMU_TO_CM2)
    modes = v * inv[:, None]
    return nu, modes
