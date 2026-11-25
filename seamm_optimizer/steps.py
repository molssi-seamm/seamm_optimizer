# -*- coding: utf-8 -*-

"""Steps and Hessian updates for minimization.

rfo_step: the rational-function (RFO) step (Banerjee, Adams, Simons, Shepard, J.
Phys. Chem. 89, 52 (1985)), and when it is longer than the trust radius, the
restricted step of that length (the trust-region subproblem, solved for the level
shift by Newton iterations on 1/|s|). Both are taken in the space orthogonal to
`null` (overall translation and rotation for Cartesians).

bfgs_update: the BFGS update, skipped when the curvature condition fails, so the
Hessian stays positive definite for minimization.
"""

import numpy as np


def _eig(H, null=None):
    """Eigenvalues and vectors of H in the space orthogonal to the columns of null."""
    if null is not None and null.shape[1] > 0:
        # an orthonormal basis of the complement
        q, _ = np.linalg.qr(np.hstack([null, np.eye(H.shape[0])]))
        C = q[:, null.shape[1] : H.shape[0]]
        Hc = C.T @ H @ C
        w, v = np.linalg.eigh(0.5 * (Hc + Hc.T))
        return w, C @ v
    w, v = np.linalg.eigh(0.5 * (H + H.T))
    return w, v


def rfo_step(g, H, trust, null=None):
    """(step, predicted energy change, kind) for gradient g (3N) and Hessian H.

    kind is "rfo" or "restricted" (the step on the trust sphere)."""
    w, v = _eig(H, null)
    f = v.T @ g
    # RFO: the lowest eigenvalue of the augmented Hessian [[diag(w), f], [f, 0]]
    aug = np.zeros((len(w) + 1, len(w) + 1))
    aug[: len(w), : len(w)] = np.diag(w)
    aug[: len(w), -1] = f
    aug[-1, : len(w)] = f
    lam = np.linalg.eigvalsh(aug)[0]
    denominator = w - lam
    denominator[np.abs(denominator) < 1.0e-12] = 1.0e-12
    s = -v @ (f / denominator)
    kind = "rfo"
    if np.linalg.norm(s) > trust:
        s = _restricted_step(w, v, f, trust)
        kind = "restricted"
    predicted = float(g @ s + 0.5 * s @ H @ s)
    return s, predicted, kind


def _restricted_step(w, v, f, trust):
    """The minimizer of the quadratic model on the sphere |s| = trust: s(mu) =
    -sum f_i/(w_i - mu) v_i with mu < min(w_1, 0) chosen so |s(mu)| = trust."""
    lowest = w[0]
    # bracket mu below the lowest eigenvalue
    hi = min(lowest, 0.0) - 1.0e-10
    lo = hi - 1.0

    def norm(mu):
        return np.linalg.norm(f / (w - mu))

    while norm(lo) > trust:
        lo = hi - 2.0 * (hi - lo)
    # norm(mu) decreases as mu decreases below w_1: bisect then polish
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm(mid) > trust:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-12 * max(1.0, abs(mid)):
            break
    mu = 0.5 * (lo + hi)
    s = -v @ (f / (w - mu))
    return s * (trust / np.linalg.norm(s))


def bfgs_update(H, s, y, null=None):
    """The BFGS update of H with step s and gradient change y; unchanged (and
    False returned as the second value) when s.y is not clearly positive."""
    if null is not None and null.shape[1] > 0:
        P = np.eye(len(s)) - null @ null.T
        s, y = P @ s, P @ y
    sy = float(s @ y)
    if sy <= 1.0e-8 * np.linalg.norm(s) * np.linalg.norm(y) or sy <= 0.0:
        return H, False
    Hs = H @ s
    sHs = float(s @ Hs)
    H = H + np.outer(y, y) / sy
    if sHs > 1.0e-12 * float(s @ s):
        H = H - np.outer(Hs, Hs) / sHs
    # else the model has no curvature along the step (e.g. a model Hessian for
    # unbonded atoms), so H s ~ 0: the y term alone satisfies the secant condition,
    # where the full update divided by zero (Design 2026-11-25)
    return 0.5 * (H + H.T), True


#: the eigenvalue floor (atomic units) for computed Hessians used in minimization
HESSIAN_FLOOR = 0.005


def positive_hessian(H, floor=HESSIAN_FLOOR, zero=1.0e-6):
    """A computed Hessian made fit for minimization: eigenvalues below `floor` are
    raised to it, except the near-zero ones (|w| < zero, the rigid-body directions,
    which the steps project out). A Hessian computed away from its own minimum has
    negative eigenvalues; clamped, a UFF Hessian on the MMFF proxy took 352 gradients
    over 25 CHNO PM7 starts against 504 raw (abs values: 418; Design 2026-11-24)."""
    w, v = np.linalg.eigh(0.5 * (H + H.T))
    w = np.where(np.abs(w) < zero, w, np.maximum(w, floor))
    return (v * w) @ v.T
