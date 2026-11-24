# -*- coding: utf-8 -*-

"""Phase 3: cheap-level Hessians at the start, refreshed, and for the minimum check."""

import numpy as np
import pytest

from seamm_optimizer import Optimizer
from seamm_optimizer.steps import positive_hessian

pytest.importorskip("rdkit")
from tests.mmff import MMFF  # noqa: E402


def _perturbed(m, amplitude=0.12, seed=4):
    return m.xyz + amplitude * np.random.default_rng(seed).normal(size=m.xyz.shape)


def test_positive_hessian_raises_negative_and_keeps_rigid_eigenvalues():
    v, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(5, 5)))
    H = (v * np.array([-0.3, 0.0, 1e-9, 0.002, 0.7])) @ v.T
    w = np.linalg.eigvalsh(positive_hessian(H))
    assert np.allclose(sorted(w), [0.0, 1e-9, 0.005, 0.005, 0.7], atol=1e-9)


def test_a_cheap_starting_hessian_is_used_once_and_converges():
    m = MMFF("CC(=O)NC(C)C(=O)NC")
    x = _perturbed(m)
    r = Optimizer(
        m.engine, m.symbols, x, hessian="cheap", cheap_hessian_engine=m.hessian
    ).run()
    assert r.converged and r.minimum
    # one at the start and one for the minimum check (no expensive Hessian engine)
    assert r.n_cheap_hessians >= 2 and r.n_hessians == 0


def test_an_exact_hessian_beats_a_scaled_identity():
    """Not 'beats the model Hessian': from this perturbed start PEG4 takes 63 with the
    (clamped) exact Hessian and 33 with Lindh's, while over 25 CHNO PM7 starts the exact
    one takes 260 against 628."""
    m = MMFF("CC(=O)NC(C)C(=O)NC")
    x = _perturbed(m)
    identity = Optimizer(
        m.engine, m.symbols, x, check_minimum=False, hessian="identity"
    ).run()
    exact = Optimizer(
        m.engine,
        m.symbols,
        x,
        check_minimum=False,
        hessian="cheap",
        cheap_hessian_engine=m.hessian,
    ).run()
    assert exact.converged and exact.n_gradients < identity.n_gradients


def test_a_fixed_refresh_interval_refreshes():
    m = MMFF("CCCCCCO")
    x = _perturbed(m, 0.2)
    r = Optimizer(
        m.engine,
        m.symbols,
        x,
        hessian="cheap",
        cheap_hessian_engine=m.hessian,
        refresh=3,
        check_minimum=False,
    ).run()
    assert r.converged
    assert r.n_refreshes == r.n_cheap_hessians - 1 > 0


def test_the_minimum_check_uses_the_cheap_hessian_without_an_expensive_one():
    from rdkit.Chem import AllChem, rdMolTransforms

    m = MMFF("CC=C")
    h = next(
        a.GetIdx()
        for a in m.mol.GetAtomWithIdx(0).GetNeighbors()
        if a.GetSymbol() == "H"
    )
    conf = m.mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, h, 0, 1, 2, 60.0)
    ff = AllChem.MMFFGetMoleculeForceField(m.mol, m.props)
    ff.MMFFAddTorsionConstraint(h, 0, 1, 2, False, 60.0, 60.0, 1.0e4)
    ff.Minimize(maxIts=10000)
    xyz = np.array(conf.GetPositions())
    r = Optimizer(
        m.engine, m.symbols, xyz, hessian="identity", cheap_hessian_engine=m.hessian
    ).run()
    # TRIC leaves this saddle by itself; the check runs on the cheap Hessian
    assert r.minimum and r.n_hessians == 0 and r.n_cheap_hessians >= 1
