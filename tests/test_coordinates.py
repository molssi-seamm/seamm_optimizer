# -*- coding: utf-8 -*-

"""Phase 2: internal coordinates (geomeTRIC's TRIC, DLC, primitives), the checked
back-transformation with its Cartesian fallback, clusters, cages and linear groups."""

import numpy as np
import pytest

from seamm_optimizer import Optimizer
from seamm_optimizer import coordinates as cs
from seamm_optimizer.model_hessian import BOHR, HARTREE

pytest.importorskip("rdkit")
pytest.importorskip("geometric")
from tests.mmff import MMFF  # noqa: E402


def _perturbed(m, amplitude=0.1, seed=11):
    return m.xyz + amplitude * np.random.default_rng(seed).normal(size=m.xyz.shape)


@pytest.mark.parametrize("kind", ["TRIC", "DLC", "primitive"])
def test_internal_gradient_reproduces_the_cartesian_one(kind):
    m = MMFF("CCCO")
    x = _perturbed(m)
    _, gx = m.engine(x)
    c = cs.make(kind, m.symbols, x)
    gq = c.gradient(x, gx)
    gx_au = gx.ravel() * BOHR / HARTREE
    assert np.allclose(c.B(x).T @ gq, gx_au, atol=1e-6)


@pytest.mark.parametrize("kind", ["cartesian", "TRIC", "DLC"])
def test_a_small_step_round_trips(kind):
    m = MMFF("CC(=O)O")
    c = cs.make(kind, m.symbols, m.xyz)
    dq = 1.0e-3 * np.random.default_rng(2).normal(size=len(c.q(m.xyz)))
    for basis in [c.rigid(m.xyz)]:
        dq = dq - basis @ (basis.T @ dq)  # no rigid-body part
    x1, ok = c.to_cartesian(m.xyz, dq)
    assert ok
    assert np.allclose(c.difference(x1, m.xyz), dq, atol=1e-5)


def test_a_failed_back_transformation_falls_back_to_a_cartesian_step(monkeypatch):
    m = MMFF("CC(=O)NC")
    x = _perturbed(m)
    real = cs.Internal.to_cartesian
    calls = {"n": 0}

    def flaky(self, xyz, dq):
        calls["n"] += 1
        xyz_new, ok = real(self, xyz, dq)
        # three attempts in a row "fail" now and then: the step is halved twice and
        # then the optimizer must fall back to a Cartesian step
        return xyz_new, ok and calls["n"] % 8 not in (1, 2, 3)

    monkeypatch.setattr(cs.Internal, "to_cartesian", flaky)
    r = Optimizer(m.engine, m.symbols, x, hessian_engine=m.hessian).run()
    assert r.converged and r.minimum
    assert r.n_fallbacks > 0 and r.n_rebuilds == r.n_fallbacks


def test_a_water_dimer_cluster_converges_in_tric():
    m = MMFF("O.O", seed=3)
    x = m.xyz.copy()
    # RDKit orders the atoms O, O, H, H, H, H: the second water is atoms 1, 4 and 5
    second = [1, 4, 5]
    x[second] += np.array([2.6, 0.4, 0.2]) - (x[1] - x[0])  # its O ~2.7 Å away
    r = Optimizer(
        m.engine, m.symbols, x, hessian_engine=m.hessian, coordinates="TRIC"
    ).run()
    assert r.converged and r.minimum
    oo = np.linalg.norm(r.xyz[1] - r.xyz[0])
    assert 2.5 < oo < 3.2  # a hydrogen-bonded dimer


@pytest.mark.parametrize("kind", ["cartesian", "TRIC", "DLC", "primitive"])
@pytest.mark.parametrize(
    "smiles",
    [
        "C1C2CC3CC1CC(C2)C3",  # adamantane
        "C1C2CC3C4C1C5CC(C4)CC3C5C2",  # diamantane
        "C12C3C4C1C5C2C3C45",  # cubane
        "C#CC",  # propyne
        "N#CC#CC#N",  # dicyanoacetylene: linear throughout
        "C=C=C",  # allene
    ],
)
def test_cages_and_linear_groups_from_perturbed_starts(smiles, kind):
    m = MMFF(smiles)
    x = _perturbed(m, 0.08)
    r = Optimizer(
        m.engine, m.symbols, x, hessian_engine=m.hessian, coordinates=kind
    ).run()
    assert r.converged and r.max_force <= 0.0015
    assert r.minimum
    ref = Optimizer(
        m.engine, m.symbols, m.xyz, check_minimum=False, coordinates="cartesian"
    ).run()
    assert abs(r.energy - ref.energy) < 5.0e-4  # 0.05 kJ/mol: the same minimum
