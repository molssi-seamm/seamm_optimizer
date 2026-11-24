# -*- coding: utf-8 -*-

"""Phase 4: the cheap level carries the steps (SurrogateOptimizer)."""

import numpy as np
import pytest

from seamm_optimizer import Optimizer, SurrogateOptimizer

pytest.importorskip("rdkit")
from tests.mmff import MMFF  # noqa: E402


def _start(m, amplitude=0.1, seed=3):
    return m.xyz + amplitude * np.random.default_rng(seed).normal(size=m.xyz.shape)


def test_a_perfect_surrogate_needs_few_expensive_gradients():
    """With the expensive level as its own surrogate, S is the exact surface: the
    micro-iterations carry the optimization and the expensive level is needed only
    as often as the trust radius allows steps (under a third of a plain run)."""
    m = MMFF("CC(=O)NC")
    x = _start(m)
    r = SurrogateOptimizer(m.engine, m.engine, m.symbols, x, check_minimum=False).run()
    plain = Optimizer(m.engine, m.symbols, x, check_minimum=False).run()
    assert r.converged and 3 * r.n_gradients < plain.n_gradients
    assert r.n_cheap_gradients > r.n_gradients


@pytest.mark.parametrize("variant", ["delta-hessian", "delta-gradient"])
def test_a_surrogate_with_the_wrong_curvature_still_reaches_the_minimum(variant):
    """A surrogate whose curvature is 40% off (the same surface scaled by 0.6):
    the Δ terms correct it, and the end point is the expensive level's minimum."""
    m = MMFF("CCC(=O)OC")
    x = _start(m)

    def soft(xyz):
        e, g = m.engine(xyz)
        return 0.6 * e, 0.6 * np.asarray(g)

    r = SurrogateOptimizer(
        m.engine, soft, m.symbols, x, variant=variant, check_minimum=False
    ).run()
    reference = Optimizer(m.engine, m.symbols, x, check_minimum=False).run()
    assert r.converged and r.max_force <= 0.0015
    assert abs(r.energy - reference.energy) < 1.0e-3  # eV, the same minimum


def test_the_delta_hessian_learns_what_the_delta_gradient_cannot():
    """With a surrogate whose curvature is wrong, learning ΔB takes no more expensive
    gradients than the Δ-gradient method (ΔB = 0)."""
    m = MMFF("CCC(=O)OC")
    x = _start(m)

    def soft(xyz):
        e, g = m.engine(xyz)
        return 0.6 * e, 0.6 * np.asarray(g)

    dh = SurrogateOptimizer(m.engine, soft, m.symbols, x, check_minimum=False).run()
    dg = SurrogateOptimizer(
        m.engine, soft, m.symbols, x, variant="delta-gradient", check_minimum=False
    ).run()
    assert dh.converged and dg.converged
    assert dh.n_gradients <= dg.n_gradients


def test_an_unknown_variant_is_refused():
    m = MMFF("C")
    with pytest.raises(ValueError):
        SurrogateOptimizer(m.engine, m.engine, m.symbols, m.xyz, variant="magic")
