# -*- coding: utf-8 -*-

"""The cost-aware choice of strategy (planner) and the cost-capped micro-iterations."""

import numpy as np
import pytest

from seamm_optimizer import SurrogateOptimizer, optimize, plan

pytest.importorskip("rdkit")
from tests.mmff import MMFF  # noqa: E402


def test_a_nearly_free_cheap_level_is_used_as_a_surrogate():
    assert plan(20, {"expensive": 1.0, "cheap": 0.001}).strategy == "surrogate"


def test_an_expensive_cheap_level_is_not_worth_it():
    """r2SCAN-3c under revDSD-like costs (r ~ 0.3): Lindh alone is cheaper."""
    assert plan(30, {"expensive": 1.0, "cheap": 0.3}).strategy == "lindh"


def test_without_a_cheap_level_it_is_lindh():
    p = plan(20, {"expensive": 1.0}, cheap=False, cheap_hessian=False)
    assert p.strategy == "lindh" and list(p.expected) == ["lindh"]


def test_a_cheap_analytic_hessian_without_a_surrogate_is_used():
    p = plan(20, {"expensive": 1.0, "cheap_hessian": 0.5}, cheap=False)
    assert p.strategy == "cheap-hessian"


@pytest.mark.parametrize("ratio, cap", [(0.001, 50), (0.1, 10), (0.3, 3), (0.9, 2)])
def test_the_micro_iterations_are_capped_by_the_cost_ratio(ratio, cap):
    m = MMFF("C")
    s = SurrogateOptimizer(
        m.engine, m.engine, m.symbols, m.xyz, costs={"expensive": 1.0, "cheap": ratio}
    )
    assert s._micro_cap() == cap


def test_optimize_reports_its_plan_and_converges():
    m = MMFF("CCC(=O)OC")
    x = m.xyz + 0.1 * np.random.default_rng(3).normal(size=m.xyz.shape)

    def soft(xyz):
        e, g = m.engine(xyz)
        return 0.6 * e, 0.6 * np.asarray(g)

    r = optimize(
        m.engine,
        m.symbols,
        x,
        cheap=soft,
        costs={"expensive": 1.0, "cheap": 0.01},
        check_minimum=False,
    )
    assert r.plan.strategy == "surrogate" and r.converged
    r = optimize(
        m.engine,
        m.symbols,
        x,
        cheap=soft,
        costs={"expensive": 1.0, "cheap": 0.5},
        check_minimum=False,
    )
    assert r.plan.strategy == "lindh" and r.converged
