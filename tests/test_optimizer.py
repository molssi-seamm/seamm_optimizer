# -*- coding: utf-8 -*-

"""Tests of seamm_optimizer phase 1 on analytic models and MMFF94 surfaces (RDKit)."""

import numpy as np
import pytest

from seamm_optimizer import Optimizer, frequencies, lindh_hessian, rfo_step, rigid_modes
from seamm_optimizer.geometry import b_bend, b_stretch, b_torsion
from seamm_optimizer.steps import bfgs_update

pytest.importorskip("rdkit")
from tests.mmff import MMFF  # noqa: E402


def _numeric_b(f, xyz, h=1.0e-6):
    x0 = xyz.ravel()
    b = np.zeros(len(x0))
    for k in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h
        xm[k] -= h
        b[k] = (f(xp.reshape(-1, 3)) - f(xm.reshape(-1, 3))) / (2 * h)
    return b


@pytest.fixture
def butane():
    return MMFF("CCCC")


def test_b_vectors_match_finite_differences(butane):
    x = butane.xyz
    r, b = b_stretch(x, 0, 1)
    assert np.allclose(b, _numeric_b(lambda y: b_stretch(y, 0, 1)[0], x), atol=1e-6)
    phi, b = b_bend(x, 0, 1, 2)
    assert np.allclose(b, _numeric_b(lambda y: b_bend(y, 0, 1, 2)[0], x), atol=1e-6)
    tau, b = b_torsion(x, 0, 1, 2, 3)
    assert np.allclose(
        b, _numeric_b(lambda y: b_torsion(y, 0, 1, 2, 3)[0], x), atol=1e-6
    )


def test_lindh_hessian_is_symmetric_positive_and_invariant(butane):
    H = lindh_hessian(butane.symbols, butane.xyz)
    assert np.allclose(H, H.T)
    assert np.linalg.eigvalsh(H).min() > -1e-8  # a sum of positive outer products
    Q = rigid_modes(butane.xyz)
    assert np.abs(H @ Q).max() < 1e-6  # internal coordinates: no rigid-body curvature
    # C-C stretches are stiff, torsions soft
    assert np.linalg.eigvalsh(H).max() > 30.0


def test_rfo_step_is_newton_inside_the_trust_radius():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    H = A @ A.T + 6 * np.eye(6)
    g = rng.normal(size=6) * 0.01
    s, predicted, kind = rfo_step(g, H, trust=10.0)
    newton = -np.linalg.solve(H, g)
    assert kind == "rfo"
    assert np.allclose(s, newton, rtol=1e-3, atol=1e-6)  # RFO -> Newton for small g
    s, predicted, kind = rfo_step(g * 1000, H, trust=0.1)
    assert kind == "restricted" and np.isclose(np.linalg.norm(s), 0.1)
    assert predicted < 0


def test_bfgs_update_satisfies_the_secant_condition_and_skips_bad_curvature():
    H = np.eye(4)
    s, y = np.array([0.1, 0, 0.05, 0]), np.array([0.3, 0.01, 0.1, 0])
    H1, updated = bfgs_update(H, s, y)
    assert updated and np.allclose(H1 @ s, y)
    H2, updated = bfgs_update(H, s, -y)
    assert not updated and np.array_equal(H2, H)


def test_frequencies_of_a_minimum_are_real():
    m = MMFF("O")
    r = Optimizer(m.engine, m.symbols, m.xyz, check_minimum=False).run()
    nu, modes = frequencies(m.symbols, r.xyz, m.hessian(r.xyz))
    assert len(nu) == 3 and nu.min() > 1000.0


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "C1CCCCC1",
        "C1C2CC3CC1CC(C2)C3",
        "C12C3C4C1C5C2C3C45",
        "C#CC",
        "COCCOCCOC",
    ],
)
def test_converges_to_a_minimum_at_the_shared_criterion(smiles):
    m = MMFF(smiles)
    r = Optimizer(m.engine, m.symbols, m.xyz, hessian_engine=m.hessian).run()
    assert r.converged and r.max_force <= 0.0015
    assert r.minimum
    assert r.n_gradients == m.n_calls


def test_fewer_gradients_than_ase_bfgs_and_no_higher_energy():
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import BFGS

    class Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, engine):
            super().__init__()
            self.engine = engine

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, g = self.engine(atoms.get_positions())
            self.results = {"energy": e, "forces": -g}

    for smiles in ("C1CCCCC1", "CC(=O)NC(C)C(=O)NC"):
        ours = MMFF(smiles)
        r = Optimizer(ours.engine, ours.symbols, ours.xyz, check_minimum=False).run()
        theirs = MMFF(smiles)
        atoms = Atoms(theirs.symbols, positions=theirs.xyz)
        atoms.calc = Calc(theirs.engine)
        BFGS(atoms, logfile=None).run(fmax=0.0015, steps=2000)
        e_ase = theirs.engine(atoms.get_positions())[0]
        assert r.n_gradients < theirs.n_calls
        assert (
            r.energy <= e_ase + 5.0e-4
        )  # 0.05 kJ/mol, the benchmark's basin criterion


def test_a_planted_methyl_saddle_is_escaped():
    """Propene with the methyl torsion at the top of its barrier (an H eclipsing
    nothing: dihedral H-C-C=C 60 degrees), optimized with the torsion held, then
    released: the optimizer ends there (zero force by symmetry) and must escape."""
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
        m.engine, m.symbols, xyz, hessian_engine=m.hessian, hessian="identity"
    ).run()
    assert r.minimum, r.escapes
    dihedral = rdMolTransforms.GetDihedralDeg(
        _with_positions(m.mol, r.xyz).GetConformer(), h, 0, 1, 2
    )
    assert abs((dihedral + 60.0) % 120.0 - 60.0) < 10.0  # an H eclipses C=C again


def _with_positions(mol, xyz):
    from rdkit import Chem

    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return mol


def test_lindh_leaves_out_near_linear_torsions():
    """Lindh's rho-weighted torsions include non-bonded pairs. In cumene the
    isopropyl C, the ipso C, the para C and its H are collinear, and torsions through
    those near-linear angles (B vectors ~ 1/sin) made the model ~300x too stiff along
    the para C (38 gradients on GFN2-xTB instead of 7). A real stretch is ~30-60
    eV/Å^2; Lindh's largest eigenvalue for cumene is now ~210."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz = mol.GetConformer().GetPositions()
    assert np.linalg.eigvalsh(lindh_hessian(symbols, xyz)).max() < 400.0


def test_lindh_gives_both_bends_of_a_linear_angle():
    """A linear (or near-linear) angle bends in two directions; an ordinary bend
    describes one. Without the other, HC#CI took 54 gradients on GFN2-xTB and NCCCCN
    105 (now 6 each). Both bends of CO2 have curvature in the model."""
    symbols = ["O", "C", "O"]
    xyz = np.array([[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]])
    H = lindh_hessian(symbols, xyz)
    Q = rigid_modes(xyz)
    P = np.eye(9) - Q @ Q.T
    w = np.linalg.eigvalsh(P @ H @ P)
    assert np.sum(w > 1.0e-3) == 4  # two stretches and two bends (3N - 5 = 4)
    assert np.abs(H @ Q).max() < 1e-6


def test_a_starting_hessian_can_be_given_as_an_array():
    """Optimizer's hessian may be an array (Cartesian, eV/Å^2), as the surrogate's
    switch to the cheap-Hessian optimizer passes it; a string comparison on it once
    failed."""
    m = MMFF("CCO")
    x = m.xyz + 0.05 * np.random.default_rng(1).normal(size=m.xyz.shape)
    r = Optimizer(
        m.engine, m.symbols, x, hessian=m.hessian(x), check_minimum=False
    ).run()
    assert r.converged
