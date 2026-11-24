# -*- coding: utf-8 -*-

"""MMFF94 (RDKit) as a test engine: a real molecular surface at no cost.

engine(xyz) -> (energy eV, gradient N x 3 eV/Å); hessian(xyz) -> 3N x 3N eV/Å^2 by
central differences of the gradient. RDKit's CalcGrad is checked against finite
differences of CalcEnergy when the engine is made, and scaled to match (an earlier
test in the workspace saw them differ by ~4.18).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

KCAL_EV = 0.0433641043


class MMFF:
    def __init__(self, smiles, seed=7, xyz=None):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if xyz is None:
            AllChem.EmbedMolecule(mol, randomSeed=seed)
        else:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, p in enumerate(xyz):
                conf.SetAtomPosition(i, [float(v) for v in p])
            mol.AddConformer(conf, assignId=True)
        self.mol = mol
        self.symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        self.props = AllChem.MMFFGetMoleculeProperties(mol)
        self.ff = AllChem.MMFFGetMoleculeForceField(mol, self.props)
        self.xyz = np.array(mol.GetConformer().GetPositions())
        self.scale = self._grad_scale()
        self.n_calls = 0

    def _energy(self, xyz):
        return self.ff.CalcEnergy(xyz.ravel().tolist())

    def _grad_scale(self):
        x = self.xyz + 0.05  # off the embedded structure
        g = np.array(self.ff.CalcGrad(x.ravel().tolist()))
        k = int(np.argmax(np.abs(g)))
        h = 1.0e-4
        xp, xm = x.ravel().copy(), x.ravel().copy()
        xp[k] += h
        xm[k] -= h
        fd = (self._energy(xp) - self._energy(xm)) / (2 * h)
        return fd / g[k]

    def engine(self, xyz):
        self.n_calls += 1
        x = np.asarray(xyz, float).ravel().tolist()
        e = self.ff.CalcEnergy(x) * KCAL_EV
        g = np.array(self.ff.CalcGrad(x)) * self.scale * KCAL_EV
        return e, g.reshape(-1, 3)

    def hessian(self, xyz, h=1.0e-4):
        x0 = np.asarray(xyz, float).ravel()
        n = len(x0)
        H = np.zeros((n, n))
        for k in range(n):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            H[k] = (self.engine(xp)[1].ravel() - self.engine(xm)[1].ravel()) / (2 * h)
            self.n_calls -= 2  # not counted as optimizer gradients
        return 0.5 * (H + H.T)
