#!/usr/bin/env python
"""A free proxy of the optimizer benchmark: the same starts on MMFF94 surfaces (RDKit).

Compares seamm_optimizer (Lindh model Hessian, and ASE's scaled identity to separate
the effect of the Hessian from the step) with ASE's BFGS, all stopping at the shared
0.0015 eV/Å. Reports gradients to converge, success, energy differences, and whether
the end point is a minimum (finite-difference MMFF Hessian, not counted).

Usage: python benchmark/mmff_proxy.py [set ...]   (sets from ~/Work/SEAMM/optimizer_benchmark/sets)
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seamm_optimizer import Optimizer  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SETS = Path.home() / "Work/SEAMM/optimizer_benchmark/sets"
KCAL_EV = 0.0433641043
EV_KJ = 96.48533212


def read_sdf(path):
    out = []
    for block in Path(path).read_text().split("$$$$"):
        lines = block.strip("\n").splitlines()
        if len(lines) < 4:
            continue
        n = int(lines[3][:3])
        atoms = [ln.split() for ln in lines[4 : 4 + n]]
        charge = 0
        for k, ln in enumerate(lines):
            if ln.startswith(">") and "net charge" in ln:
                charge = int(lines[k + 1])
        out.append((lines[0].strip(), [a[3] for a in atoms],
                    np.array([[float(v) for v in a[:3]] for a in atoms]), charge))
    return out


def mmff_engine(symbols, xyz, charge):
    """(engine, hessian) on MMFF94 for this structure, bonds perceived from xyz; None
    if MMFF cannot type it."""
    block = f"{len(symbols)}\n\n" + "\n".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, xyz)
    )
    mol = Chem.MolFromXYZBlock(block)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    calls = [0]

    def engine(x):
        calls[0] += 1
        flat = np.asarray(x, float).ravel().tolist()
        return ff.CalcEnergy(flat) * KCAL_EV, np.array(ff.CalcGrad(flat)).reshape(-1, 3) * KCAL_EV

    def hessian(x, h=1.0e-4):
        x0 = np.asarray(x, float).ravel()
        H = np.zeros((len(x0), len(x0)))
        for k in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            gp = np.array(ff.CalcGrad(xp.tolist()))
            gm = np.array(ff.CalcGrad(xm.tolist()))
            H[k] = (gp - gm) / (2 * h) * KCAL_EV
        return 0.5 * (H + H.T)

    return engine, hessian, calls


def ase_bfgs(engine, symbols, xyz, fmax=0.0015, steps=5000):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import BFGS

    class Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, g = engine(atoms.get_positions())
            self.results = {"energy": e, "forces": -g}

    atoms = Atoms(symbols, positions=xyz)
    atoms.calc = Calc()
    converged = BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    e, g = engine(atoms.get_positions())
    return atoms.get_positions(), e, converged


def main(names):
    rows = []
    for name in names:
        for key, symbols, xyz, charge in read_sdf(SETS / f"{name}.sdf"):
            built = mmff_engine(symbols, xyz, charge)
            if built is None:
                rows.append({"set": name, "key": key, "skipped": "MMFF cannot type"})
                continue
            engine, hessian, calls = built
            row = {"set": name, "key": key, "natoms": len(symbols)}
            for label, kw in (("lindh", {"hessian": "lindh"}), ("identity", {"hessian": "identity"})):
                calls[0] = 0
                t = time.time()
                r = Optimizer(engine, symbols, xyz, hessian_engine=hessian, **kw).run()
                row[label] = {"grads": r.n_gradients, "converged": r.converged, "E": r.energy,
                              "minimum": r.minimum, "escapes": len(r.escapes), "s": round(time.time() - t, 2)}
            calls[0] = 0
            _, e_ase, conv = ase_bfgs(engine, symbols, xyz)
            row["ase"] = {"grads": calls[0], "converged": bool(conv), "E": e_ase}
            rows.append(row)
            print(f"{name:10s} {key:14s} N={len(symbols):3d}  lindh {row['lindh']['grads']:4d}"
                  f"{'' if row['lindh']['minimum'] else '*'}  identity {row['identity']['grads']:4d}"
                  f"  ase {row['ase']['grads']:5d}  dE(lindh-ase) {(row['lindh']['E'] - e_ase) * EV_KJ:+.3f} kJ/mol",
                  flush=True)
    done = [r for r in rows if "lindh" in r]
    summary = {}
    for label in ("lindh", "identity", "ase"):
        grads = [r[label]["grads"] for r in done]
        summary[label] = {"n": len(done), "median": statistics.median(grads), "mean": round(statistics.mean(grads), 1),
                          "p90": sorted(grads)[int(0.9 * (len(grads) - 1))],
                          "converged": sum(r[label]["converged"] for r in done)}
    for label in ("lindh", "identity"):
        summary[label]["minimum"] = sum(bool(r[label]["minimum"]) for r in done)
        summary[label]["lower or equal E than ASE (0.05 kJ/mol)"] = sum(
            (r[label]["E"] - r["ase"]["E"]) * EV_KJ <= 0.05 for r in done)
    summary["skipped"] = [r["key"] for r in rows if "skipped" in r]
    out = Path(__file__).with_name("mmff_proxy_results.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["fragments", "breadth", "failures", "chno"])
