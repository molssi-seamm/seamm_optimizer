#!/usr/bin/env python
"""A free proxy of the optimizer benchmark: the same starts on MMFF94 surfaces (RDKit).

Compares seamm_optimizer (Lindh model Hessian, and ASE's scaled identity to separate
the effect of the Hessian from the step) with ASE's BFGS, all stopping at the shared
0.0015 eV/Å. Reports gradients to converge, success, energy differences, and whether
the end point is a minimum (finite-difference MMFF Hessian, not counted).

Usage: python benchmark/mmff_proxy.py [set or file.sdf ...] [--variants=TRIC,TRIC-cheap] [--tag=name]
       [--perturb=Å]   (sets from ~/Work/SEAMM/optimizer_benchmark/sets)
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
    calls = [0]

    def _eg(x):
        # a force field built at x: RDKit's reused force fields gave inconsistent
        # gradients (CalcGrad(pos) is stateful; no-argument calls did not follow the
        # conformer), a fresh one matches finite differences
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i, p in enumerate(np.asarray(x, float).reshape(-1, 3)):
            conf.SetAtomPosition(i, p.tolist())
        ff = AllChem.MMFFGetMoleculeForceField(m, props)
        return ff.CalcEnergy(), np.array(ff.CalcGrad())

    def engine(x):
        calls[0] += 1
        e, g = _eg(x)
        return e * KCAL_EV, g.reshape(-1, 3) * KCAL_EV

    def hessian(x, h=1.0e-4):
        x0 = np.asarray(x, float).ravel()
        H = np.zeros((len(x0), len(x0)))
        for k in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            H[k] = (_eg(xp)[1] - _eg(xm)[1]) / (2 * h) * KCAL_EV
        return 0.5 * (H + H.T)

    return engine, hessian, calls


def uff_hessian_engine(symbols, xyz, charge):
    """A cheap level for the proxy: UFF's Hessian (central differences of UFF
    gradients, a fresh force field per geometry), or None if UFF cannot type it.
    MMFF94 plays the expensive level, UFF a cheaper, different one (as PM7 or GFN2-xTB
    against DFT)."""
    block = f"{len(symbols)}\n\n" + "\n".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, xyz)
    )
    mol = Chem.MolFromXYZBlock(block)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
        if not AllChem.UFFHasAllMoleculeParams(mol):
            return None
    except Exception:
        return None

    def gradient(x):
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i, p in enumerate(np.asarray(x, float).reshape(-1, 3)):
            conf.SetAtomPosition(i, p.tolist())
        ff = AllChem.UFFGetMoleculeForceField(m)
        return np.array(ff.CalcGrad()) * KCAL_EV

    def hessian(x, h=1.0e-4):
        x0 = np.asarray(x, float).ravel()
        H = np.zeros((len(x0), len(x0)))
        for k in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            H[k] = (gradient(xp) - gradient(xm)) / (2 * h)
        return 0.5 * (H + H.T)

    return hessian


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


def geometric_opt(engine, symbols, xyz, fmax=0.0015, maxiter=2000):
    """geomeTRIC (TRIC) with a custom engine, stopping on the max atomic force only."""
    import logging
    import os
    import tempfile

    import geometric.engine
    import geometric.optimize
    from geometric.molecule import Molecule

    bohr, hartree = 0.529177210903, 27.211386245988
    molecule = Molecule()
    molecule.elem = list(symbols)
    molecule.xyzs = [np.asarray(xyz, float)]

    class Engine(geometric.engine.Engine):
        def calc_new(self, coords, dirname):
            e, g = engine(coords.reshape(-1, 3) * bohr)
            return {"energy": e / hartree, "gradient": (g * bohr / hartree).ravel()}

        def copy_scratch(self, src, dest):
            pass

    import contextlib
    import io

    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        os.chdir(tmp)
        logging.disable(logging.CRITICAL)
        try:
            m = geometric.optimize.run_optimizer(
                customengine=Engine(molecule), input="opt.txt", coordsys="tric", maxiter=maxiter,
                convergence_energy=99.9, convergence_grms=99.9, convergence_drms=99.9,
                convergence_dmax=99.9, convergence_gmax=fmax * bohr / hartree,
            )
            converged = True
            final = np.array(m.xyzs[-1])
        except Exception:
            converged, final = False, None
        finally:
            logging.disable(logging.NOTSET)
            os.chdir(here)
    return final, converged


VARIANTS = {
    "TRIC": {"coordinates": "TRIC", "hessian": "lindh"},
    "cartesian": {"coordinates": "cartesian", "hessian": "lindh"},
    "cart-identity": {"coordinates": "cartesian", "hessian": "identity"},
    "TRIC-geoguess": {"coordinates": "TRIC", "hessian": "geometric"},
    # phase 3: the cheap level's Hessian (UFF here) at the start, refreshed adaptively
    "TRIC-cheap": {"coordinates": "TRIC", "hessian": "cheap", "cheap": True},
    "TRIC-cheap-norefresh": {"coordinates": "TRIC", "hessian": "cheap", "cheap": True, "refresh": "never"},
    "TRIC-cheap-every5": {"coordinates": "TRIC", "hessian": "cheap", "cheap": True, "refresh": 5},
    "TRIC-exact": {"coordinates": "TRIC", "hessian": "computed"},
}


def main(names, variants=("TRIC", "cartesian"), baselines=("ase", "geometric"), perturb=0.0, tag=""):
    rng = np.random.default_rng(20261124)
    rows = []
    for name in names:
        for key, symbols, xyz, charge in read_sdf(Path(name) if name.endswith(".sdf") else SETS / f"{name}.sdf"):
            built = mmff_engine(symbols, xyz, charge)
            if built is None:
                rows.append({"set": name, "key": key, "skipped": "MMFF cannot type"})
                continue
            engine, hessian, calls = built
            if perturb:
                xyz = xyz + perturb * rng.normal(size=xyz.shape)
            row = {"set": name, "key": key, "natoms": len(symbols)}
            cheap = uff_hessian_engine(symbols, xyz, charge)
            for label in variants:
                kw = dict(VARIANTS[label])
                if kw.pop("cheap", False):
                    if cheap is None:
                        continue
                    kw["cheap_hessian_engine"] = cheap
                calls[0] = 0
                t = time.time()
                r = Optimizer(engine, symbols, xyz, hessian_engine=hessian, **kw).run()
                row[label] = {"grads": r.n_gradients, "converged": r.converged, "E": r.energy,
                              "minimum": r.minimum, "escapes": len(r.escapes), "fallbacks": r.n_fallbacks,
                              "cheap hessians": r.n_cheap_hessians, "refreshes": r.n_refreshes,
                              "s": round(time.time() - t, 2)}
            if "ase" in baselines:
                calls[0] = 0
                _, e_ase, conv = ase_bfgs(engine, symbols, xyz)
                row["ase"] = {"grads": calls[0], "converged": bool(conv), "E": e_ase}
            if "geometric" in baselines:
                calls[0] = 0
                final, conv = geometric_opt(engine, symbols, xyz)
                n = calls[0]
                e_geo = engine(final)[0] if final is not None else None
                row["geometric"] = {"grads": n, "converged": bool(conv), "E": e_geo}
            rows.append(row)
            ref = min(v["E"] for k, v in row.items() if isinstance(v, dict) and v.get("E") is not None)
            print(f"{name:10s} {key:14s} N={len(symbols):3d}  " + "  ".join(
                f"{k} {v['grads']:4d}{'' if v.get('minimum', True) else '*'}{'' if v['converged'] else '!'}"
                f"({(v['E'] - ref) * EV_KJ:+.2f})" for k, v in row.items() if isinstance(v, dict)), flush=True)
    done = [r for r in rows if variants[0] in r]
    labels = list(variants) + [b for b in baselines]
    summary = {"perturb": perturb}
    for label in labels:
        rs = [r for r in done if label in r]
        grads = sorted(r[label]["grads"] for r in rs)
        summary[label] = {"n": len(rs), "median": statistics.median(grads), "mean": round(statistics.mean(grads), 1),
                          "p90": grads[int(0.9 * (len(grads) - 1))], "converged": sum(r[label]["converged"] for r in rs)}
        best = [min(v["E"] for k, v in r.items() if isinstance(v, dict) and v.get("E") is not None) for r in rs]
        summary[label]["within 0.05 kJ/mol of the best"] = sum(
            r[label]["E"] is not None and (r[label]["E"] - b) * EV_KJ <= 0.05 for r, b in zip(rs, best))
        if label in VARIANTS:
            summary[label]["minimum"] = sum(bool(r[label]["minimum"]) for r in rs)
            summary[label]["fallbacks"] = sum(r[label]["fallbacks"] for r in rs)
            summary[label]["cheap hessians"] = sum(r[label].get("cheap hessians", 0) for r in rs)
        for s_name in names:
            sub = sorted(r[label]["grads"] for r in rs if r["set"] == s_name)
            if sub:
                summary[label][f"{s_name} mean/median"] = (round(statistics.mean(sub), 1), statistics.median(sub))
    summary["skipped"] = [r["key"] for r in rows if "skipped" in r]
    tag = (f"_{tag}" if tag else "") + (f"_p{perturb:g}" if perturb else "")
    out = Path(__file__).with_name(f"mmff_proxy_results{tag}.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    perturb = float(next((a.split("=")[1] for a in args if a.startswith("--perturb=")), 0.0))
    names = [a for a in args if not a.startswith("--")] or ["fragments", "breadth", "failures", "chno"]
    tag = next((a.split("=")[1] for a in args if a.startswith("--tag=")), "")
    variants = next((a.split("=")[1].split(",") for a in args if a.startswith("--variants=")), ("TRIC", "cartesian"))
    main(names, variants=variants, perturb=perturb, tag=tag)
