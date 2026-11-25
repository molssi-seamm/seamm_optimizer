#!/usr/bin/env python
"""Optimizers on a stratified sample of Paul's GDB hydrocarbons, PM7 as the expensive level.

The sample (fixed seed) is drawn from GDB_HC_PM7_new.csv by the 'Best' column, the
way the best PM7 structure was found (Emin, or simulated annealing to 300-8000 K):
a difficulty label. The start is built as the CBH step builds one: the lowest MMFF
structure of 20 RDKit conformers of the IsomericSMILES. Each optimizer runs from that
same start on PM7 (MOPAC, PRECISE), with GFN2-xTB (tblite, in a worker) as the cheap
level; the end ΔHf is compared with the file's Ebest. Results stream to a JSON-lines
file, one molecule per line, so a run can be resumed or read while it goes.

    python benchmark/gdb_proxy.py --shard=0/4 [--per-stratum=...] [--out=gdb_results]
        [--tblite-python=...] [--mopac=...]
"""

import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mmff_proxy as mp  # noqa: E402
import xtb_proxy as xp  # noqa: E402
from seamm_optimizer import Optimizer, SurrogateOptimizer  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CSV = Path.home() / "Dropbox/Science/Thermochemistry/GDB_HC_PM7_new.csv"
KCAL = 0.0433641043  # eV
PER_STRATUM = {"Emin": 40, "SA_300.0": 30, "SA_500.0": 30, "SA_1000.0": 30, "SA_2000.0": 40,
               "SA_4000.0": 50, "SA_8000.0": 50, "": 30}
MOPAC = Path.home() / "miniforge3/envs/seamm-mopac/bin/mopac"


def sample(per_stratum, seed=20261124):
    """The stratified sample: reservoir-sample each 'Best' stratum while streaming."""
    rng = random.Random(seed)
    pools, seen = {k: [] for k in per_stratum}, {k: 0 for k in per_stratum}
    with open(CSV) as fd:
        for row in csv.DictReader(fd):
            k = row["Best"]
            if k not in per_stratum:
                continue
            seen[k] += 1
            n = per_stratum[k]
            if len(pools[k]) < n:
                pools[k].append(row)
            else:
                j = rng.randrange(seen[k])
                if j < n:
                    pools[k][j] = row
    out = []
    for k in per_stratum:
        out += sorted(pools[k], key=lambda r: int(r["ID"]))
    return out


def start(smiles, seed=11):
    """The lowest MMFF structure of 20 conformers (as the CBH step starts), else UFF."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=seed))
    if not ids:
        return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
    best = min(range(len(ids)), key=lambda i: res[i][1])
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return symbols, np.array(mol.GetConformer(ids[best]).GetPositions())


def pm7_engine(symbols, workdir, mopac=MOPAC):
    calls = [0]
    env = dict(os.environ, OMP_NUM_THREADS="1")

    def engine(x):
        calls[0] += 1
        x = np.asarray(x, float).reshape(-1, 3)
        (workdir / "m.mop").write_text(
            "PM7 1SCF GRADIENTS AUX(PRECISION=9) PRECISE CHARGE=0\n\n\n"
            + "".join(f"{s} {a:.10f} 1 {b:.10f} 1 {c:.10f} 1\n" for s, (a, b, c) in zip(symbols, x))
        )
        for f in ("m.aux", "m.out"):
            (workdir / f).unlink(missing_ok=True)
        subprocess.run([str(mopac), "m.mop"], cwd=workdir, capture_output=True, env=env)
        aux = (workdir / "m.aux").read_text()
        e = float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([-+0-9.EeDd]+)", aux).group(1).replace("D", "E"))
        block = re.search(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)(?:\n\s*[A-Z_]+[:\[=]|\Z)", aux, re.S).group(1)
        g = np.array([float(v.replace("D", "E")) for v in block.split()]).reshape(-1, 3)
        return e * KCAL, g * KCAL

    return engine, calls


def run_one(row, tb, workdir, mopac):
    built = start(row["IsomericSMILES"])
    rec = {"ID": row["ID"], "smiles": row["IsomericSMILES"], "Best": row["Best"],
           "Ebest": float(row["Ebest"]) if row["Ebest"] else None}
    if built is None:
        rec["skipped"] = "no conformer"
        return rec
    symbols, xyz = built
    rec["natoms"] = len(symbols)
    engine, calls = pm7_engine(symbols, workdir, mopac)
    gfn2_e, gfn2_hess, _, _ = xp.levels(tb, symbols, 0)
    gfn2_grad = xp.levels.gfn2_gradient
    rec["start ΔHf"] = engine(xyz)[0] / KCAL
    variants = {
        "SUR-gfn2": lambda: SurrogateOptimizer(engine, gfn2_grad, symbols, xyz, cheap_hessian_engine=gfn2_hess).run(),
        "TRIC-gfn2": lambda: Optimizer(engine, symbols, xyz, hessian="cheap", cheap_hessian_engine=gfn2_hess).run(),
        "TRIC": lambda: Optimizer(engine, symbols, xyz, hessian="lindh", cheap_hessian_engine=gfn2_hess,
                                  refresh="never").run(),
    }
    only = getattr(run_one, "only", None)
    for name, make in variants.items():
        if only and name not in only:
            continue
        calls[0] = 0
        t = time.time()
        try:
            r = make()
            rec[name] = {"grads": r.n_gradients, "cheap grads": getattr(r, "n_cheap_gradients", 0),
                         "cheap hessians": getattr(r, "n_cheap_hessians", 0), "converged": bool(r.converged),
                         "minimum": r.minimum, "lowest": r.lowest_frequency, "ΔHf": r.energy / KCAL,
                         "max force": r.max_force, "escapes": len(r.escapes), "s": round(time.time() - t, 1)}
        except Exception as e:  # noqa: BLE001 - record and go on
            rec[name] = {"error": f"{type(e).__name__}: {e}"[:300], "grads": calls[0]}
    if getattr(run_one, "only", None) and "geometric" not in run_one.only:
        return rec
    calls[0] = 0
    t = time.time()
    try:
        final, conv = mp.geometric_opt(engine, symbols, xyz, maxiter=1000)
        n = calls[0]
        rec["geometric"] = {"grads": n, "converged": bool(conv),
                            "ΔHf": engine(final)[0] / KCAL if final is not None else None,
                            "s": round(time.time() - t, 1)}
    except Exception as e:  # noqa: BLE001
        rec["geometric"] = {"error": f"{type(e).__name__}: {e}"[:300], "grads": calls[0]}
    return rec


def main(args):
    def option(name, default):
        return next((a.split("=", 1)[1] for a in args if a.startswith(f"--{name}=")), default)

    i, n = (int(v) for v in option("shard", "0/1").split("/"))
    out = Path(option("out", "gdb_results")) / f"shard{i}of{n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    per = json.loads(option("per-stratum", json.dumps(PER_STRATUM)))
    seed = int(option("seed", "20261124"))
    only = option("variants", "")
    run_one.only = set(only.split(",")) if only else None
    mopac = Path(option("mopac", str(MOPAC)))
    rows = sample(per, seed)[i::n]
    done = set()
    if out.exists():
        done = {json.loads(line)["ID"] for line in out.read_text().splitlines() if line.strip()}
    tb = xp.Tblite(option("tblite-python", str(xp.TBLITE_PYTHON)))
    work = Path(tempfile.mkdtemp(prefix=f"gdb{i}_"))
    try:
        with open(out, "a") as fd:
            for row in rows:
                if row["ID"] in done:
                    continue
                rec = run_one(row, tb, work, mopac)
                fd.write(json.dumps(rec, default=float) + "\n")
                fd.flush()
                print(row["ID"], row["Best"], rec.get("natoms"),
                      {k: v.get("grads") for k, v in rec.items() if isinstance(v, dict)}, flush=True)
    finally:
        tb.close()


if __name__ == "__main__":
    main(sys.argv[1:])
