#!/usr/bin/env python
"""A GFN2-xTB proxy of the optimizer benchmark, with realistic cheap levels.

GFN2-xTB plays the expensive level. The cheap Hessians come from levels related to it
the way GFN2-xTB or a force field relates to DFT: MMFF94 (RDKit) and GFN1-xTB. The MMFF
proxy cannot show what a cheap Hessian is worth, because UFF's curvature is a poor
guide to MMFF's (Design 2026-11-24). Gradients are counted to the shared max force
0.0015 eV/Å; Hessians (cheap, and the exact one used to check for a minimum) are
not counted as gradients.

tblite runs in a worker process with its own Python (default: the seamm-tblite conda
environment), so this environment needs no tblite:

    python benchmark/xtb_proxy.py file.sdf [file.sdf ...] [--variants=TRIC,TRIC-mmff]
        [--baselines=ase,geometric] [--tag=name] [--tblite-python=/path/to/python]
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mmff_proxy as mp  # noqa: E402
from seamm_optimizer import Optimizer, SurrogateOptimizer  # noqa: E402

HARTREE = 27.211386245988  # eV
BOHR = 0.529177210903  # Å
EV_KJ = 96.48533212
TBLITE_PYTHON = Path.home() / "miniconda3/envs/seamm-tblite/bin/python"
ACCURACY = None  # tblite's accuracy (default 1.0; smaller is tighter)
#: how the end point is checked: "exact" (GFN2 Hessian), "cheap" (GFN1 Hessian, its
#: imaginary modes confirmed from GFN2 gradients, as in production) or "none"
CHECK = "exact"


class Tblite:
    """A persistent tblite worker process."""

    def __init__(self, python=TBLITE_PYTHON):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.process = subprocess.Popen(
            [str(python), str(Path(__file__).with_name("tblite_worker.py"))],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def request(self, **kwargs):
        self.process.stdin.write(json.dumps(kwargs) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def levels(tb, symbols, charge):
    """The GFN2-xTB engine (eV, eV/Å, counted), its Hessian, and GFN1's (eV/Å^2)."""
    table = Chem.GetPeriodicTable()
    numbers = [table.GetAtomicNumber(s) for s in symbols]
    uhf = (sum(numbers) - int(charge)) % 2
    common = {"numbers": numbers, "charge": int(charge), "uhf": uhf}
    if ACCURACY is not None:
        common["accuracy"] = ACCURACY
    calls = [0]

    def engine(x):
        calls[0] += 1
        r = tb.request(op="gradient", method="GFN2-xTB", xyz=np.asarray(x).tolist(), **common)
        return r["energy"] * HARTREE, np.array(r["gradient"]) * HARTREE / BOHR

    def gradient_of(method):
        def gradient(x):
            r = tb.request(op="gradient", method=method, xyz=np.asarray(x).tolist(), **common)
            return r["energy"] * HARTREE, np.array(r["gradient"]) * HARTREE / BOHR

        return gradient

    def hessian_of(method):
        def hessian(x):
            r = tb.request(op="hessian", method=method, xyz=np.asarray(x).tolist(), **common)
            return np.array(r["hessian"]) * HARTREE / BOHR**2

        return hessian

    levels.gfn1_gradient = gradient_of("GFN1-xTB")
    return engine, hessian_of("GFN2-xTB"), hessian_of("GFN1-xTB"), calls


VARIANTS = {
    "TRIC": {"coordinates": "TRIC", "hessian": "lindh"},
    "TRIC-geoguess": {"coordinates": "TRIC", "hessian": "geometric"},
    "TRIC-mmff": {"coordinates": "TRIC", "hessian": "cheap", "cheap": "mmff"},
    "TRIC-gfn1": {"coordinates": "TRIC", "hessian": "cheap", "cheap": "gfn1"},
    "TRIC-exact": {"coordinates": "TRIC", "hessian": "computed"},
    # phase 4: the cheap level carries the steps (SurrogateOptimizer)
    "SUR-gfn1": {"surrogate": "gfn1", "variant": "delta-hessian"},
    "SUR-gfn1-dg": {"surrogate": "gfn1", "variant": "delta-gradient"},
    "SUR-mmff": {"surrogate": "mmff", "variant": "delta-hessian"},
}


def checking(hessian, gfn1):
    """The minimum-check arguments for CHECK."""
    if CHECK == "none":
        return {"check_minimum": False}
    if CHECK == "cheap":
        return {"cheap_hessian_engine": gfn1}
    return {"hessian_engine": hessian}


def main(paths, variants=("TRIC", "TRIC-mmff", "TRIC-gfn1", "TRIC-exact"),
         baselines=("ase", "geometric"), tag="", python=TBLITE_PYTHON):
    tb = Tblite(python)
    rows = []
    try:
        for path in paths:
            for key, symbols, xyz, charge in mp.read_sdf(path):
                engine, hessian, gfn1, calls = levels(tb, symbols, charge)
                mmff = mp.mmff_engine(symbols, xyz, charge)
                row = {"set": Path(path).stem, "key": key, "natoms": len(symbols)}
                for label in variants:
                    kw = dict(VARIANTS[label])
                    if "surrogate" in kw:
                        which = kw.pop("surrogate")
                        if which == "mmff":
                            if mmff is None:
                                continue
                            sur = mmff[0]
                        else:
                            sur = levels.gfn1_gradient
                        calls[0] = 0
                        t = time.time()
                        try:
                            r = SurrogateOptimizer(engine, sur, symbols, xyz, **checking(hessian, gfn1), **kw).run()
                        except Exception as e:  # noqa: BLE001 - record and go on
                            row[label] = {"grads": calls[0], "converged": False, "E": None, "error": str(e)}
                            continue
                        row[label] = {"grads": r.n_gradients, "cheap grads": r.n_cheap_gradients,
                                      "converged": r.converged, "E": r.energy, "minimum": r.minimum,
                                      "lowest": r.lowest_frequency, "rejected": r.n_rejected,
                                      "escapes": 0, "fallbacks": 0, "s": round(time.time() - t, 1)}
                        continue
                    cheap = kw.pop("cheap", None)
                    if cheap == "mmff":
                        if mmff is None:
                            continue
                        kw["cheap_hessian_engine"] = mmff[1]
                    elif cheap == "gfn1":
                        kw["cheap_hessian_engine"] = gfn1
                    calls[0] = 0
                    t = time.time()
                    try:
                        if CHECK == "cheap" and kw.get("hessian") != "cheap":
                            kw.setdefault("refresh", "never")  # the GFN1 engine only checks
                        r = Optimizer(engine, symbols, xyz, **{**checking(hessian, gfn1), **kw}).run()
                    except Exception as e:  # noqa: BLE001 - record and go on
                        row[label] = {"grads": calls[0], "converged": False, "E": None, "error": str(e)}
                        continue
                    row[label] = {"grads": r.n_gradients, "converged": r.converged, "E": r.energy,
                                  "minimum": r.minimum, "lowest": r.lowest_frequency,
                                  "escapes": len(r.escapes), "fallbacks": r.n_fallbacks,
                                  "cheap hessians": r.n_cheap_hessians, "refreshes": r.n_refreshes,
                                  "s": round(time.time() - t, 1)}
                if "ase" in baselines:
                    calls[0] = 0
                    _, e_ase, conv = mp.ase_bfgs(engine, symbols, xyz, steps=500)
                    row["ase"] = {"grads": calls[0], "converged": bool(conv), "E": e_ase}
                if "geometric" in baselines:
                    calls[0] = 0
                    final, conv = mp.geometric_opt(engine, symbols, xyz, maxiter=500)
                    n = calls[0]
                    row["geometric"] = {"grads": n, "converged": bool(conv),
                                        "E": engine(final)[0] if final is not None else None}
                rows.append(row)
                done = {k: v for k, v in row.items() if isinstance(v, dict)}
                ref = min((v["E"] for v in done.values() if v.get("E") is not None), default=0.0)
                print(f"{key:14s} N={len(symbols):3d} q={charge:+d}  " + "  ".join(
                    f"{k} {v['grads']:4d}{'' if v.get('minimum', True) else '*'}"
                    f"{'' if v['converged'] else '!'}"
                    f"({(v['E'] - ref) * EV_KJ:+.2f})" if v.get("E") is not None else f"{k} ERR"
                    for k, v in done.items()), flush=True)
    finally:
        tb.close()
    labels = list(variants) + list(baselines)
    summary = {}
    for label in labels:
        rs = [r for r in rows if label in r and r[label].get("E") is not None]
        if not rs:
            continue
        grads = sorted(r[label]["grads"] for r in rs)
        best = [min(v["E"] for v in r.values() if isinstance(v, dict) and v.get("E") is not None) for r in rs]
        summary[label] = {
            "n": len(rs), "median": statistics.median(grads), "mean": round(statistics.mean(grads), 1),
            "p90": grads[int(0.9 * (len(grads) - 1))],
            "converged": sum(r[label]["converged"] for r in rs),
            "within 0.05 kJ/mol of the best": sum((r[label]["E"] - b) * EV_KJ <= 0.05 for r, b in zip(rs, best)),
        }
        if label in VARIANTS:
            summary[label]["minimum"] = sum(bool(r[label]["minimum"]) for r in rs)
            summary[label]["fallbacks"] = sum(r[label]["fallbacks"] for r in rs)
    out = Path(__file__).with_name(f"xtb_proxy_results{'_' + tag if tag else ''}.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]

    def option(name, default):
        return next((a.split("=", 1)[1] for a in args if a.startswith(f"--{name}=")), default)

    paths = [a for a in args if not a.startswith("--")]
    variants = option("variants", "TRIC,TRIC-mmff,TRIC-gfn1,TRIC-exact").split(",")
    baselines = [b for b in option("baselines", "ase,geometric").split(",") if b]
    CHECK = option("check", "exact")
    main(paths, variants, baselines, option("tag", ""), option("tblite-python", TBLITE_PYTHON))
