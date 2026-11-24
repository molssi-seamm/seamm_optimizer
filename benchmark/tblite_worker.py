#!/usr/bin/env python
"""A tblite worker for the xTB proxy: JSON requests on stdin, JSON replies on stdout.

Run with a Python that has tblite (e.g. the seamm-tblite conda environment), so the
proxy needs no tblite in its own environment. Requests:

    {"op": "gradient", "method": "GFN2-xTB", "numbers": [...], "charge": 0, "uhf": 0,
     "xyz": [[x, y, z], ...]}                       (Å)
    {"op": "hessian", ..., "step": 0.005}          (central differences, Å)

Replies: {"energy": E_h, "gradient": [[...]] (E_h/bohr)} or {"hessian": 3N x 3N
(E_h/bohr^2)}, or {"error": "..."}.
"""

import json
import sys

import numpy as np
from tblite.interface import Calculator

BOHR = 0.529177210903


def gradient(request, xyz):
    calc = Calculator(
        request["method"],
        np.array(request["numbers"]),
        np.asarray(xyz, float) / BOHR,
        charge=float(request.get("charge", 0)),
        uhf=int(request.get("uhf", 0)),
    )
    calc.set("verbosity", 0)
    calc.set("max-iter", 500)
    if "accuracy" in request:
        calc.set("accuracy", float(request["accuracy"]))
    result = calc.singlepoint()
    return float(result.get("energy")), np.array(result.get("gradient"))


def main():
    for line in sys.stdin:
        request = json.loads(line)
        try:
            xyz = np.asarray(request["xyz"], float)
            if request["op"] == "gradient":
                e, g = gradient(request, xyz)
                reply = {"energy": e, "gradient": g.tolist()}
            elif request["op"] == "hessian":
                h = float(request.get("step", 0.005))
                x0 = xyz.ravel()
                n = len(x0)
                H = np.zeros((n, n))
                for k in range(n):
                    xp, xm = x0.copy(), x0.copy()
                    xp[k] += h
                    xm[k] -= h
                    gp = gradient(request, xp.reshape(-1, 3))[1].ravel()
                    gm = gradient(request, xm.reshape(-1, 3))[1].ravel()
                    H[k] = (gp - gm) / (2 * h / BOHR)
                reply = {"hessian": (0.5 * (H + H.T)).tolist()}
            else:
                reply = {"error": f"unknown op {request['op']!r}"}
        except Exception as e:  # noqa: BLE001 - report every failure to the caller
            reply = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
