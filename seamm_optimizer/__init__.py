# -*- coding: utf-8 -*-

"""seamm_optimizer: robust geometry optimization for SEAMM.

Phase 1 (proposal OPTIMIZER_DESIGN_PROPOSAL_20261124): quasi-Newton minimization in
Cartesians with an RFO / trust-region step, Lindh's model Hessian or a computed one,
BFGS updates, and a minimum check with an escape from saddle points.
"""

from .geometry import frequencies, projector, rigid_modes  # noqa: F401
from .model_hessian import lindh_hessian  # noqa: F401
from .optimizer import Optimizer, Result  # noqa: F401
from .steps import bfgs_update, rfo_step  # noqa: F401

# Handle versioneer
from ._version import __version__  # noqa: F401
