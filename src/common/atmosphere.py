"""ISA-style atmosphere model used in Term Project.pdf.

Density formula given in the project:
    rho(h) = rho0 * (1 - 2.2257e-5 * h)^4.2586

Valid for the troposphere (h up to ~11 km), which covers all three
flights. The formula is implemented with a math facade (numpy or
casadi) so the same function works inside CasADi NLPs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .constants import ATM_EXP, ATM_LAPSE, RHO0


def rho(h: Any, math: Any = np) -> Any:
    """Air density at altitude h.

    Args:
        h: Altitude [m]. Scalar, numpy array, or CasADi symbol.
        math: Math backend (default numpy). Pass `casadi` when building
            symbolic expressions for an NLP.

    Returns:
        Air density [kg/m^3], same shape/type as `h`.
    """
    return RHO0 * (1.0 - ATM_LAPSE * h) ** ATM_EXP
