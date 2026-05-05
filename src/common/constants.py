"""Physical and unit-conversion constants used across the project.

All units are SI unless noted. Constants live here so other modules
never re-define them with slightly-different precision.
"""
from __future__ import annotations

# Earth and gravity
G0: float = 9.80665           # standard gravity [m/s^2]
R_EARTH: float = 6_371_000.0  # mean Earth radius [m]

# ISA sea-level
RHO0: float = 1.225           # sea-level air density [kg/m^3]
T0: float = 288.15            # sea-level temperature [K]
P0: float = 101_325.0         # sea-level pressure [Pa]

# Troposphere lapse used by the project's atmosphere model
ATM_LAPSE: float = 2.2257e-5  # [1/m]
ATM_EXP: float = 4.2586       # dimensionless exponent

# Unit conversions
DEG_TO_RAD: float = 0.017453292519943295
RAD_TO_DEG: float = 57.29577951308232
M_TO_FT: float = 3.28084      # meters -> feet
FT_TO_M: float = 1.0 / M_TO_FT
MPS_TO_KT: float = 1.9438444924406046  # m/s -> knots
KT_TO_MPS: float = 1.0 / MPS_TO_KT
