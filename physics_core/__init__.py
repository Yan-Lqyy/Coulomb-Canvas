# physics_core/__init__.py
from .constants import K_E, EPSILON_0
from .distributions import (
    ChargeDistribution,
    PointCharge,
    InfinitePlaneCharge,
    ChargedSphereShell,
    FiniteLineCharge, # Added new class here
    create_distribution_from_dict,
    DISTRIBUTION_CLASSES # Might be useful to expose
)
from .system import ChargeSystem

# Optional: Define __all__ for explicit exports
__all__ = [
    "K_E", "EPSILON_0",
    "ChargeDistribution", "PointCharge", "InfinitePlaneCharge", "ChargedSphereShell",
    "FiniteLineCharge", "create_distribution_from_dict", "DISTRIBUTION_CLASSES",
    "ChargeSystem"
]