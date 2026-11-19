"""Agent Recovery Bench (ARB): failure detection and recovery evaluation for tool-using agents.

ARB extends failure *description* (taxonomy, telemetry) into failure *management*:
detection timing, recovery policy quality, and cost-aware outcomes, all verified
against deterministic environment state rather than agent self-reports.
"""

from arb.taxonomy import (
    FailureType,
    RecoveryAction,
    FAILURE_SPECS,
    FailureSpec,
    RECOVERY_ACTIONS,
    canonical_action,
    is_recoverable,
)
from arb import schema  # noqa: F401  (re-export convenience)

__version__ = "1.0.0"

__all__ = [
    "FailureType",
    "RecoveryAction",
    "FAILURE_SPECS",
    "FailureSpec",
    "RECOVERY_ACTIONS",
    "canonical_action",
    "is_recoverable",
    "schema",
    "__version__",
]
