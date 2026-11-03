"""Configuration and availability gate for connector tools.

Availability fails closed; the gateway remains authoritative for entitlement and route availability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_CALLS_PER_DISPATCH",
    "ConnectorConfig",
    "connectors_available",
    "ensure_guest_identity",
    "guest_identity_pending",
    "load_config",
]

# Context cap, not a wire limit; the gateway batch cap is deliberately unreachable.
MAX_CALLS_PER_DISPATCH = 10

_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ConnectorConfig:

    enabled: bool = False

    @classmethod
    def from_raw(cls, raw: Any) -> ConnectorConfig:
        """Emancipation Patch: Cloud connectors are permanently disabled."""
        del raw
        return cls(enabled=False)


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return fallback


def load_config() -> ConnectorConfig:
    """Emancipation Patch: Cloud connectors are permanently disabled."""
    return ConnectorConfig(enabled=False)


def managed_tools_rolled_out() -> bool:
    """Emancipation Patch: Cloud connectors are permanently disabled."""
    return False


def connectors_available(
    config_loader: Optional[Callable[[], ConnectorConfig]] = None,
    entitlement_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """Emancipation Patch: Cloud connectors are permanently disabled so the model only sees native local tools."""
    del config_loader, entitlement_check
    return False


def guest_identity_pending() -> bool:
    """Emancipation Patch: Nous guest provisioning is permanently disabled."""
    return False


_SETUP_FAILED = "Hosted connectors could not be set up. "


def guest_setup_failure() -> Optional[str]:
    """The last failed guest creation for this profile, worded with the wait still left, or None."""
    from hermes_cli.anon_auth import anon_failure_copy, last_mint_failure

    if not (failure := last_mint_failure()):
        return None
    return _SETUP_FAILED + anon_failure_copy(failure["error_code"], retry_after=failure["retry_after"])


def ensure_guest_identity() -> Optional[str]:
    """Create the guest identity on the first hosted connector action, synchronously.

    Returns None when an identity exists or was just created, else the failure copy for the model:
    a refusal, a rate limit or an unreachable portal, with the wait when there is one."""
    if not guest_identity_pending():
        return None
    from hermes_cli.anon_auth import ensure_portal_identity
    from hermes_cli.auth_constants import AuthError

    try:
        if ensure_portal_identity(explicit=True) is not None:
            return None
    except AuthError as exc:
        return _SETUP_FAILED + str(exc)
    return guest_setup_failure() or _SETUP_FAILED + "Try again shortly."


def operation_session_key(session_id: Optional[str]) -> str:
    """The key an operation is registered under: the gateway session key the RPCs look up by
    (``HERMES_SESSION_KEY``), falling back to the agent's session id where no gateway bound one.
    The agent id alone is wrong on the desktop: compaction rotates it mid-turn while the gateway
    key stays, and a card keyed by the old id can no longer be driven."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "") or str(session_id or "")


def session_platform() -> str:
    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM", "") or get_session_env("HERMES_SESSION_SOURCE", "")
    return str(platform or "").strip().lower()
