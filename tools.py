"""
Compatibility wrappers for legacy event-registration helpers.

The canonical implementation lives in `event_registration.py`.
This module keeps the older function names/signatures available while delegating
to the canonical logic so the codebase no longer has two drifting
implementations.
"""

from __future__ import annotations

from typing import Any

from event_registration import (
    fetch_event_registrations_by_email,
    register_for_event,
)


def check_if_user_is_registered_for_event(
    event_registrations_collection,
    project_map,
    user_email: str,
    event_id: str,
) -> dict[str, Any]:
    """
    Compatibility wrapper around the canonical registration lookup logic.

    The collection and project arguments are retained for backward
    compatibility with older callers but are no longer used here.
    """
    registrations = fetch_event_registrations_by_email(user_email)
    is_registered = any(row.get("event_id") == event_id for row in registrations)
    if is_registered:
        return {
            "success": True,
            "message": "User is registered for this event",
            "event_id": event_id,
        }
    return {
        "success": True,
        "message": "User is not registered for this event",
        "event_id": event_id,
    }


def get_user_registered_events(
    event_registrations_collection,
    project_map,
    user_email: str,
) -> list[dict[str, Any]]:
    """
    Compatibility wrapper returning registrations in the normalized schema.

    The collection and project arguments are retained for backward
    compatibility with older callers but are no longer used here.
    """
    return fetch_event_registrations_by_email(user_email)


def register_user_for_event(
    event_registrations_collection,
    project_map,
    user_email: str,
    event_id: str,
) -> dict[str, Any]:
    """
    Compatibility wrapper around the canonical registration function.

    The collection and project arguments are retained for backward
    compatibility with older callers but are no longer used here.
    """
    return register_for_event(user_email, event_id)


__all__ = [
    "check_if_user_is_registered_for_event",
    "get_user_registered_events",
    "register_user_for_event",
]
