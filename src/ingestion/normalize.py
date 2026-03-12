from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence

from .schema import NormalizedCommunityRecord


def normalize_project_record(
    record: dict[str, Any],
    *,
    source_collection: str,
    fallback_source_doc_id: str = "",
) -> NormalizedCommunityRecord:
    if not isinstance(record, dict):
        raise TypeError(f"Expected a dict record, received {type(record)}")

    source_doc_id = (
        _coalesce_text(
            record.get("_firestore_doc_id"),
            record.get("source_doc_id"),
            fallback_source_doc_id,
        )
        or _coalesce_text(record.get("id"))
        or "unknown-record"
    )

    normalized_id = _coalesce_text(record.get("id"), source_doc_id)
    return NormalizedCommunityRecord(
        id=normalized_id,
        source_doc_id=source_doc_id,
        source_collection=source_collection,
        name=_coalesce_text(record.get("name"), record.get("title")),
        shortDescription=_coalesce_text(
            record.get("shortDescription"),
            record.get("summary"),
        ),
        description=_coalesce_text(
            record.get("description"),
            record.get("content"),
            record.get("details"),
            record.get("shortDescription"),
        ),
        venue=_coalesce_text(record.get("venue"), record.get("location")),
        startDate=_stringify_value(record.get("startDate")),
        endDate=_stringify_value(record.get("endDate")),
        projectType=_coalesce_text(record.get("projectType"), record.get("type")),
        status=_coalesce_text(record.get("status")),
        parentProjectId=_coalesce_text(record.get("parentProjectId")),
        createdAt=_stringify_value(record.get("createdAt")),
        updatedAt=_stringify_value(record.get("updatedAt")),
        domains=_normalize_domains(record.get("domains")),
    )


def normalize_project_records(
    records: Sequence[dict[str, Any]],
    *,
    source_collection: str,
) -> list[NormalizedCommunityRecord]:
    normalized: list[NormalizedCommunityRecord] = []
    for index, record in enumerate(records):
        normalized.append(
            normalize_project_record(
                record,
                source_collection=source_collection,
                fallback_source_doc_id=f"record-{index}",
            )
        )
    return normalized


def _normalize_domains(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _coalesce_text(value)
    return [text] if text else []


def _coalesce_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value).strip()
