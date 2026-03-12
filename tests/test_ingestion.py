from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ingestion import (
    FirebaseCMSExtractor,
    normalize_project_record,
    write_records_json,
    write_records_jsonl,
)


def test_normalize_project_record_serializes_expected_fields():
    record = {
        "_firestore_doc_id": "doc-1",
        "id": "event-1",
        "name": "Community Meetup",
        "shortDescription": "Short summary",
        "description": "Longer body",
        "venue": "Main Hall",
        "startDate": datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
        "endDate": datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        "projectType": "CONFERENCE",
        "status": "PUBLISHED",
        "parentProjectId": "org-1",
        "domains": ["community", "events"],
    }

    normalized = normalize_project_record(record, source_collection="projects")

    assert normalized.id == "event-1"
    assert normalized.source_doc_id == "doc-1"
    assert normalized.name == "Community Meetup"
    assert normalized.startDate.endswith("+00:00")
    assert normalized.endDate.endswith("+00:00")
    assert normalized.domains == ["community", "events"]


def test_write_records_jsonl_and_json(tmp_path):
    records = [
        normalize_project_record(
            {
                "_firestore_doc_id": "doc-1",
                "id": "event-1",
                "description": "Hello world",
            },
            source_collection="projects",
        )
    ]

    jsonl_path = tmp_path / "community.jsonl"
    json_path = tmp_path / "community.json"
    write_records_jsonl(records, jsonl_path)
    write_records_json(records, json_path)

    assert jsonl_path.exists()
    assert json_path.exists()
    assert '"id": "event-1"' in json_path.read_text(encoding="utf-8")


def test_firebase_cms_extractor_fetches_normalized_records():
    snap = MagicMock()
    snap.id = "doc-1"
    snap.to_dict.return_value = {
        "id": "event-1",
        "name": "Community Meetup",
        "description": "Body",
        "projectType": "CONFERENCE",
    }

    collection = MagicMock()
    collection.stream.return_value = iter([snap])
    db = MagicMock()
    db.collection.return_value = collection

    extractor = FirebaseCMSExtractor(collection_name="projects")
    with patch("ingestion.firebase_cms.get_firestore_client", return_value=db):
        records = extractor.fetch_normalized_records(project_types=["CONFERENCE"])

    assert len(records) == 1
    assert records[0].id == "event-1"
    assert records[0].source_doc_id == "doc-1"
