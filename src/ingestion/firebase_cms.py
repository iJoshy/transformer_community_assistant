from __future__ import annotations

import json
import os
from typing import Any, Iterable

import firebase_admin
from firebase_admin import credentials, firestore

from .normalize import normalize_project_records
from .schema import NormalizedCommunityRecord

load_dotenv = None
try:
    from dotenv import load_dotenv as _load_dotenv

    load_dotenv = _load_dotenv
except ImportError:
    pass


def _ensure_dotenv() -> None:
    if load_dotenv:
        load_dotenv(override=True)


def ensure_firebase_initialized() -> None:
    if firebase_admin._apps:
        return
    _ensure_dotenv()
    config_raw = os.getenv("FIREBASE_CONFIG_JSON")
    if not config_raw:
        raise RuntimeError("FIREBASE_CONFIG_JSON is not set")
    config = json.loads(config_raw)
    cred = credentials.Certificate(config)
    firebase_admin.initialize_app(cred)


def get_firestore_client() -> firestore.Client:
    ensure_firebase_initialized()
    return firestore.client()


class FirebaseCMSExtractor:
    def __init__(self, collection_name: str | None = None):
        self.collection_name = (
            collection_name
            or os.getenv("CMS_COLLECTION")
            or os.getenv("EVENTS_COLLECTION", "projects")
        )

    def fetch_raw_records(self) -> list[dict[str, Any]]:
        db = get_firestore_client()
        rows: list[dict[str, Any]] = []
        for snap in db.collection(self.collection_name).stream():
            payload = snap.to_dict() or {}
            payload["_firestore_doc_id"] = snap.id
            rows.append(payload)
        return rows

    def fetch_normalized_records(
        self,
        *,
        project_types: Iterable[str] | None = None,
    ) -> list[NormalizedCommunityRecord]:
        raw_records = self.fetch_raw_records()
        allowed_types = {
            item.strip()
            for item in (project_types or [])
            if str(item).strip()
        }
        if allowed_types:
            raw_records = [
                record
                for record in raw_records
                if record.get("projectType") in allowed_types
            ]
        return normalize_project_records(
            raw_records,
            source_collection=self.collection_name,
        )
