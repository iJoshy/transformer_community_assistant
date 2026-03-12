from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model_provider import build_embeddings


@dataclass(frozen=True)
class RagConfig:
    persist_dir: str = "vector_db"
    embedding_model: str | None = None
    chunk_size: int = 500
    chunk_overlap: int = 200


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]

    @property
    def source_id(self) -> str:
        return str(
            self.metadata.get("source_id")
            or self.metadata.get("source_doc_id")
            or ""
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": dict(self.metadata),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    k: int
    context: str
    chunks: tuple[RetrievedChunk, ...]

    @property
    def source_ids(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for chunk in self.chunks:
            source_id = chunk.source_id
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            ordered.append(source_id)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "context": self.context,
            "source_ids": self.source_ids,
            "documents": [chunk.to_dict() for chunk in self.chunks],
        }


RETRIEVAL_METADATA_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("id", "source_id"),
    ("source_doc_id", "source_doc_id"),
    ("source_collection", "source_collection"),
    ("source_system", "source_system"),
    ("name", "record_name"),
    ("projectType", "project_type"),
    ("status", "status"),
    ("venue", "venue"),
    ("startDate", "start_date"),
    ("endDate", "end_date"),
    ("parentProjectId", "parent_project_id"),
    ("shortDescription", "short_description"),
    ("domains", "domains"),
)


def load_records(path: str | Path) -> list:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    if raw.lstrip().startswith("["):
        return json.loads(raw)

    records = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def records_to_documents(
    records: Sequence,
    *,
    text_key: str = "description",
    id_key: Optional[str] = "id",
    metadata_keys: Optional[Sequence[str]] = None,
    formatter: Optional[Callable[[dict], str]] = None,
) -> List[Document]:
    documents: List[Document] = []
    for idx, record in enumerate(records):
        if isinstance(record, str):
            text = record
            metadata = {}
        elif isinstance(record, dict):
            if formatter is not None:
                text = formatter(record)
            else:
                text = record.get(text_key)
                if text is None:
                    raise ValueError(f"Record {idx} missing text key '{text_key}'.")
            metadata = _build_metadata(
                record,
                metadata_keys=metadata_keys,
                id_key=id_key,
            )
        else:
            raise TypeError(f"Unsupported record type at {idx}: {type(record)}")

        documents.append(Document(page_content=str(text), metadata=metadata))

    return documents


def normalized_records_to_documents(
    records: Sequence[Any],
    *,
    formatter: Optional[Callable[[dict[str, Any]], str]] = None,
    extra_metadata_keys: Optional[Sequence[str]] = None,
) -> List[Document]:
    documents: List[Document] = []
    formatter = formatter or format_event_page_content

    for idx, record in enumerate(records):
        record_dict = _coerce_record_to_dict(record, idx=idx)
        text = formatter(record_dict)
        metadata = build_retrieval_metadata(
            record_dict,
            extra_metadata_keys=extra_metadata_keys,
        )
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def format_event_page_content(data: dict) -> str:
    domains = _normalize_metadata_value(data.get("domains"))
    lines = [
        f"Event: {data.get('name', '') or ''}",
        f"Record ID: {data.get('id', '') or ''}",
        f"Short Description: {data.get('shortDescription', '') or ''}",
        f"Description: {data.get('description', '') or ''}",
        f"Venue: {data.get('venue', '') or ''}",
        f"Start Date: {data.get('startDate', '') or ''}",
        f"End Date: {data.get('endDate', '') or ''}",
        f"Project Type: {data.get('projectType', '') or ''}",
        f"Status: {data.get('status', '') or ''}",
        f"Domains: {domains}",
        f"Parent Project ID: {data.get('parentProjectId', '') or ''}",
    ]
    return "\n".join(lines).strip()


def build_retrieval_metadata(
    record: dict[str, Any],
    *,
    extra_metadata_keys: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source_key, metadata_key in RETRIEVAL_METADATA_FIELD_MAP:
        value = record.get(source_key)
        if value in (None, "", []):
            continue
        metadata[metadata_key] = _normalize_metadata_value(value)

    for key in extra_metadata_keys or ():
        if key not in record:
            continue
        metadata[key] = _normalize_metadata_value(record[key])

    if "source_id" not in metadata and record.get("source_doc_id"):
        metadata["source_id"] = _normalize_metadata_value(record["source_doc_id"])
    return metadata


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = splitter.split_documents(documents)
    chunk_counts: dict[str, int] = {}
    enriched_chunks: List[Document] = []

    for chunk in chunked_docs:
        metadata = dict(chunk.metadata)
        source_id = str(
            metadata.get("source_id")
            or metadata.get("source_doc_id")
            or "unknown-source"
        )
        chunk_index = chunk_counts.get(source_id, 0)
        chunk_counts[source_id] = chunk_index + 1
        metadata["chunk_index"] = chunk_index
        metadata["chunk_id"] = f"{source_id}#chunk-{chunk_index}"
        enriched_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )

    return enriched_chunks


def build_vectorstore(
    documents: Sequence[Document],
    *,
    config: RagConfig,
    reset: bool = False,
) -> Chroma:
    embeddings = build_embeddings(model=config.embedding_model)
    persist_dir = Path(config.persist_dir)

    if reset and persist_dir.exists():
        Chroma(persist_directory=str(persist_dir), embedding_function=embeddings).delete_collection()

    return Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def load_vectorstore(*, config: RagConfig) -> Chroma:
    embeddings = build_embeddings(model=config.embedding_model)
    return Chroma(
        persist_directory=str(config.persist_dir),
        embedding_function=embeddings,
    )


def similarity_search(
    vectorstore: Chroma,
    query: str,
    *,
    k: int = 4,
) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)


def retrieve_chunks(
    vectorstore: Chroma,
    query: str,
    *,
    k: int = 4,
) -> list[RetrievedChunk]:
    docs = similarity_search(vectorstore, query, k=k)
    return [
        RetrievedChunk(content=doc.page_content, metadata=dict(doc.metadata))
        for doc in docs
    ]


def build_context(docs: Iterable[Document], *, max_chars: int = 4000) -> str:
    chunks: List[str] = []
    total = 0
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        chunks.append(text)
        total += len(text)
    return "\n\n".join(chunks)


def build_context_from_chunks(
    chunks: Iterable[RetrievedChunk],
    *,
    max_chars: int = 4000,
) -> str:
    docs = [
        Document(page_content=chunk.content, metadata=dict(chunk.metadata))
        for chunk in chunks
    ]
    return build_context(docs, max_chars=max_chars)


def retrieve_context(
    vectorstore: Chroma,
    query: str,
    *,
    k: int = 4,
    max_chars: int = 4000,
) -> RetrievalResult:
    chunks = tuple(retrieve_chunks(vectorstore, query, k=k))
    context = build_context_from_chunks(chunks, max_chars=max_chars)
    return RetrievalResult(
        query=query,
        k=k,
        context=context,
        chunks=chunks,
    )


class RagRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    @classmethod
    def from_config(cls, *, config: RagConfig) -> "RagRetriever":
        return cls(load_vectorstore(config=config))

    def search(self, query: str, *, k: int = 4) -> list[RetrievedChunk]:
        return retrieve_chunks(self.vectorstore, query, k=k)

    def retrieve(
        self,
        query: str,
        *,
        k: int = 4,
        max_chars: int = 4000,
    ) -> RetrievalResult:
        return retrieve_context(
            self.vectorstore,
            query,
            k=k,
            max_chars=max_chars,
        )


def _coerce_record_to_dict(record: Any, *, idx: int) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, dict):
        return dict(record)
    raise TypeError(f"Unsupported record type at {idx}: {type(record)}")


def _build_metadata(
    record: dict[str, Any],
    *,
    metadata_keys: Optional[Sequence[str]] = None,
    id_key: Optional[str] = "id",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in metadata_keys or ():
        if key not in record:
            continue
        metadata[key] = _normalize_metadata_value(record[key])
    if id_key and id_key in record:
        metadata["source_id"] = _normalize_metadata_value(record[id_key])
    return metadata


def _normalize_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)
