from __future__ import annotations

from langchain_core.documents import Document

from ingestion.schema import NormalizedCommunityRecord
from rag import RagConfig, RagRetriever, chunk_documents, load_vectorstore, normalized_records_to_documents


def test_normalized_records_to_documents_accepts_ingestion_schema():
    record = NormalizedCommunityRecord(
        id="event-1",
        source_doc_id="doc-1",
        source_collection="projects",
        name="Community Meetup",
        shortDescription="Monthly community meetup",
        description="Members gather to discuss ecosystem updates.",
        venue="Main Hall",
        startDate="2026-04-02T10:00:00+00:00",
        endDate="2026-04-02T12:00:00+00:00",
        projectType="CONFERENCE",
        status="PUBLISHED",
        parentProjectId="org-1",
        domains=["community", "events"],
    )

    documents = normalized_records_to_documents([record])

    assert len(documents) == 1
    document = documents[0]
    assert "Event: Community Meetup" in document.page_content
    assert "Record ID: event-1" in document.page_content
    assert document.metadata["source_id"] == "event-1"
    assert document.metadata["source_doc_id"] == "doc-1"
    assert document.metadata["source_collection"] == "projects"
    assert document.metadata["record_name"] == "Community Meetup"
    assert document.metadata["project_type"] == "CONFERENCE"
    assert document.metadata["domains"] == "community, events"


def test_chunk_documents_adds_stable_chunk_metadata():
    documents = [
        Document(
            page_content="A" * 160,
            metadata={"source_id": "event-1", "record_name": "Community Meetup"},
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=60,
        chunk_overlap=10,
    )

    assert len(chunks) >= 2
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata["chunk_id"] == "event-1#chunk-0"
    assert all(chunk.metadata["source_id"] == "event-1" for chunk in chunks)


def test_rag_retriever_returns_reusable_retrieval_result():
    class StubVectorStore:
        def __init__(self, docs: list[Document]):
            self.docs = docs
            self.calls: list[tuple[str, int]] = []

        def similarity_search(self, query: str, k: int = 4) -> list[Document]:
            self.calls.append((query, k))
            return self.docs[:k]

    docs = [
        Document(
            page_content="Community meetup happens every first Thursday.",
            metadata={"source_id": "event-1", "record_name": "Community Meetup"},
        ),
        Document(
            page_content="Hack night is scheduled for the last Friday of the month.",
            metadata={"source_id": "event-2", "record_name": "Hack Night"},
        ),
    ]
    retriever = RagRetriever(StubVectorStore(docs))

    result = retriever.retrieve(
        "What events are coming up this month?",
        k=2,
        max_chars=500,
    )

    assert result.query == "What events are coming up this month?"
    assert result.source_ids == ["event-1", "event-2"]
    assert len(result.chunks) == 2
    assert "Community meetup happens every first Thursday." in result.context
    assert "Hack night is scheduled for the last Friday of the month." in result.context
    assert result.to_dict()["source_ids"] == ["event-1", "event-2"]
    assert retriever.vectorstore.calls == [("What events are coming up this month?", 2)]


def test_load_vectorstore_uses_dynamic_embedding_builder(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_embeddings(*, model=None):
        captured["model"] = model
        return "embeddings"

    class StubChroma:
        def __init__(self, *, persist_directory, embedding_function):
            captured["persist_directory"] = persist_directory
            captured["embedding_function"] = embedding_function

    monkeypatch.setattr("rag.pipeline.build_embeddings", fake_build_embeddings)
    monkeypatch.setattr("rag.pipeline.Chroma", StubChroma)

    load_vectorstore(config=RagConfig(persist_dir="vector_db", embedding_model=None))

    assert captured["model"] is None
    assert captured["persist_directory"] == "vector_db"
    assert captured["embedding_function"] == "embeddings"
