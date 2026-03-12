from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True, frozen=True)
class NormalizedCommunityRecord:
    id: str
    source_doc_id: str
    source_collection: str
    source_system: str = "firebase"
    name: str = ""
    shortDescription: str = ""
    description: str = ""
    venue: str = ""
    startDate: str = ""
    endDate: str = ""
    projectType: str = ""
    status: str = ""
    parentProjectId: str = ""
    createdAt: str = ""
    updatedAt: str = ""
    domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
