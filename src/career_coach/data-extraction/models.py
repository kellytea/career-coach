"""Extraction DTOs and PostgreSQL metadata, including immutable validation reports."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def uid() -> str:
    return str(uuid4())


def now() -> datetime:
    return datetime.now(timezone.utc)


class Block(BaseModel):
    id: str
    start: int
    end: int
    page: int | None = None
    section: str = "overview"
    heading: bool = False
    bullet: bool = False


class Mention(BaseModel):
    id: str = Field(default_factory=uid)
    label: str
    ontology_type: str | None = None
    text: str
    start: int
    end: int
    block_id: str
    block_ids: list[str] = Field(default_factory=list)
    method: str
    score: float | None = None
    polarity: Literal["positive", "negative", "uncertain"] = "positive"


class Relation(BaseModel):
    id: str = Field(default_factory=uid)
    subject: str
    predicate: str
    object: str
    evidence_start: int
    evidence_end: int
    method: str = "same_block_rule"
    attributes: dict = Field(default_factory=dict)


class Entry(BaseModel):
    id: str
    kind: Literal["experience", "education", "project"]
    block_ids: list[str]
    start: int
    end: int
    ambiguous: bool = False
    method: str = "section_paragraph_rule"


class Claim(BaseModel):
    id: str = Field(default_factory=uid)
    kind: Literal["skill", "degree", "certification"]
    mention_id: str
    entry_id: str | None = None
    evidence_start: int
    evidence_end: int
    polarity: Literal["positive", "negative", "uncertain"]
    basis: Literal["listed", "contextual"]


class Requirement(BaseModel):
    id: str = Field(default_factory=uid)
    kind: Literal["skill", "degree", "certification", "experience"]
    mention_id: str
    evidence_start: int
    evidence_end: int
    strength: Literal["required", "preferred", "unspecified"]
    polarity: Literal["positive", "negative", "uncertain"]
    alternatives_present: bool = False
    duration_mention_id: str | None = None


class Candidate(BaseModel):
    id: str
    namespace: str
    label: str
    version: str
    method: str
    score: float
    description: str = ""
    scores: dict[str, float] = Field(default_factory=dict)


class Resolution(BaseModel):
    mention_id: str
    status: Literal["linked_exact", "ambiguous", "unresolved", "candidate"]
    candidates: list[Candidate] = Field(default_factory=list)
    selected_id: str | None = None
    selected_namespace: str | None = None
    candidate_count: int = 0
    reason: str


class Extraction(BaseModel):
    scope: str = "stages_1_to_5_unvalidated"
    document_version_id: str
    text: str
    text_sha256: str
    blocks: list[Block]
    mentions: list[Mention]
    entries: list[Entry]
    claims: list[Claim]
    requirements: list[Requirement]
    relations: list[Relation]
    resolutions: list[Resolution]
    provenance: dict
    warnings: list[str] = Field(default_factory=list)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source", "external_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    kind: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(64))
    external_id: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Version(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    number: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    raw_key: Mapped[str] = mapped_column(String(512))
    format: Mapped[str] = mapped_column(String(16))
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Run(Base):
    __tablename__ = "extraction_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id"))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    config: Mapped[dict] = mapped_column(JSONB)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Sync(Base):
    __tablename__ = "source_syncs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    source: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    config: Mapped[dict] = mapped_column(JSONB)
    pages: Mapped[list] = mapped_column(JSONB, default=list)
    run_ids: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ValidationRecord(Base):
    __tablename__ = "validation_reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(ForeignKey("extraction_runs.id"), index=True)
    report_json: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
