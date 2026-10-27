from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    published: datetime
    updated: datetime
    categories: tuple[str, ...]
    abstract_url: str
    pdf_url: str
    source: str = "arXiv"
    doi: str = ""
    venue: str = ""
    citation_count: int | None = None
    influential_citation_count: int | None = None
    publication_types: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def prompt_text(self) -> str:
        authors = ", ".join(self.authors)
        categories = ", ".join(self.categories)
        metadata = (
            f"Paper ID: {self.paper_id}\n"
            f"Source: {self.source}\n"
            f"Title: {self.title}\n"
            f"Authors: {authors}\n"
            f"Categories: {categories}\n"
            f"Published: {self.published.date().isoformat()}\n"
        )
        if self.venue:
            metadata += f"Venue: {self.venue}\n"
        if self.doi:
            metadata += f"DOI: {self.doi}\n"
        if self.citation_count is not None:
            metadata += f"Citation count: {self.citation_count}\n"
        if self.publication_types:
            metadata += f"Publication types: {', '.join(self.publication_types)}\n"
        return metadata + f"Abstract: {self.abstract or '[Not supplied by source]'}"


@dataclass(frozen=True, slots=True)
class MatchResult:
    score: int
    core_terms: tuple[str, ...]
    supporting_terms: tuple[str, ...]
    focus_terms: tuple[str, ...] = ()
    preferred_terms: tuple[str, ...] = ()

    @property
    def matched_terms(self) -> tuple[str, ...]:
        ordered = (
            self.preferred_terms
            + self.core_terms
            + self.focus_terms
            + self.supporting_terms
        )
        return tuple(dict.fromkeys(ordered))


@dataclass(frozen=True, slots=True)
class Recommendation:
    paper: Paper
    relevance_score: int
    reason: str
    key_relevance: tuple[str, ...]
    title_zh: str
    summary_zh: str
    used_ai: bool
    priority_score: int = 0
    group_fit_score: int = 0
    novelty_score: int = 0
    method_value_score: int = 0
    evidence_score: int = 0
    study_type: str = "未判断"
    reading_action: str = "速读"
    quality_signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DeepRead:
    paper: Paper
    title_zh: str
    selection_reason: str
    one_sentence_summary: str
    technical_route: tuple[str, ...]
    takeaways: tuple[str, ...]
    advances: tuple[str, ...]
    limitations: tuple[str, ...]
    group_inspirations: tuple[str, ...]
    author_context: tuple[str, ...] = ()
