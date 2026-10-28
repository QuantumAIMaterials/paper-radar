from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime

from paper_radar.arxiv import fetch_recent_papers as fetch_arxiv_papers
from paper_radar.config import RadarConfig
from paper_radar.crossref import fetch_recent_papers as fetch_crossref_papers
from paper_radar.identity import normalize_doi, paper_keys
from paper_radar.models import Paper
from paper_radar.openalex import fetch_recent_arxiv_papers as fetch_openalex_arxiv_papers
from paper_radar.semantic_scholar import EnrichmentError, describe_error, enrich_papers

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchResult:
    papers: list[Paper]
    warnings: tuple[str, ...]
    failures: tuple[str, ...] = ()


def _merge(primary: Paper, secondary: Paper) -> Paper:
    # Preserve the direct arXiv PDF regardless of source arrival order.
    if secondary.source.startswith("arXiv") and not primary.source.startswith("arXiv"):
        primary, secondary = secondary, primary
    source_names = []
    for source in (primary.source, secondary.source):
        for name in source.split(" + "):
            if name and name not in source_names:
                source_names.append(name)
    return replace(
        primary,
        authors=primary.authors or secondary.authors,
        abstract=primary.abstract or secondary.abstract,
        doi=normalize_doi(primary.doi) or normalize_doi(secondary.doi),
        venue=(secondary.venue or primary.venue)
        if primary.venue.lower() in {"", "arxiv"} else primary.venue,
        citation_count=(
            max(primary.citation_count or 0, secondary.citation_count or 0)
            if primary.citation_count is not None or secondary.citation_count is not None
            else None
        ),
        influential_citation_count=(
            max(primary.influential_citation_count or 0, secondary.influential_citation_count or 0)
            if (primary.influential_citation_count is not None
                or secondary.influential_citation_count is not None) else None
        ),
        publication_types=tuple(dict.fromkeys(
            primary.publication_types + secondary.publication_types
        )),
        aliases=tuple(sorted(paper_keys(primary) | paper_keys(secondary))),
        source=" + ".join(source_names),
    )


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    groups: dict[int, Paper] = {}
    key_to_group: dict[str, int] = {}
    for index, paper in enumerate(papers):
        keys = paper_keys(paper)
        existing = sorted({key_to_group[key] for key in keys if key in key_to_group})
        group = existing[0] if existing else index
        if existing:
            merged = groups[group]
            for other in existing[1:]:
                merged = _merge(merged, groups.pop(other))
            paper = _merge(merged, paper)
        groups[group] = paper
        # All old aliases follow the merged group, including a DOI learned
        # from a same-title journal record and titles linked later by that DOI.
        for key in paper_keys(paper):
            key_to_group[key] = group
    return sorted(groups.values(), key=lambda paper: paper.published, reverse=True)


def fetch_all_papers(
    config: RadarConfig,
    *,
    now: datetime | None = None,
    enrich_semantic_scholar: bool = True,
) -> FetchResult:
    papers: list[Paper] = []
    warnings: list[str] = []
    failures: list[str] = []
    successful_sources = 0

    try:
        arxiv_papers = fetch_arxiv_papers(config.arxiv, now=now)
        papers.extend(arxiv_papers)
        successful_sources += 1
    except Exception as error:
        LOGGER.warning(
            "arXiv source unavailable; trying OpenAlex fallback (%s)",
            type(error).__name__,
        )
        if config.openalex.enabled:
            try:
                fallback_papers = fetch_openalex_arxiv_papers(
                    config.openalex,
                    config.arxiv,
                    now=now,
                )
                papers.extend(fallback_papers)
                successful_sources += 1
                warnings.append(
                    f"arXiv 官方接口暂不可用（{describe_error(error)}），"
                    "已自动切换 OpenAlex 备份数据。"
                )
            except Exception:
                LOGGER.exception("OpenAlex arXiv fallback failed")
                message = (
                    "arXiv 官方接口和 OpenAlex 备份暂时不可用，本次结果仅来自"
                    "其他可用数据源；系统会在后续计划任务中自动补查。"
                )
                warnings.append(message)
                failures.append(message)
        else:
            message = f"arXiv 数据源失败：{type(error).__name__}"
            warnings.append(message)
            failures.append(message)

    if config.crossref.enabled:
        try:
            crossref_papers = fetch_crossref_papers(config.crossref, now=now)
            papers.extend(crossref_papers)
            successful_sources += 1
        except Exception as error:
            LOGGER.exception("Crossref source failed")
            message = f"Crossref 数据源失败：{type(error).__name__}"
            warnings.append(message)
            failures.append(message)

    if successful_sources == 0:
        raise RuntimeError("All configured paper sources failed")

    papers = deduplicate_papers(papers)
    if enrich_semantic_scholar and config.semantic_scholar.enabled:
        try:
            papers = enrich_papers(papers, config.semantic_scholar)
        except Exception as error:
            LOGGER.exception("Semantic Scholar enrichment failed")
            if isinstance(error, EnrichmentError):
                papers = error.papers
            warnings.append(
                "Semantic Scholar 附加元数据暂不可用："
                f"{describe_error(error)}"
            )

    LOGGER.info("Aggregated %d unique paper(s) from all sources", len(papers))
    return FetchResult(
        papers=papers,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )
