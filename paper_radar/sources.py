from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime

from paper_radar.arxiv import fetch_recent_papers as fetch_arxiv_papers
from paper_radar.config import RadarConfig
from paper_radar.crossref import fetch_recent_papers as fetch_crossref_papers
from paper_radar.models import Paper
from paper_radar.openalex import fetch_recent_arxiv_papers as fetch_openalex_arxiv_papers
from paper_radar.semantic_scholar import EnrichmentError, describe_error, enrich_papers

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchResult:
    papers: list[Paper]
    warnings: tuple[str, ...]
    failures: tuple[str, ...] = ()


def _title_key(title: str) -> str:
    value = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def _merge(primary: Paper, secondary: Paper) -> Paper:
    source_names = []
    for source in (primary.source, secondary.source):
        for name in source.split(" + "):
            if name and name not in source_names:
                source_names.append(name)
    return replace(
        primary,
        abstract=primary.abstract or secondary.abstract,
        doi=primary.doi or secondary.doi,
        venue=primary.venue or secondary.venue,
        citation_count=(
            max(primary.citation_count or 0, secondary.citation_count or 0)
            if primary.citation_count is not None or secondary.citation_count is not None
            else None
        ),
        publication_types=primary.publication_types or secondary.publication_types,
        source=" + ".join(source_names),
    )


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    by_key: dict[str, Paper] = {}
    title_to_key: dict[str, str] = {}
    for paper in papers:
        doi_key = f"doi:{paper.doi}" if paper.doi else ""
        title_key = _title_key(paper.title)
        existing_key = doi_key if doi_key in by_key else title_to_key.get(title_key, "")
        if existing_key:
            by_key[existing_key] = _merge(by_key[existing_key], paper)
            continue
        key = doi_key or f"id:{paper.paper_id}"
        by_key[key] = paper
        if title_key:
            title_to_key[title_key] = key
    return sorted(by_key.values(), key=lambda paper: paper.published, reverse=True)


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
                warnings.append("arXiv 官方接口限流，已自动切换 OpenAlex 备份数据。")
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
