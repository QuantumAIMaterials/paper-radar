from __future__ import annotations

import html
import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from paper_radar.config import CrossrefConfig
from paper_radar.identity import normalize_doi
from paper_radar.models import Paper

LOGGER = logging.getLogger(__name__)


def _first_text(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    return re.sub(r"\s+", " ", str(value[0] or "")).strip()


def _plain_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _published(item: dict[str, Any]) -> datetime | None:
    for field in ("published-online", "published-print", "published", "created"):
        value = item.get(field)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts or not date_parts[0]:
            continue
        try:
            parts = [int(part) for part in date_parts[0]]
            return datetime(
                parts[0],
                parts[1] if len(parts) > 1 else 1,
                parts[2] if len(parts) > 2 else 1,
                tzinfo=UTC,
            )
        except (TypeError, ValueError):
            continue
    return None


def _updated(item: dict[str, Any], published: datetime) -> datetime:
    indexed = item.get("indexed")
    if isinstance(indexed, dict):
        value = str(indexed.get("date-time", "")).strip()
        if value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                pass
    return published


def parse_item(item: dict[str, Any]) -> Paper | None:
    doi = normalize_doi(item.get("DOI"))
    title = _plain_text(_first_text(item.get("title")))
    published = _published(item)
    if not doi or not title or published is None:
        return None

    authors = []
    for author in item.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part
            for part in (
                str(author.get("given") or "").strip(),
                str(author.get("family") or "").strip(),
            )
            if part
        )
        if name:
            authors.append(name)

    abstract_url = str(item.get("URL") or "").strip() or f"https://doi.org/{doi}"
    pdf_url = abstract_url
    for link in item.get("link") or []:
        if not isinstance(link, dict):
            continue
        if str(link.get("content-type", "")).lower() == "application/pdf":
            pdf_url = str(link.get("URL") or "").strip() or pdf_url
            break

    venue = _plain_text(_first_text(item.get("container-title")))
    subjects = tuple(
        _plain_text(subject)
        for subject in item.get("subject") or []
        if _plain_text(subject)
    )
    citation_count = item.get("is-referenced-by-count")
    return Paper(
        paper_id=f"crossref:{doi}",
        title=title,
        authors=tuple(authors),
        abstract=_plain_text(item.get("abstract")),
        published=published,
        updated=_updated(item, published),
        categories=subjects,
        abstract_url=abstract_url,
        pdf_url=pdf_url,
        source="Crossref",
        doi=doi,
        venue=venue,
        citation_count=(
            citation_count if type(citation_count) is int and citation_count >= 0 else None
        ),
        publication_types=(str(item.get("type") or "journal-article"),),
    )


def fetch_recent_papers(
    config: CrossrefConfig,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> list[Paper]:
    if not config.enabled:
        return []

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = current_time - timedelta(days=config.lookback_days)
    contact = os.getenv("ARXIV_CONTACT", "").strip()
    user_agent = "paper-radar/0.2"
    if contact:
        user_agent += f" (mailto:{contact})"

    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    )
    try:
        papers_by_doi: dict[str, Paper] = {}
        for index, term in enumerate(config.query_terms):
            params = {
                "query.bibliographic": term,
                "filter": (
                    f"from-pub-date:{cutoff.date().isoformat()},"
                    f"until-pub-date:{current_time.date().isoformat()},"
                    "type:journal-article"
                ),
                "rows": config.max_results_per_query,
                "sort": "published",
                "order": "desc",
            }
            if contact:
                params["mailto"] = contact
            for attempt in range(3):
                try:
                    response = http_client.get(config.api_url, params=params)
                    response.raise_for_status()
                    payload = response.json()
                    message = payload.get("message") if isinstance(payload, dict) else None
                    items = message.get("items") if isinstance(message, dict) else None
                    if not isinstance(items, list):
                        raise ValueError("Crossref response is missing a valid items list")
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        paper = parse_item(item)
                        if paper is not None and cutoff <= paper.published <= current_time:
                            papers_by_doi[paper.doi] = paper
                    break
                except (httpx.HTTPError, ValueError, TypeError):
                    if attempt == 2:
                        raise
                    delay = 2**attempt
                    LOGGER.warning(
                        "Crossref request failed; retrying in %d second(s)", delay
                    )
                    time.sleep(delay)
            if index < len(config.query_terms) - 1:
                time.sleep(1)

        papers = sorted(
            papers_by_doi.values(), key=lambda paper: paper.published, reverse=True
        )
        LOGGER.info("Fetched %d recent Crossref journal paper(s)", len(papers))
        return papers
    finally:
        if owns_client:
            http_client.close()
