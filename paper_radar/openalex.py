from __future__ import annotations

import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from paper_radar.config import ArxivConfig, OpenAlexConfig
from paper_radar.identity import normalize_doi
from paper_radar.models import Paper

LOGGER = logging.getLogger(__name__)
ARXIV_SOURCE_ID = "S4306400194"
ARXIV_ID_PATTERN = re.compile(r"(?:abs/|arxiv[.:])([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def _batches(terms: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
    return [terms[index : index + batch_size] for index in range(0, len(terms), batch_size)]


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value).astimezone(UTC)
        except (TypeError, ValueError):
            return None
        return max((retry_at - datetime.now(UTC)).total_seconds(), 0.0)


def _should_retry(error: Exception) -> bool:
    if not isinstance(error, httpx.HTTPStatusError):
        return True
    return error.response.status_code == 429 or error.response.status_code >= 500


def _abstract(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positioned_words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned_words.append((position, str(word)))
    positioned_words.sort()
    return " ".join(word for _, word in positioned_words)


def _arxiv_id(work: dict[str, Any]) -> str:
    location = work.get("primary_location")
    if not isinstance(location, dict):
        location = {}
    candidates = [
        location.get("landing_page_url"),
        location.get("id"),
        work.get("doi"),
    ]
    for candidate in candidates:
        match = ARXIV_ID_PATTERN.search(str(candidate or ""))
        if match:
            return match.group(1)
    return ""


def parse_work(work: dict[str, Any]) -> Paper | None:
    paper_id = _arxiv_id(work)
    title = str(work.get("title") or work.get("display_name") or "").strip()
    published_value = str(work.get("publication_date") or "").strip()
    if not paper_id or not title or not published_value:
        return None
    try:
        published = datetime.fromisoformat(published_value).replace(tzinfo=UTC)
    except ValueError:
        return None

    authors = []
    for authorship in work.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            author = {}
        name = str(author.get("display_name") or authorship.get("raw_author_name") or "").strip()
        if name:
            authors.append(name)

    location = work.get("primary_location")
    if not isinstance(location, dict):
        location = {}
    abstract_url = str(location.get("landing_page_url") or f"https://arxiv.org/abs/{paper_id}")
    pdf_url = str(location.get("pdf_url") or f"https://arxiv.org/pdf/{paper_id}")
    raw_doi = normalize_doi(work.get("doi"))
    doi = "" if raw_doi.startswith("10.48550/arxiv.") else raw_doi

    return Paper(
        paper_id=paper_id,
        title=title,
        authors=tuple(authors),
        abstract=_abstract(work.get("abstract_inverted_index")),
        published=published,
        updated=published,
        categories=(),
        abstract_url=abstract_url,
        pdf_url=pdf_url,
        source="arXiv via OpenAlex",
        doi=doi,
        venue="arXiv",
        citation_count=(
            work["cited_by_count"]
            if type(work.get("cited_by_count")) is int and work["cited_by_count"] >= 0
            else None
        ),
        publication_types=("preprint",),
    )


def fetch_recent_arxiv_papers(
    config: OpenAlexConfig,
    arxiv_config: ArxivConfig,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> list[Paper]:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    start_date = (current_time - timedelta(days=arxiv_config.lookback_days)).date()
    end_date = current_time.date()
    contact = os.getenv("ARXIV_CONTACT", "").strip()
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    user_agent = "paper-radar/0.2"
    if contact:
        user_agent += f" ({contact})"

    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    )
    try:
        papers_by_id: dict[str, Paper] = {}
        term_batches = _batches(arxiv_config.query_terms, config.query_batch_size)
        for batch_index, terms in enumerate(term_batches):
            search_terms = "|".join(term.replace("|", " ") for term in terms)
            filters = (
                f"from_publication_date:{start_date},"
                f"to_publication_date:{end_date},"
                f"primary_location.source.id:{ARXIV_SOURCE_ID},"
                f"title_and_abstract.search:{search_terms}"
            )
            params = {
                "filter": filters,
                "per-page": config.max_results_per_query,
                "sort": "publication_date:desc",
                "select": (
                    "id,doi,title,display_name,publication_date,primary_location,"
                    "authorships,abstract_inverted_index,cited_by_count"
                ),
            }
            if contact:
                params["mailto"] = contact
            if api_key:
                params["api_key"] = api_key

            for attempt in range(config.retry_attempts):
                try:
                    response = http_client.get(config.api_url, params=params)
                    response.raise_for_status()
                    body = response.json()
                    results = body.get("results") if isinstance(body, dict) else None
                    if not isinstance(results, list):
                        raise ValueError("OpenAlex response is missing a valid results list")
                    for work in results:
                        if not isinstance(work, dict):
                            continue
                        paper = parse_work(work)
                        if paper is not None:
                            papers_by_id[paper.paper_id] = paper
                    break
                except (httpx.HTTPError, ValueError) as error:
                    if attempt == config.retry_attempts - 1 or not _should_retry(error):
                        raise
                    response = (
                        error.response
                        if isinstance(error, httpx.HTTPStatusError)
                        else None
                    )
                    retry_after = _retry_after_seconds(response) or 0.0
                    delay = max(
                        min(
                            config.initial_retry_delay_seconds * (2**attempt),
                            config.max_retry_delay_seconds,
                        ),
                        retry_after,
                    )
                    LOGGER.warning(
                        "OpenAlex request failed; retrying in %.0f second(s) (%d/%d)",
                        delay,
                        attempt + 1,
                        config.retry_attempts,
                    )
                    time.sleep(delay)
            if batch_index < len(term_batches) - 1:
                time.sleep(config.request_interval_seconds)

        papers = sorted(
            papers_by_id.values(), key=lambda paper: paper.published, reverse=True
        )
        LOGGER.info("Fetched %d arXiv paper(s) through OpenAlex fallback", len(papers))
        return papers
    finally:
        if owns_client:
            http_client.close()
