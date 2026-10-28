from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import httpx

from paper_radar.config import ArxivConfig
from paper_radar.identity import normalize_doi
from paper_radar.models import Paper

LOGGER = logging.getLogger(__name__)
ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"


def build_search_query(terms: tuple[str, ...]) -> str:
    clauses = []
    for term in terms:
        escaped = term.replace('"', "")
        clauses.append(f'all:"{escaped}"')
    return " OR ".join(clauses)


def _batches(terms: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
    return [terms[index : index + batch_size] for index in range(0, len(terms), batch_size)]


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _text(entry: ET.Element, name: str) -> str:
    value = entry.findtext(f"{{{ATOM}}}{name}", default="")
    return re.sub(r"\s+", " ", value).strip()


def parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    if root.tag != f"{{{ATOM}}}feed":
        raise ValueError("arXiv returned a non-Atom response")
    papers: list[Paper] = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        raw_id = _text(entry, "id")
        paper_id = raw_id.rsplit("/abs/", 1)[-1]
        paper_id = re.sub(r"v\d+$", "", paper_id)
        authors = tuple(
            re.sub(r"\s+", " ", author.findtext(f"{{{ATOM}}}name", default="")).strip()
            for author in entry.findall(f"{{{ATOM}}}author")
        )
        categories = tuple(
            category.attrib.get("term", "")
            for category in entry.findall(f"{{{ATOM}}}category")
            if category.attrib.get("term")
        )
        links = entry.findall(f"{{{ATOM}}}link")
        abstract_url = next(
            (
                link.attrib["href"]
                for link in links
                if link.attrib.get("rel") == "alternate" and link.attrib.get("href")
            ),
            raw_id,
        )
        pdf_url = next(
            (
                link.attrib["href"]
                for link in links
                if link.attrib.get("title") == "pdf" and link.attrib.get("href")
            ),
            abstract_url.replace("/abs/", "/pdf/"),
        )
        doi = normalize_doi(entry.findtext(f"{{{ARXIV}}}doi"))
        papers.append(
            Paper(
                paper_id=paper_id,
                title=_text(entry, "title"),
                authors=authors,
                abstract=_text(entry, "summary"),
                published=_parse_datetime(_text(entry, "published")),
                updated=_parse_datetime(_text(entry, "updated")),
                categories=categories,
                abstract_url=abstract_url,
                pdf_url=pdf_url,
                source="arXiv",
                doi=doi,
            )
        )
    return papers


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


def fetch_recent_papers(
    config: ArxivConfig,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> list[Paper]:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = current_time - timedelta(days=config.lookback_days)
    contact = os.getenv("ARXIV_CONTACT", "").strip()
    user_agent = "paper-radar/0.1"
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
        term_batches = _batches(config.query_terms, config.query_batch_size)
        for batch_index, terms in enumerate(term_batches):
            params = {
                "search_query": build_search_query(terms),
                "start": 0,
                "max_results": config.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            for attempt in range(config.retry_attempts):
                try:
                    response = http_client.get(config.api_url, params=params)
                    response.raise_for_status()
                    for paper in parse_feed(response.text):
                        papers_by_id[paper.paper_id] = paper
                    break
                except (httpx.HTTPError, ET.ParseError, ValueError) as error:
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
                        "arXiv request failed; retrying in %.0f second(s) (%d/%d)",
                        delay,
                        attempt + 1,
                        config.retry_attempts,
                    )
                    time.sleep(delay)
            if batch_index < len(term_batches) - 1:
                time.sleep(3)

        papers = sorted(papers_by_id.values(), key=lambda paper: paper.published, reverse=True)
        recent = [paper for paper in papers if paper.published >= cutoff]
        LOGGER.info(
            "Fetched %d unique papers; %d are within the lookback window",
            len(papers),
            len(recent),
        )
        return recent
    finally:
        if owns_client:
            http_client.close()

    return []
