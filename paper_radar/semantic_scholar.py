from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Any

import httpx

from paper_radar.config import SemanticScholarConfig
from paper_radar.models import Paper

LOGGER = logging.getLogger(__name__)

FIELDS = (
    "paperId,externalIds,title,venue,citationCount,influentialCitationCount,"
    "publicationTypes,fieldsOfStudy"
)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Semantic Scholar can temporarily throttle a valid key without Retry-After.
# Six attempts with the configured backoff give the shared quota time to recover
# without keeping a scheduled workflow occupied indefinitely.
MAX_ATTEMPTS = 6
MAX_RESPONSE_DETAIL_LENGTH = 300


class EnrichmentError(Exception):
    """Optional enrichment failed, but successful earlier results remain usable."""

    def __init__(self, papers: list[Paper], cause: Exception) -> None:
        super().__init__(describe_error(cause))
        self.papers = papers
        self.cause = cause


@dataclass
class _RequestPacer:
    interval: float
    sleep: Callable[[float], None]
    made_request: bool = False

    def wait(self) -> None:
        # Shared by normal batches, recursive splits and retries.
        if self.made_request and self.interval > 0:
            self.sleep(self.interval)
        self.made_request = True

    def backoff(self, delay: float) -> None:
        self.sleep(max(delay, self.interval))
        self.made_request = False


def _batches(papers: list[Paper], size: int) -> list[list[Paper]]:
    return [papers[index : index + size] for index in range(0, len(papers), size)]


def _identifier(paper: Paper) -> str:
    if paper.doi:
        return f"DOI:{paper.doi}"
    if paper.source.startswith("arXiv"):
        return f"ARXIV:{paper.paper_id}"
    return ""


def _enrich(paper: Paper, metadata: dict[str, Any]) -> Paper:
    external_ids = metadata.get("externalIds")
    doi = paper.doi
    if not doi and isinstance(external_ids, dict):
        doi = str(external_ids.get("DOI") or "").strip().lower()

    citation_count = metadata.get("citationCount")
    influential_count = metadata.get("influentialCitationCount")
    publication_types = metadata.get("publicationTypes")
    return replace(
        paper,
        doi=doi,
        venue=paper.venue or str(metadata.get("venue") or "").strip(),
        citation_count=(
            max(paper.citation_count or 0, citation_count)
            if type(citation_count) is int and citation_count >= 0
            else paper.citation_count
        ),
        influential_citation_count=(
            influential_count
            if type(influential_count) is int and influential_count >= 0
            else paper.influential_citation_count
        ),
        publication_types=(
            tuple(
                item.strip() for item in publication_types
                if isinstance(item, str) and item.strip()
            )
            or paper.publication_types
            if isinstance(publication_types, list)
            else paper.publication_types
        ),
    )


def _retry_delay(
    response: httpx.Response,
    attempt: int,
    *,
    initial_retry_delay_seconds: float,
    max_retry_delay_seconds: float,
) -> float:
    exponential_delay = min(
        initial_retry_delay_seconds * (2.0**attempt), max_retry_delay_seconds
    )
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = 0.0
            with suppress(TypeError, ValueError, OverflowError):
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = (retry_at - datetime.now(UTC)).total_seconds()
        if isfinite(seconds):
            exponential_delay = max(seconds, exponential_delay)
    return exponential_delay


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if isinstance(payload, dict):
        detail = payload.get("message") or payload.get("error") or payload
    else:
        detail = payload
    normalized = " ".join(str(detail).split())
    return normalized[:MAX_RESPONSE_DETAIL_LENGTH] or "no response detail"


def describe_error(error: Exception) -> str:
    if isinstance(error, EnrichmentError):
        return describe_error(error.cause)
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _fetch_batch(
    http_client: httpx.Client,
    api_url: str,
    identifiers: list[str],
    headers: dict[str, str],
    initial_retry_delay_seconds: float,
    max_retry_delay_seconds: float,
    pacer: _RequestPacer,
) -> list[Any]:
    previous_status: int | None = None
    retried_transient_bad_request = False
    for attempt in range(MAX_ATTEMPTS):
        pacer.wait()
        try:
            response = http_client.post(
                f"{api_url.rstrip('/')}/paper/batch",
                params={"fields": FIELDS},
                json={"ids": identifiers},
                headers=headers,
            )
        except httpx.TransportError as error:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = min(initial_retry_delay_seconds * (2.0**attempt), max_retry_delay_seconds)
            LOGGER.warning(
                "Semantic Scholar transport failed (%s); retrying in %.1fs (%d/%d)",
                type(error).__name__,
                delay,
                attempt + 1,
                MAX_ATTEMPTS,
            )
            pacer.backoff(delay)
            continue
        if (
            response.status_code == 400
            and _response_detail(response).casefold().rstrip(".") == "no valid paper ids given"
        ):
            # This lookup found no indexed records. Splitting the same IDs cannot
            # produce metadata and only consumes more of the provider's quota.
            LOGGER.info(
                "Semantic Scholar has no indexed records for %d identifier(s)", len(identifiers)
            )
            return [None] * len(identifiers)
        retryable = response.status_code in RETRYABLE_STATUS_CODES
        if (
            response.status_code == 400
            and previous_status in RETRYABLE_STATUS_CODES
            and not retried_transient_bad_request
        ):
            # The API has been observed returning 400 immediately after a 429
            # for the same valid payload. Give that inconsistent response one
            # retry before treating it as a genuine bad request.
            retryable = True
            retried_transient_bad_request = True
        if not retryable:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Semantic Scholar returned a non-list response")
            if len(payload) != len(identifiers):
                raise ValueError("Semantic Scholar returned a mismatched batch length")
            if any(item is not None and not isinstance(item, dict) for item in payload):
                raise ValueError("Semantic Scholar returned an invalid metadata entry")
            return payload

        if attempt == MAX_ATTEMPTS - 1:
            LOGGER.error(
                "Semantic Scholar retries exhausted with HTTP %d: %s",
                response.status_code,
                _response_detail(response),
            )
            response.raise_for_status()
        delay = _retry_delay(
            response,
            attempt,
            initial_retry_delay_seconds=initial_retry_delay_seconds,
            max_retry_delay_seconds=max_retry_delay_seconds,
        )
        if delay > max_retry_delay_seconds:
            LOGGER.warning(
                "Semantic Scholar requested %.1fs cooldown; leaving optional enrichment "
                "incomplete instead of retrying before Retry-After", delay,
            )
            response.raise_for_status()
        LOGGER.warning(
            "Semantic Scholar request returned %d: %s; retrying in %.1fs (%d/%d)",
            response.status_code,
            _response_detail(response),
            delay,
            attempt + 1,
            MAX_ATTEMPTS,
        )
        pacer.backoff(delay)
        previous_status = response.status_code

    raise RuntimeError("Semantic Scholar retry loop ended unexpectedly")


def _fetch_resilient_batch(
    http_client: httpx.Client,
    api_url: str,
    identifiers: list[str],
    headers: dict[str, str],
    config: SemanticScholarConfig,
    pacer: _RequestPacer,
    metadata_by_id: dict[str, dict[str, Any]],
) -> None:
    try:
        payload = _fetch_batch(
            http_client,
            api_url,
            identifiers,
            headers,
            config.initial_retry_delay_seconds,
            config.max_retry_delay_seconds,
            pacer,
        )
        for identifier, item in zip(identifiers, payload, strict=True):
            if isinstance(item, dict):
                metadata_by_id[identifier] = item
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 400:
            raise
        if error.response.headers.get("Retry-After"):
            delay = _retry_delay(
                error.response, 0,
                initial_retry_delay_seconds=config.initial_retry_delay_seconds,
                max_retry_delay_seconds=config.max_retry_delay_seconds,
            )
            if delay > config.max_retry_delay_seconds:
                raise
            pacer.backoff(delay)
        detail = _response_detail(error.response)
        if len(identifiers) == 1:
            LOGGER.warning(
                "Semantic Scholar rejected identifier %s with HTTP 400: %s; "
                "leaving its optional metadata unchanged",
                identifiers[0],
                detail,
            )
            return

        midpoint = len(identifiers) // 2
        LOGGER.warning(
            "Semantic Scholar rejected a batch of %d identifiers with HTTP 400: %s; "
            "splitting the batch to isolate invalid identifiers",
            len(identifiers),
            detail,
        )
        _fetch_resilient_batch(
            http_client,
            api_url,
            identifiers[:midpoint],
            headers,
            config,
            pacer,
            metadata_by_id,
        )
        _fetch_resilient_batch(
            http_client,
            api_url,
            identifiers[midpoint:],
            headers,
            config,
            pacer,
            metadata_by_id,
        )


def enrich_papers(
    papers: list[Paper],
    config: SemanticScholarConfig,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Paper]:
    if not config.enabled or not papers:
        return papers

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"User-Agent": "paper-radar/0.2"}
    if api_key:
        headers["x-api-key"] = api_key
    owns_client = client is None
    http_client = client or httpx.Client(timeout=45)
    pacer = _RequestPacer(config.request_interval_seconds, sleep)
    metadata_by_id: dict[str, dict[str, Any]] = {}

    def enriched_papers() -> list[Paper]:
        return [
            _enrich(paper, metadata) if (metadata := metadata_by_id.get(_identifier(paper)))
            else paper
            for paper in papers
        ]

    try:
        batches = _batches(papers, config.batch_size)
        for batch in batches:
            identifiers = list(dict.fromkeys(filter(None, map(_identifier, batch))))
            if identifiers:
                _fetch_resilient_batch(
                    http_client,
                    config.api_url,
                    identifiers,
                    headers,
                    config,
                    pacer,
                    metadata_by_id,
                )
        LOGGER.info(
            "Semantic Scholar matched metadata for %d of %d paper(s)",
            sum(_identifier(paper) in metadata_by_id for paper in papers),
            len(papers),
        )
        return enriched_papers()
    except (httpx.HTTPError, ValueError) as error:
        raise EnrichmentError(enriched_papers(), error) from error
    finally:
        if owns_client:
            http_client.close()
