from datetime import UTC, datetime

import httpx
import pytest

from paper_radar.config import ArxivConfig, OpenAlexConfig
from paper_radar.openalex import fetch_recent_arxiv_papers, parse_work

WORK = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.48550/arxiv.2609.09422",
    "title": "Quantum transport in rhombohedral graphene",
    "publication_date": "2026-09-08",
    "primary_location": {
        "id": "pmh:oai:arXiv.org:2609.09422",
        "landing_page_url": "https://arxiv.org/abs/2609.09422",
        "pdf_url": "https://arxiv.org/pdf/2609.09422",
    },
    "authorships": [
        {"author": {"display_name": "Alice Example"}},
        {"author": {"display_name": "Bob Example"}},
    ],
    "abstract_inverted_index": {
        "Graphene": [3],
        "We": [0],
        "study": [1],
        "transport.": [2],
    },
}


def test_parse_work_restores_arxiv_metadata_and_abstract() -> None:
    paper = parse_work(WORK)

    assert paper is not None
    assert paper.paper_id == "2609.09422"
    assert paper.authors == ("Alice Example", "Bob Example")
    assert paper.abstract == "We study transport. Graphene"
    assert paper.source == "arXiv via OpenAlex"
    assert paper.doi == ""


def test_fetch_batches_queries_and_deduplicates(monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": [WORK]}, request=request)

    monkeypatch.setattr("paper_radar.openalex.time.sleep", lambda _seconds: None)
    openalex_config = OpenAlexConfig(
        enabled=True,
        api_url="https://example.test/works",
        max_results_per_query=100,
        query_batch_size=1,
        request_interval_seconds=1,
    )
    arxiv_config = ArxivConfig(
        api_url="https://example.test/api/query",
        max_results=75,
        lookback_days=8,
        query_batch_size=1,
        query_terms=("graphene", "quantum transport"),
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_recent_arxiv_papers(
            openalex_config,
            arxiv_config,
            now=datetime(2026, 9, 14, tzinfo=UTC),
            client=client,
        )

    assert len(requests) == 2
    assert len(papers) == 1
    assert papers[0].paper_id == "2609.09422"
    assert "primary_location.source.id%3AS4306400194" in str(requests[0].url)
    assert "cited_by_count" in requests[0].url.params["select"]


def test_fetch_retries_rate_limit_and_honors_retry_after(monkeypatch) -> None:
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "75"},
                request=request,
            )
        return httpx.Response(200, json={"results": [WORK]}, request=request)

    monkeypatch.setattr("paper_radar.openalex.time.sleep", sleeps.append)
    openalex_config = OpenAlexConfig(
        enabled=True,
        api_url="https://example.test/works",
        max_results_per_query=100,
        query_batch_size=50,
        request_interval_seconds=1,
    )
    arxiv_config = ArxivConfig(
        api_url="https://example.test/api/query",
        max_results=75,
        lookback_days=8,
        query_batch_size=6,
        query_terms=("graphene", "quantum transport"),
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_recent_arxiv_papers(
            openalex_config,
            arxiv_config,
            now=datetime(2026, 9, 14, tzinfo=UTC),
            client=client,
        )

    assert attempts == 2
    assert sleeps == [75]
    assert [paper.paper_id for paper in papers] == ["2609.09422"]


@pytest.mark.parametrize("count, expected", [(0, 0), (12, 12), (None, None), (-1, None)])
def test_openalex_citations_remain_available_without_semantic_scholar(count, expected) -> None:
    paper = parse_work({**WORK, "cited_by_count": count})
    assert paper is not None
    assert paper.citation_count == expected
