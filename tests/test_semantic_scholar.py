import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from paper_radar.config import SemanticScholarConfig
from paper_radar.models import Paper
from paper_radar.semantic_scholar import EnrichmentError, describe_error, enrich_papers


def test_enrichment_adds_quality_metadata() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00001",
        title="Graphene paper",
        authors=("Author",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "externalIds": {"DOI": "10.1234/example"},
                    "venue": "Nature Physics",
                    "citationCount": 4,
                    "influentialCitationCount": 1,
                    "publicationTypes": ["JournalArticle"],
                }
            ],
            request=request,
        )

    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=100,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched = enrich_papers([paper], config, client=client)[0]

    assert enriched.doi == "10.1234/example"
    assert enriched.venue == "Nature Physics"
    assert enriched.citation_count == 4
    assert enriched.influential_citation_count == 1


def test_enrichment_sends_api_key(monkeypatch) -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00001",
        title="Graphene paper",
        authors=("Author",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-semantic-scholar-key"
        return httpx.Response(200, json=[None], request=request)

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key")
    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=100,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enrich_papers([paper], config, client=client)


def test_enrichment_retries_rate_limit() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00001",
        title="Graphene paper",
        authors=("Author",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "2.5"},
                request=request,
            )
        return httpx.Response(200, json=[{"citationCount": 7}], request=request)

    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=100,
        initial_retry_delay_seconds=1,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched = enrich_papers(
            [paper],
            config,
            client=client,
            sleep=delays.append,
        )[0]

    assert attempts == 2
    assert delays == [2.5]
    assert enriched.citation_count == 7


def test_enrichment_retries_one_bad_request_after_rate_limit() -> None:
    now = datetime(2026, 9, 5, tzinfo=UTC)
    paper = Paper(
        paper_id="2609.00001",
        title="Graphene paper",
        authors=("Author",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://arxiv.org/abs/2609.00001",
        pdf_url="https://arxiv.org/pdf/2609.00001",
    )
    responses = [429, 400, 200]
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = responses.pop(0)
        if status == 200:
            return httpx.Response(200, json=[{"citationCount": 9}], request=request)
        return httpx.Response(
            status,
            json={"message": f"temporary HTTP {status}"},
            request=request,
        )

    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=100,
        initial_retry_delay_seconds=10,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched = enrich_papers(
            [paper],
            config,
            client=client,
            sleep=delays.append,
        )[0]

    assert responses == []
    assert delays == [10, 20]
    assert enriched.citation_count == 9


def test_enrichment_splits_bad_batch_and_skips_only_invalid_identifier(
    caplog,
) -> None:
    now = datetime(2026, 9, 5, tzinfo=UTC)
    papers = [
        Paper(
            paper_id=f"2609.0000{index}",
            title=f"Graphene paper {index}",
            authors=("Author",),
            abstract="Abstract",
            published=now,
            updated=now,
            categories=(),
            abstract_url=f"https://arxiv.org/abs/2609.0000{index}",
            pdf_url=f"https://arxiv.org/pdf/2609.0000{index}",
        )
        for index in range(2)
    ]
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        identifiers = json.loads(request.content)["ids"]
        requests.append(identifiers)
        if len(identifiers) == 2:
            return httpx.Response(
                400,
                json={"message": "one identifier is invalid"},
                request=request,
            )
        if identifiers == ["ARXIV:2609.00000"]:
            return httpx.Response(200, json=[{"citationCount": 5}], request=request)
        return httpx.Response(
            400,
            json={"message": "invalid paper identifier"},
            request=request,
        )

    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=100,
        request_interval_seconds=0,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched = enrich_papers(papers, config, client=client)

    assert requests == [
        ["ARXIV:2609.00000", "ARXIV:2609.00001"],
        ["ARXIV:2609.00000"],
        ["ARXIV:2609.00001"],
    ]
    assert enriched[0].citation_count == 5
    assert enriched[1].citation_count is None
    assert "invalid paper identifier" in caplog.text


def test_enrichment_waits_between_batches() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    papers = [
        Paper(
            paper_id=f"2608.0000{index}",
            title=f"Graphene paper {index}",
            authors=("Author",),
            abstract="Abstract",
            published=now,
            updated=now,
            categories=(),
            abstract_url=f"https://arxiv.org/abs/2608.0000{index}",
            pdf_url=f"https://arxiv.org/pdf/2608.0000{index}",
        )
        for index in range(2)
    ]
    requests: list[list[str]] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content)["ids"])
        return httpx.Response(200, json=[None], request=request)

    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=1,
        request_interval_seconds=1.5,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enrich_papers(papers, config, client=client, sleep=delays.append)

    assert requests == [["ARXIV:2608.00000"], ["ARXIV:2608.00001"]]
    assert delays == [1.5]


def _paper(index: int = 0) -> Paper:
    now = datetime(2026, 10, 27, tzinfo=UTC)
    return Paper(
        paper_id=f"2610.0000{index}", title=f"Graphene paper {index}", authors=(),
        abstract="", published=now, updated=now, categories=(),
        abstract_url=f"https://arxiv.org/abs/2610.0000{index}", pdf_url="",
    )


def _config(**kwargs) -> SemanticScholarConfig:
    return SemanticScholarConfig(
        enabled=True, api_url="https://example.test/graph/v1", batch_size=1, **kwargs,
    )


@pytest.mark.parametrize("rate_limit_first", [False, True])
def test_unindexed_batch_does_not_trigger_retry_or_split_storm(rate_limit_first) -> None:
    papers = [_paper(i) for i in range(4)]
    requests = []
    delays = []

    def handler(request):
        requests.append(json.loads(request.content)["ids"])
        if rate_limit_first and len(requests) == 1:
            return httpx.Response(429)
        return httpx.Response(400, json={"error": "No valid paper ids given"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = enrich_papers(papers, replace(_config(), batch_size=100),
                               client=client, sleep=delays.append)

    assert result == papers
    assert len(requests) == (2 if rate_limit_first else 1)
    assert delays == ([10] if rate_limit_first else [])


def test_every_recursive_split_request_is_paced() -> None:
    elapsed = 0.0
    request_times = []

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds

    def handler(request):
        request_times.append(elapsed)
        ids = json.loads(request.content)["ids"]
        if len(ids) > 1:
            return httpx.Response(400, json={"error": "invalid identifier"})
        return httpx.Response(200, json=[{"citationCount": 2}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = enrich_papers(
            [_paper(i) for i in range(4)],
            replace(_config(request_interval_seconds=1.5), batch_size=100),
            client=client, sleep=sleep,
        )

    assert len(request_times) == 7
    assert all(b - a >= 1.5 for a, b in zip(request_times, request_times[1:], strict=False))
    assert all(paper.citation_count == 2 for paper in result)


@pytest.mark.parametrize("split_batch", [False, True])
def test_successful_metadata_survives_a_later_rate_limit(split_batch) -> None:
    papers = [_paper(0), replace(_paper(1), citation_count=8)]
    requests = []

    def handler(request):
        ids = json.loads(request.content)["ids"]
        requests.append(ids)
        if len(ids) > 1:
            return httpx.Response(400, json={"error": "invalid identifier"})
        if ids == ["ARXIV:2610.00000"]:
            return httpx.Response(200, json=[{"citationCount": 12}])
        return httpx.Response(429)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(EnrichmentError) as caught,
    ):
        enrich_papers(papers, replace(_config(), batch_size=100 if split_batch else 1),
                      client=client, sleep=lambda _: None)

    assert describe_error(caught.value) == "HTTP 429"
    assert [p.citation_count for p in caught.value.papers] == [12, 8]
    assert len(requests) == (8 if split_batch else 7)


@pytest.mark.parametrize("http_date", [False, True])
def test_retry_after_is_honored_for_seconds_and_http_dates(monkeypatch, http_date) -> None:
    now = datetime(2026, 10, 27, tzinfo=UTC)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr("paper_radar.semantic_scholar.datetime", FixedDateTime)
    header = format_datetime(now + timedelta(seconds=45), usegmt=True) if http_date else "45"
    requests = []
    delays = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": header})
        return httpx.Response(200, json=[{"citationCount": 1}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = enrich_papers([_paper()], _config(), client=client, sleep=delays.append)
    assert delays == [45]
    assert result[0].citation_count == 1


@pytest.mark.parametrize("status", [400, 429])
def test_long_retry_after_stops_optional_lookup_without_retrying_early(status) -> None:
    requests = []
    delays = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"Retry-After": "120"})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(EnrichmentError),
    ):
        enrich_papers([_paper(), _paper(1)], replace(_config(), batch_size=100),
                      client=client, sleep=delays.append)
    assert len(requests) == 1
    assert delays == []


def test_transient_timeout_is_retried() -> None:
    requests = []
    delays = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("temporarily unavailable", request=request)
        return httpx.Response(200, json=[{"citationCount": 3}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = enrich_papers([_paper()], _config(), client=client, sleep=delays.append)
    assert len(requests) == 2
    assert delays == [10]
    assert result[0].citation_count == 3


def test_null_metadata_does_not_create_fake_doi_or_erase_existing_fields() -> None:
    paper = replace(_paper(), citation_count=5, influential_citation_count=2,
                    publication_types=("JournalArticle",))
    metadata = {"externalIds": {"DOI": None}, "venue": None, "citationCount": None,
                "influentialCitationCount": None, "publicationTypes": [None, " "]}
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=[metadata])
    )) as client:
        result = enrich_papers([paper], _config(), client=client)
    assert result == [paper]


@pytest.mark.parametrize("payload", [[], [{}, {}], ["invalid"]])
def test_malformed_batch_cannot_silently_misattribute_metadata(payload) -> None:
    paper = _paper()
    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)
        )) as client,
        pytest.raises(EnrichmentError) as caught,
    ):
        enrich_papers([paper], _config(), client=client)
    assert caught.value.papers == [paper]
    assert isinstance(caught.value.cause, ValueError)
