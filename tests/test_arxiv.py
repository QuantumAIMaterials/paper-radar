from datetime import UTC, datetime

import httpx
import pytest

from paper_radar.arxiv import build_search_query, fetch_recent_papers, parse_feed
from paper_radar.config import ArxivConfig

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2608.01234v2</id>
    <updated>2026-08-14T05:00:00Z</updated>
    <published>2026-08-13T12:00:00Z</published>
    <title>  Twisted bilayer graphene\n transport  </title>
    <summary>We study correlated transport.</summary>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
    <category term="cond-mat.mes-hall" />
    <link href="https://arxiv.org/abs/2608.01234v2" rel="alternate" type="text/html" />
    <link title="pdf" href="https://arxiv.org/pdf/2608.01234v2"
          rel="related" type="application/pdf" />
  </entry>
</feed>
"""


def test_parse_feed_normalizes_paper_id_and_whitespace() -> None:
    papers = parse_feed(FEED)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.paper_id == "2608.01234"
    assert paper.title == "Twisted bilayer graphene transport"
    assert paper.authors == ("Alice Example", "Bob Example")
    assert paper.categories == ("cond-mat.mes-hall",)
    assert paper.pdf_url.endswith("2608.01234v2")


def test_build_search_query_uses_quoted_all_fields() -> None:
    query = build_search_query(("2D materials", "graphene"))
    assert query == 'all:"2D materials" OR all:"graphene"'


def test_fetch_batches_queries_and_deduplicates(monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=FEED, request=request)

    monkeypatch.setattr("paper_radar.arxiv.time.sleep", lambda _seconds: None)
    config = ArxivConfig(
        api_url="https://example.test/api/query",
        max_results=10,
        lookback_days=8,
        query_batch_size=1,
        query_terms=("graphene", "MoS2"),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_recent_papers(
            config,
            now=datetime(2026, 8, 14, tzinfo=UTC),
            client=client,
        )

    assert len(requests) == 2
    assert len(papers) == 1
    assert papers[0].paper_id == "2608.01234"


def test_fetch_honors_retry_after_for_rate_limits(monkeypatch) -> None:
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
        return httpx.Response(200, text=FEED, request=request)

    monkeypatch.setattr("paper_radar.arxiv.time.sleep", sleeps.append)
    config = ArxivConfig(
        api_url="https://example.test/api/query",
        max_results=10,
        lookback_days=8,
        query_batch_size=1,
        query_terms=("graphene",),
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_recent_papers(
            config,
            now=datetime(2026, 8, 14, tzinfo=UTC),
            client=client,
        )

    assert len(papers) == 1
    assert attempts == 2
    assert sleeps == [75]


@pytest.mark.parametrize("body", ["<html><body>Temporary outage</body></html>", "<error/>"])
def test_non_atom_success_cannot_be_mistaken_for_zero_papers(body):
    with pytest.raises(ValueError, match="non-Atom"):
        parse_feed(body)


def test_valid_empty_atom_feed_is_still_successful():
    assert parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"/>') == []
