from datetime import UTC, datetime

import httpx
import pytest

from paper_radar.config import CrossrefConfig
from paper_radar.crossref import fetch_recent_papers, parse_item


def _item() -> dict:
    return {
        "DOI": "10.1234/Graphene.1",
        "title": ["Graphene transport in a van der Waals device"],
        "abstract": "<jats:p>We report a mobility of 10,000 cm2/Vs.</jats:p>",
        "published-online": {"date-parts": [[2026, 8, 13]]},
        "indexed": {"date-time": "2026-08-14T01:00:00Z"},
        "author": [{"given": "Alice", "family": "Example"}],
        "URL": "https://doi.org/10.1234/graphene.1",
        "container-title": ["Journal of 2D Materials"],
        "subject": ["Materials Science"],
        "is-referenced-by-count": 2,
        "type": "journal-article",
    }


def test_parse_item_builds_crossref_paper() -> None:
    paper = parse_item(_item())
    assert paper is not None
    assert paper.paper_id == "crossref:10.1234/graphene.1"
    assert paper.source == "Crossref"
    assert paper.venue == "Journal of 2D Materials"
    assert paper.citation_count == 2
    assert "jats" not in paper.abstract


def test_fetch_recent_crossref_papers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"items": [_item()]}},
            request=request,
        )

    config = CrossrefConfig(
        enabled=True,
        api_url="https://example.test/works",
        max_results_per_query=10,
        lookback_days=8,
        query_terms=("graphene",),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_recent_papers(
            config,
            now=datetime(2026, 8, 14, tzinfo=UTC),
            client=client,
        )

    assert len(papers) == 1
    assert papers[0].doi == "10.1234/graphene.1"


@pytest.mark.parametrize("payload", [{}, {"message": {}}, {"message": {"items": None}}, []])
def test_malformed_success_response_is_not_an_empty_success(monkeypatch, payload):
    monkeypatch.setattr("paper_radar.crossref.time.sleep", lambda _: None)
    config = CrossrefConfig(True, "https://example.test/works", 10, 8, ("graphene",))
    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)
        )) as client,
        pytest.raises(ValueError, match="items list"),
    ):
        fetch_recent_papers(config, client=client)


def test_null_optional_fields_do_not_break_crossref_record():
    item = {**_item(), "author": None, "link": None, "subject": None, "URL": None,
            "container-title": [None], "type": None, "is-referenced-by-count": -1}
    paper = parse_item(item)
    assert paper is not None
    assert paper.abstract_url == "https://doi.org/10.1234/graphene.1"
    assert paper.authors == ()
    assert paper.venue == ""
    assert paper.citation_count is None
    assert paper.publication_types == ("journal-article",)
    assert parse_item({**item, "DOI": None}) is None
    assert parse_item({**item, "title": [None]}) is None


def test_future_publication_is_excluded_from_recent_results():
    item = {**_item(), "published-online": {"date-parts": [[2026, 9, 1]]}}
    config = CrossrefConfig(True, "https://example.test/works", 10, 8, ("graphene",))
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"message": {"items": [item]}})
    )) as client:
        assert fetch_recent_papers(config, client=client,
                                   now=datetime(2026, 8, 14, tzinfo=UTC)) == []
