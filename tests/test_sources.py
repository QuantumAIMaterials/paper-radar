from datetime import UTC, datetime
from types import SimpleNamespace

import httpx

from paper_radar.models import Paper
from paper_radar.semantic_scholar import EnrichmentError
from paper_radar.sources import deduplicate_papers, fetch_all_papers


def _paper(paper_id: str, source: str, *, doi: str = "", abstract: str = "") -> Paper:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    return Paper(
        paper_id=paper_id,
        title="The same graphene result",
        authors=("Author",),
        abstract=abstract,
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://example.test/paper",
        pdf_url="https://example.test/paper.pdf",
        source=source,
        doi=doi,
    )


def test_deduplication_merges_arxiv_and_crossref_metadata() -> None:
    arxiv = _paper("2608.00001", "arXiv", abstract="Detailed abstract")
    crossref = _paper(
        "crossref:10.1234/example",
        "Crossref",
        doi="10.1234/example",
    )

    papers = deduplicate_papers([arxiv, crossref])

    assert len(papers) == 1
    assert papers[0].paper_id == arxiv.paper_id
    assert papers[0].doi == "10.1234/example"
    assert papers[0].source == "arXiv + Crossref"


def _config():
    return SimpleNamespace(
        arxiv=object(),
        openalex=SimpleNamespace(enabled=True),
        crossref=SimpleNamespace(enabled=False),
        semantic_scholar=SimpleNamespace(enabled=False),
    )


def test_openalex_replaces_a_failed_arxiv_source(monkeypatch) -> None:
    fallback_paper = _paper("2608.00002", "arXiv via OpenAlex")
    monkeypatch.setattr(
        "paper_radar.sources.fetch_arxiv_papers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("limited")),
    )
    monkeypatch.setattr(
        "paper_radar.sources.fetch_openalex_arxiv_papers",
        lambda *args, **kwargs: [fallback_paper],
    )

    result = fetch_all_papers(_config(), enrich_semantic_scholar=False)

    assert result.papers == [fallback_paper]
    assert result.failures == ()
    assert result.warnings == (
        "arXiv 官方接口限流，已自动切换 OpenAlex 备份数据。",
    )


def test_failed_arxiv_and_fallback_are_reported_as_incomplete(monkeypatch) -> None:
    crossref_paper = _paper("crossref:example", "Crossref")
    config = _config()
    config.crossref.enabled = True
    monkeypatch.setattr(
        "paper_radar.sources.fetch_arxiv_papers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("limited")),
    )
    monkeypatch.setattr(
        "paper_radar.sources.fetch_openalex_arxiv_papers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        "paper_radar.sources.fetch_crossref_papers",
        lambda *args, **kwargs: [crossref_paper],
    )

    result = fetch_all_papers(config, enrich_semantic_scholar=False)

    assert result.papers == [crossref_paper]
    assert result.failures == result.warnings
    assert result.failures == (
        "arXiv 官方接口和 OpenAlex 备份暂时不可用，本次结果仅来自其他可用数据源；"
        "系统会在后续计划任务中自动补查。",
    )


def test_optional_enrichment_retains_partial_results(monkeypatch) -> None:
    original = _paper("2608.00001", "arXiv")
    enriched = _paper("2608.00001", "arXiv", doi="10.1234/enriched")
    config = _config()
    config.semantic_scholar.enabled = True
    monkeypatch.setattr("paper_radar.sources.fetch_arxiv_papers", lambda *a, **kw: [original])

    def fail(*args):
        raise EnrichmentError([enriched], httpx.ReadTimeout("offline"))

    monkeypatch.setattr("paper_radar.sources.enrich_papers", fail)
    result = fetch_all_papers(config)
    assert result.papers == [enriched]
    assert result.failures == ()
    assert "ReadTimeout" in result.warnings[0]
