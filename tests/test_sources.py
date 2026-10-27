from dataclasses import replace
from datetime import UTC, datetime
from itertools import permutations
from types import SimpleNamespace

import httpx
import pytest

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
        "arXiv 官方接口暂不可用（RuntimeError），已自动切换 OpenAlex 备份数据。",
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


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_deduplication_keeps_transitive_doi_and_title_aliases(order) -> None:
    arxiv = _paper("2608.00001", "arXiv", abstract="Detailed abstract")
    journal = _paper("crossref:10.1234/example", "Crossref", doi="10.1234/example")
    variant = replace(journal, paper_id="other-source", title="Retitled published result",
                      doi="https://doi.org/10.1234/EXAMPLE", citation_count=7)
    inputs = [arxiv, journal, variant]
    papers = deduplicate_papers([inputs[i] for i in order])
    assert len(papers) == 1
    assert papers[0].paper_id == arxiv.paper_id
    assert papers[0].citation_count == 7
    assert papers[0].abstract == "Detailed abstract"
    assert "id:other-source" in papers[0].aliases
    assert "doi:10.1234/example" in papers[0].aliases


def test_same_run_deduplication_keeps_distinct_greek_phases() -> None:
    alpha = replace(_paper("alpha", "arXiv"), title="Transport in α phase")
    beta = replace(_paper("beta", "arXiv"), title="Transport in β phase")
    assert len(deduplicate_papers([alpha, beta])) == 2


def test_duplicate_id_merges_metadata_instead_of_overwriting_it() -> None:
    original = _paper("2608.00001", "arXiv", abstract="Available abstract")
    updated = replace(original, title="Updated title", abstract="", citation_count=4)
    papers = deduplicate_papers([original, updated])
    assert len(papers) == 1
    assert papers[0].abstract == "Available abstract"
    assert papers[0].citation_count == 4
