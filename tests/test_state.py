import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from paper_radar.models import Paper, Recommendation
from paper_radar.state import StateStore


def make_paper(paper_id: str) -> Paper:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        authors=("A. Researcher",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


def make_recommendation(paper: Paper) -> Recommendation:
    return Recommendation(
        paper=paper,
        relevance_score=3,
        reason="cached reason",
        key_relevance=("graphene",),
        title_zh="cached title",
        summary_zh="cached summary",
        used_ai=True,
        priority_score=88,
        group_fit_score=5,
        novelty_score=4,
        method_value_score=4,
        evidence_score=4,
        study_type="experiment",
        reading_action="read",
        quality_signals=("clear method",),
    )


def test_state_round_trip_and_prune(tmp_path) -> None:
    path = tmp_path / "seen.json"
    store = StateStore(path)
    old_time = datetime(2025, 1, 1, tzinfo=UTC)
    current_time = datetime(2026, 8, 14, tzinfo=UTC)
    old_paper = make_paper("old")
    new_paper = make_paper("new")
    store.mark(old_paper, "sent", now=old_time)
    store.mark(new_paper, "sent", now=current_time)
    store.cache_ai_evaluation("old-cache", make_recommendation(old_paper), now=old_time)
    store.cache_ai_evaluation(
        "new-cache",
        make_recommendation(new_paper),
        now=current_time,
    )
    store.prune(365, now=current_time + timedelta(days=1))
    store.save()

    loaded = StateStore(path)
    assert not loaded.contains("old")
    assert loaded.contains("new")
    assert loaded.cached_ai_evaluation(old_paper, "old-cache") is None
    assert loaded.cached_ai_evaluation(new_paper, "new-cache") is not None


def test_deferred_paper_is_reconsidered() -> None:
    store = StateStore.__new__(StateStore)
    store.path = None
    store.data = {"version": 2, "papers": {}}
    paper = make_paper("deferred")

    store.mark(paper, "deferred")

    assert store.contains(paper.paper_id)
    assert store.should_consider(paper.paper_id)
    store.mark(paper, "sent")
    assert not store.should_consider(paper.paper_id)


def test_ai_evaluation_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "seen.json"
    paper = make_paper("cached")
    store = StateStore(path)

    store.cache_ai_evaluation("cache-key", make_recommendation(paper))
    store.save()

    loaded = StateStore(path)
    cached = loaded.cached_ai_evaluation(paper, "cache-key")

    assert cached is not None
    assert cached.paper is paper
    assert cached.used_ai is True
    assert cached.priority_score == 88
    assert cached.key_relevance == ("graphene",)
    assert cached.quality_signals == ("clear method",)
    assert loaded.cached_ai_evaluation(make_paper("other"), "cache-key") is None


def test_completed_beijing_date_round_trip(tmp_path) -> None:
    path = tmp_path / "seen.json"
    store = StateStore(path)

    store.mark_completed(date(2026, 9, 9))
    store.save()

    loaded = StateStore(path)
    assert loaded.completed_on(date(2026, 9, 9))
    assert not loaded.completed_on(date(2026, 9, 10))


def test_alert_date_round_trip_is_scoped_by_kind(tmp_path) -> None:
    path = tmp_path / "seen.json"
    store = StateStore(path)

    store.mark_alerted(date(2026, 9, 9), "source_failure")
    store.save()

    loaded = StateStore(path)
    assert loaded.alerted_on(date(2026, 9, 9), "source_failure")
    assert not loaded.alerted_on(date(2026, 9, 10), "source_failure")
    assert not loaded.alerted_on(date(2026, 9, 9), "delivery_failure")


def test_version_one_skipped_records_migrate_to_deferred(tmp_path) -> None:
    path = tmp_path / "seen.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "papers": {
                    "old-skipped": {"status": "skipped"},
                    "already-sent": {"status": "sent"},
                },
            }
        ),
        encoding="utf-8",
    )

    store = StateStore(path)

    assert store.data["version"] == 2
    assert store.status("old-skipped") == "deferred"
    assert store.status("already-sent") == "sent"


def test_sent_preprint_blocks_later_journal_duplicate_from_existing_state(tmp_path) -> None:
    path = tmp_path / "seen.json"
    # Exercise the pre-existing state format, which has no DOI field.
    path.write_text(json.dumps({"version": 2, "papers": {
        "2609.01258": {"status": "sent", "title": "Electric-field-driven graphene transport"}
    }}))
    journal = replace(make_paper("crossref:10.1234/graphene"), source="Crossref",
                      doi="10.1234/graphene", title="Electric field driven graphene transport")
    assert not StateStore(path).should_consider(journal)


def test_legacy_crossref_doi_blocks_preprint_even_when_title_changes(tmp_path) -> None:
    store = StateStore(tmp_path / "seen.json")
    store.data["papers"]["crossref:10.1234/graphene"] = {
        "status": "sent", "title": "Old journal title",
    }
    paper = replace(make_paper("2609.01258"), doi="https://doi.org/10.1234/GRAPHENE")
    assert not store.should_consider(paper)


def test_doi_is_persisted_for_cross_source_deduplication(tmp_path) -> None:
    path = tmp_path / "seen.json"
    store = StateStore(path)
    store.mark(replace(make_paper("2609.01258"), doi="10.1234/graphene"), "sent")
    store.save()
    journal = replace(make_paper("crossref:10.1234/graphene"), doi="10.1234/graphene")
    assert not StateStore(path).should_consider(journal)


@pytest.mark.parametrize("status", ["deferred", "skipped"])
def test_unsent_other_source_does_not_block_a_new_candidate(tmp_path, status) -> None:
    store = StateStore(tmp_path / "seen.json")
    prior = make_paper("2609.01258")
    store.mark(prior, status)
    assert store.should_consider(replace(prior, paper_id="crossref:10.1234/new"))


def test_title_matching_preserves_distinct_greek_phases(tmp_path) -> None:
    store = StateStore(tmp_path / "seen.json")
    store.mark(replace(make_paper("alpha"), title="Transport in α phase"), "sent")
    assert store.should_consider(replace(make_paper("beta"), title="Transport in β phase"))
