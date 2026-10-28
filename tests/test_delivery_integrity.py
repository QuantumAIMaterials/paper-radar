from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from paper_radar import app
from paper_radar.feishu import FeishuClient
from paper_radar.models import MatchResult, Paper, Recommendation
from paper_radar.sources import FetchResult
from paper_radar.state import StateStore


def _paper(paper_id="2610.00001", title="Graphene quantum transport"):
    now = datetime.now(UTC)
    return Paper(paper_id, title, (), "Quantum transport in graphene", now, now, (),
                 f"https://arxiv.org/abs/{paper_id}", f"https://arxiv.org/pdf/{paper_id}")


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    run = SimpleNamespace(papers=[_paper()], warnings=(), failures=(), sent=[], alerts=[],
                          path=tmp_path / "state.json")
    monkeypatch.setattr(app, "fetch_all_papers", lambda *a, **kw: FetchResult(
        run.papers, run.warnings, run.failures,
    ))
    monkeypatch.setattr(app, "match_paper", lambda *a: MatchResult(5, ("graphene",), ()))
    monkeypatch.setattr(app, "_enrich_candidates", lambda ps, config: (ps, ()))
    monkeypatch.setattr(app, "_feishu_client", lambda: SimpleNamespace(
        send_digest=lambda recs, *a, **kw: run.sent.append(recs),
    ))
    monkeypatch.setattr(app, "_send_alert", lambda *a, **kw: run.alerts.append(a) or True)
    return run


def test_successful_delivery_persists_enriched_doi_once(pipeline, monkeypatch):
    original = pipeline.papers[0]
    enriched = replace(original, doi="10.1234/graphene")
    monkeypatch.setattr(app, "_enrich_candidates", lambda *a: ([enriched], ()))
    args = app._arguments(["--state", str(pipeline.path), "--no-ai"])
    assert app.run(args) == 0
    state = StateStore(pipeline.path)
    record = state.data["papers"][original.paper_id]
    assert record["doi"] == "10.1234/graphene"
    assert record["attempts"] == 1
    assert record["status"] == "sent"
    assert app.run(args) == 0
    assert len(pipeline.sent) == 1


def test_doi_learned_during_enrichment_blocks_a_previous_delivery(pipeline, monkeypatch):
    original = pipeline.papers[0]
    previous = replace(_paper("crossref:10.1234/graphene", "Previous journal title"),
                       source="Crossref", doi="10.1234/graphene")
    state = StateStore(pipeline.path)
    state.mark(previous, "sent")
    state.save()
    monkeypatch.setattr(app, "_enrich_candidates", lambda *a: (
        [replace(original, doi="10.1234/graphene")], (),
    ))
    assert app.run(app._arguments(["--state", str(pipeline.path), "--no-ai"])) == 0
    assert pipeline.sent == []
    assert StateStore(pipeline.path).status(original.paper_id) is None


def test_same_run_duplicate_learned_from_metadata_is_delivered_once(pipeline, monkeypatch):
    other = _paper("2610.00002", "Alternate paper title")
    pipeline.papers.append(other)
    monkeypatch.setattr(app, "_enrich_candidates", lambda ps, config: (
        [replace(p, doi="10.1234/graphene") for p in ps], (),
    ))
    assert app.run(app._arguments(["--state", str(pipeline.path), "--no-ai"])) == 0
    assert len(pipeline.sent) == 1
    assert len(pipeline.sent[0]) == 1
    state = StateStore(pipeline.path)
    assert all(not state.should_consider(p) for p in pipeline.papers)


@pytest.mark.parametrize("has_candidates", [False, True])
def test_resend_source_failure_never_writes_state(pipeline, monkeypatch, has_candidates):
    state = StateStore(pipeline.path)
    state.mark(pipeline.papers[0], "sent")
    state.save()
    before = pipeline.path.read_bytes()
    pipeline.failures = pipeline.warnings = ("Source temporarily unavailable",)
    if not has_candidates:
        pipeline.papers.clear()
    monkeypatch.setattr(app, "_evaluate", lambda ps, *a, **kw: (
        [Recommendation(p, 1, "Not relevant", (), "", "", True) for p in ps], (),
    ))
    args = app._arguments(["--state", str(pipeline.path), "--resend-latest", "--no-ai"])
    assert app.run(args) == 1
    assert pipeline.path.read_bytes() == before
    assert len(pipeline.alerts) == 1
    assert pipeline.sent == []


def test_missing_webhook_cannot_report_success_or_consume_candidates(pipeline, monkeypatch):
    monkeypatch.setattr(app, "_feishu_client", lambda: None)
    args = app._arguments(["--state", str(pipeline.path), "--no-ai", "--once-per-beijing-day"])
    assert app.run(args) == 1
    assert not pipeline.path.exists()
    assert pipeline.sent == []


def test_retry_deep_read_observes_delivery_window_before_daily_guard(pipeline, monkeypatch):
    state = StateStore(pipeline.path)
    state.mark_completed(app._beijing_date())
    card = {"schema": "2.0"}
    state.data["pending_deep_reads"] = {"paper": card}
    state.save()
    events = []
    monkeypatch.setattr(app, "_wait_for_delivery_window", lambda target: events.append(target))
    monkeypatch.setattr(app, "_feishu_client", lambda: SimpleNamespace(
        send_deep_read_card=lambda card: events.append(card),
    ))
    args = app._arguments(["--state", str(pipeline.path), "--once-per-beijing-day",
                           "--deliver-not-before", "19:00", "--no-ai"])
    assert app.run(args) == 0
    assert events == ["19:00", card]
    assert StateStore(pipeline.path).data["pending_deep_reads"] == {}


def test_unacknowledged_digest_remains_eligible_and_recovers(pipeline, monkeypatch):
    replies = [{"msg": "unknown status"}, {"code": 0}]
    attempts = []

    def post(url, **kwargs):
        attempts.append(kwargs)
        return httpx.Response(200, json=replies.pop(0), request=httpx.Request("POST", url))

    monkeypatch.setattr("paper_radar.feishu.httpx.post", post)
    monkeypatch.setattr(app, "_feishu_client", lambda: FeishuClient("https://example.test/webhook"))
    args = app._arguments(["--state", str(pipeline.path), "--no-ai", "--once-per-beijing-day"])
    assert app.run(args) == 1
    state = StateStore(pipeline.path)
    assert state.should_consider(pipeline.papers[0])
    assert not state.completed_on(app._beijing_date())
    assert app.run(args) == 0
    assert StateStore(pipeline.path).status(pipeline.papers[0].paper_id) == "sent"
    assert StateStore(pipeline.path).completed_on(app._beijing_date())
    assert app.run(args) == 0
    assert len(attempts) == 2


@pytest.mark.parametrize("new_deep_read", [False, True])
def test_failed_old_deep_read_cannot_block_digest_or_repeat_it(
    pipeline, monkeypatch, new_deep_read,
):
    state = StateStore(pipeline.path)
    old_card = {"paper": "old"}
    new_card = {"paper": "new"}
    state.data["pending_deep_reads"] = {"old-id": old_card}
    state.save()
    attempts = []

    def send_deep(card):
        attempts.append(card)
        if card == old_card:
            raise RuntimeError("old card rejected")

    monkeypatch.setattr(app, "_feishu_client", lambda: SimpleNamespace(
        send_digest=lambda recs, *a, **kw: pipeline.sent.append(recs),
        send_deep_read_card=send_deep,
    ))
    monkeypatch.setattr(app, "_generate_deep_read", lambda *a, **kw:
                        SimpleNamespace(paper=pipeline.papers[0]) if new_deep_read else None)
    monkeypatch.setattr(app, "build_deep_read_card", lambda _: new_card)
    args = app._arguments(["--state", str(pipeline.path), "--no-ai", "--once-per-beijing-day"])
    assert app.run(args) == 1
    assert len(pipeline.sent) == 1
    assert attempts == ([old_card, new_card] if new_deep_read else [old_card])
    state = StateStore(pipeline.path)
    assert state.completed_on(app._beijing_date())
    assert state.data["pending_deep_reads"] == {"old-id": old_card}

    # Even newly discovered papers cannot produce a second daily digest while
    # the optional card keeps failing. They remain available for tomorrow.
    pipeline.papers = [_paper("2610.00002", "A new graphene result")]
    assert app.run(args) == 1
    assert len(pipeline.sent) == 1
    assert StateStore(pipeline.path).should_consider(pipeline.papers[0])


def test_empty_day_with_failed_optional_card_remains_failed_but_digest_complete(
    pipeline, monkeypatch,
):
    state = StateStore(pipeline.path)
    state.data["pending_deep_reads"] = {"old-id": {"paper": "old"}}
    state.save()
    pipeline.papers = []

    def fail(*args):
        raise RuntimeError("old card rejected")

    monkeypatch.setattr(app, "_feishu_client", lambda: SimpleNamespace(send_deep_read_card=fail))
    args = app._arguments(["--state", str(pipeline.path), "--no-ai", "--once-per-beijing-day"])
    assert app.run(args) == 1
    assert StateStore(pipeline.path).completed_on(app._beijing_date())
    assert "old-id" in StateStore(pipeline.path).data["pending_deep_reads"]
