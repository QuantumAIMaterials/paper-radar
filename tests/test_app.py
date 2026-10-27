from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx

from paper_radar import app
from paper_radar.models import MatchResult, Paper
from paper_radar.semantic_scholar import EnrichmentError
from paper_radar.sources import FetchResult
from paper_radar.state import StateStore


def _paper() -> Paper:
    now = datetime.now(UTC)
    return Paper(
        paper_id="2608.00001",
        title="Graphene transport",
        authors=("Test Author",),
        abstract="Quantum transport in graphene.",
        published=now,
        updated=now,
        categories=("cond-mat.mes-hall",),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )


def test_enrichment_warning_preserves_successful_metadata(monkeypatch) -> None:
    paper = _paper()
    enriched = replace(paper, citation_count=7)
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    error = httpx.HTTPStatusError("limited", request=response.request, response=response)

    def fail(*args):
        raise EnrichmentError([enriched], error)

    monkeypatch.setattr(app, "enrich_papers", fail)
    papers, notices = app._enrich_candidates([paper], app.load_config(app._arguments([]).config))
    assert papers == [enriched]
    assert "HTTP 429" in notices[0]
    assert "已保留现有数据" in notices[0]


def test_run_filters_previously_sent_cross_source_duplicate(tmp_path, monkeypatch) -> None:
    paper = _paper()
    path = tmp_path / "seen.json"
    state = StateStore(path)
    state.mark(paper, "sent")
    state.save()
    before = path.read_bytes()
    journal = replace(paper, paper_id="crossref:10.1234/graphene", source="Crossref")
    monkeypatch.setattr(app, "fetch_all_papers", lambda *a, **kw: FetchResult([journal], ()))

    def unexpected_enrichment(*args):
        raise AssertionError("An already sent paper must not be evaluated or delivered again")

    monkeypatch.setattr(app, "_enrich_candidates", unexpected_enrichment)
    assert app.run(app._arguments(["--state", str(path), "--dry-run", "--no-ai"])) == 0
    assert path.read_bytes() == before


def test_resend_latest_includes_seen_paper_without_changing_state(
    tmp_path, monkeypatch
) -> None:
    paper = _paper()
    state_path = tmp_path / "seen.json"
    state = StateStore(state_path)
    state.mark(paper, "sent")
    state.save()
    original_state = state_path.read_bytes()
    previewed = []

    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(papers=[paper], warnings=()),
    )
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(3, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_enrich_candidates",
        lambda candidates, config: (candidates, ()),
    )
    monkeypatch.setattr(
        app,
        "_print_preview",
        lambda recommendations, matches, notices, **kwargs: previewed.extend(
            recommendations
        ),
    )

    args = app._arguments(
        [
            "--state",
            str(state_path),
            "--dry-run",
            "--no-ai",
            "--resend-latest",
        ]
    )

    assert app.run(args) == 0
    assert [item.paper.paper_id for item in previewed] == [paper.paper_id]
    assert state_path.read_bytes() == original_state


def test_run_enriches_only_ranked_candidates(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    papers = [
        Paper(
            paper_id=f"paper-{index:02d}",
            title=f"Graphene transport {index}",
            authors=("Author",),
            abstract="Quantum transport in graphene.",
            published=now + timedelta(seconds=index),
            updated=now + timedelta(seconds=index),
            categories=("cond-mat.mes-hall",),
            abstract_url="https://example.test",
            pdf_url="https://example.test/paper.pdf",
        )
        for index in range(25)
    ]
    fetch_kwargs = []
    enriched_ids = []

    def fake_fetch_all_papers(config, **kwargs):
        fetch_kwargs.append(kwargs)
        return FetchResult(papers=papers, warnings=())

    def fake_enrich_papers(candidates, semantic_config):
        enriched_ids.extend(paper.paper_id for paper in candidates)
        return candidates

    monkeypatch.setattr(app, "fetch_all_papers", fake_fetch_all_papers)
    monkeypatch.setattr(app, "enrich_papers", fake_enrich_papers)
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(5, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_print_preview",
        lambda recommendations, matches, notices, **kwargs: None,
    )

    args = app._arguments(["--state", str(tmp_path / "seen.json"), "--dry-run", "--no-ai"])

    assert app.run(args) == 0
    assert fetch_kwargs == [{"enrich_semantic_scholar": False}]
    assert len(enriched_ids) == 20
    assert enriched_ids[0] == "paper-24"
    assert enriched_ids[-1] == "paper-05"


def test_optional_metadata_failure_does_not_send_standalone_alert(
    tmp_path, monkeypatch
) -> None:
    paper = _paper()
    alerts = []
    waits = []

    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(papers=[paper], warnings=()),
    )
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(5, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_enrich_candidates",
        lambda candidates, config: (
            candidates,
            ("Semantic Scholar 附加元数据暂不可用（HTTP 429）。",),
        ),
    )
    monkeypatch.setattr(
        app,
        "_evaluate",
        lambda candidates, matches, **kwargs: (
            [
                app.Recommendation(
                    paper=paper,
                    relevance_score=1,
                    reason="not relevant",
                    key_relevance=("graphene",),
                    title_zh="test",
                    summary_zh="test",
                    used_ai=True,
                )
            ],
            (),
        ),
    )
    monkeypatch.setattr(app, "_feishu_client", lambda: object())
    monkeypatch.setattr(
        app,
        "_send_alert",
        lambda title, message, **kwargs: alerts.append((title, message)),
    )
    monkeypatch.setattr(
        app,
        "_wait_for_delivery_window",
        lambda target: waits.append(target),
    )

    args = app._arguments(["--state", str(tmp_path / "seen.json")])

    assert app.run(args) == 0
    assert alerts == []
    assert waits == []


def test_delivery_window_waits_until_same_beijing_day() -> None:
    seconds = app._seconds_until_beijing_time(
        "19:00",
        now=datetime(2026, 9, 2, 10, 59, 30, tzinfo=UTC),
    )

    assert seconds == 30
    assert (
        app._seconds_until_beijing_time(
            "19:00",
            now=datetime(2026, 9, 2, 11, 0, 1, tzinfo=UTC),
        )
        == 0
    )


def test_start_window_skips_only_triggers_beyond_wait_limit(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(app.time_module, "sleep", sleeps.append)
    monkeypatch.setattr(app, "_seconds_until_beijing_time", lambda target: 346 * 60)

    assert not app._wait_for_start_window("19:00", max_wait_minutes=345)
    assert sleeps == []

    monkeypatch.setattr(app, "_seconds_until_beijing_time", lambda target: 344 * 60)

    assert app._wait_for_start_window("19:00", max_wait_minutes=345)
    assert sleeps == [344 * 60]


def test_once_per_day_skips_after_successful_run(tmp_path, monkeypatch) -> None:
    run_date = datetime(2026, 9, 9, tzinfo=UTC).date()
    state_path = tmp_path / "seen.json"
    state = StateStore(state_path)
    state.mark_completed(run_date)
    state.save()

    monkeypatch.setattr(app, "_beijing_date", lambda: run_date)
    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fetch")),
    )
    args = app._arguments(
        ["--state", str(state_path), "--once-per-beijing-day", "--no-ai"]
    )

    assert app.run(args) == 0


def test_empty_successful_run_records_beijing_date(tmp_path, monkeypatch) -> None:
    run_date = datetime(2026, 9, 9, tzinfo=UTC).date()
    state_path = tmp_path / "seen.json"
    monkeypatch.setattr(app, "_beijing_date", lambda: run_date)
    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(papers=[], warnings=()),
    )
    args = app._arguments(
        ["--state", str(state_path), "--once-per-beijing-day", "--no-ai"]
    )

    assert app.run(args) == 0
    assert StateStore(state_path).completed_on(run_date)


def test_empty_degraded_run_remains_eligible_for_fallback(tmp_path, monkeypatch) -> None:
    run_date = datetime(2026, 9, 9, tzinfo=UTC).date()
    state_path = tmp_path / "seen.json"
    monkeypatch.setattr(app, "_beijing_date", lambda: run_date)
    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(
            papers=[],
            warnings=("arXiv is temporarily unavailable",),
            failures=("arXiv is temporarily unavailable",),
        ),
    )
    monkeypatch.setattr(app, "_send_alert", lambda *args, **kwargs: None)
    args = app._arguments(
        ["--state", str(state_path), "--once-per-beijing-day", "--no-ai"]
    )

    assert app.run(args) == 1
    assert not StateStore(state_path).completed_on(run_date)


def test_repeated_degraded_run_sends_only_one_daily_alert(tmp_path, monkeypatch) -> None:
    run_date = datetime(2026, 9, 9, tzinfo=UTC).date()
    state_path = tmp_path / "seen.json"
    alerts = []
    warning = "arXiv and its fallback are unavailable"

    monkeypatch.setattr(app, "_beijing_date", lambda: run_date)
    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(
            papers=[],
            warnings=(warning,),
            failures=(warning,),
        ),
    )
    monkeypatch.setattr(
        app,
        "_send_alert",
        lambda *args, **kwargs: alerts.append(args) or True,
    )
    args = app._arguments(
        ["--state", str(state_path), "--once-per-beijing-day", "--no-ai"]
    )

    assert app.run(args) == 1
    assert app.run(args) == 1
    assert len(alerts) == 1


def test_degraded_run_with_candidates_is_not_recorded_as_complete(
    tmp_path, monkeypatch
) -> None:
    run_date = datetime(2026, 9, 9, tzinfo=UTC).date()
    state_path = tmp_path / "seen.json"
    paper = _paper()
    warning = "arXiv and its fallback are unavailable"

    monkeypatch.setattr(app, "_beijing_date", lambda: run_date)
    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(
            papers=[paper],
            warnings=(warning,),
            failures=(warning,),
        ),
    )
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(5, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_enrich_candidates",
        lambda candidates, config: (candidates, ()),
    )
    monkeypatch.setattr(
        app,
        "_print_preview",
        lambda recommendations, matches, notices, **kwargs: None,
    )

    args = app._arguments(
        [
            "--state",
            str(state_path),
            "--once-per-beijing-day",
            "--dry-run",
            "--no-ai",
        ]
    )

    assert app.run(args) == 1
    assert not StateStore(state_path).completed_on(run_date)


def test_calibration_caps_high_priority_recommendations() -> None:
    papers = [
        Paper(
            paper_id=f"paper-{index}",
            title=f"Paper {index}",
            authors=("Author",),
            abstract="Graphene",
            published=datetime(2026, 8, 14, index, tzinfo=UTC),
            updated=datetime(2026, 8, 14, index, tzinfo=UTC),
            categories=(),
            abstract_url="https://example.test",
            pdf_url="https://example.test/paper.pdf",
        )
        for index in range(5)
    ]
    recommendations = [
        app.Recommendation(
            paper=paper,
            relevance_score=3,
            reason="相关",
            key_relevance=("石墨烯",),
            title_zh="论文",
            summary_zh="摘要",
            used_ai=True,
            priority_score=90 - index,
            reading_action="精读",
        )
        for index, paper in enumerate(papers)
    ]
    matches = {
        paper.paper_id: MatchResult(3, ("graphene",), ()) for paper in papers
    }

    calibrated = app._calibrate_priorities(
        recommendations,
        matches,
        max_high_priority=3,
    )

    assert [item.relevance_score for item in calibrated] == [3, 3, 3, 2, 2]
    assert calibrated[-1].reading_action == "速读"


def test_delivery_selection_caps_non_preferred_machine_learning() -> None:
    papers = [
        Paper(
            paper_id=f"paper-{index}",
            title=f"Paper {index}",
            authors=("Author",),
            abstract="Physics AI",
            published=datetime(2026, 8, 14, index, tzinfo=UTC),
            updated=datetime(2026, 8, 14, index, tzinfo=UTC),
            categories=(),
            abstract_url="https://example.test",
            pdf_url="https://example.test/paper.pdf",
        )
        for index in range(6)
    ]
    recommendations = [
        app.Recommendation(
            paper=paper,
            relevance_score=2,
            reason="相关",
            key_relevance=("凝聚态物理",),
            title_zh="论文",
            summary_zh="摘要",
            used_ai=True,
        )
        for paper in papers
    ]
    matches = {
        paper.paper_id: MatchResult(
            10,
            ("large language model",) if index < 2 else ("machine learning",),
            (),
            ("condensed matter",),
            ("large language model",) if index < 2 else (),
        )
        for index, paper in enumerate(papers)
    }

    selected, quota_skipped = app._select_for_delivery(
        recommendations,
        matches,
        minimum_relevance=2,
        max_papers=10,
        max_non_preferred_papers=2,
    )

    assert [item.paper.paper_id for item in selected] == [
        "paper-0",
        "paper-1",
        "paper-2",
        "paper-3",
    ]
    assert quota_skipped == {"paper-4", "paper-5"}

    traditional_matches = {
        paper.paper_id: MatchResult(
            5,
            ("machine learning",),
            (),
            ("condensed matter",),
        )
        for paper in papers
    }
    selected, quota_skipped = app._select_for_delivery(
        recommendations,
        traditional_matches,
        minimum_relevance=2,
        max_papers=10,
        max_non_preferred_papers=2,
    )

    assert [item.paper.paper_id for item in selected] == ["paper-0", "paper-1"]
    assert quota_skipped == {"paper-2", "paper-3", "paper-4", "paper-5"}


def test_delivery_selection_preserves_profiles_without_preferred_terms() -> None:
    paper = _paper()
    recommendation = app.Recommendation(
        paper=paper,
        relevance_score=2,
        reason="相关",
        key_relevance=("二维材料",),
        title_zh="论文",
        summary_zh="摘要",
        used_ai=True,
    )
    matches = {paper.paper_id: MatchResult(5, ("graphene",), ())}

    selected, quota_skipped = app._select_for_delivery(
        [recommendation],
        matches,
        minimum_relevance=2,
        max_papers=10,
        max_non_preferred_papers=10,
    )

    assert selected == [recommendation]
    assert quota_skipped == set()


def test_deep_read_candidate_requires_inspiring_ai_evaluation() -> None:
    paper = _paper()
    routine = app.Recommendation(
        paper=paper,
        relevance_score=3,
        reason="相关",
        key_relevance=("石墨烯",),
        title_zh="常规论文",
        summary_zh="摘要",
        used_ai=True,
        priority_score=86,
        method_value_score=3,
        evidence_score=4,
    )
    inspiring = app.Recommendation(
        paper=paper,
        relevance_score=3,
        reason="高度相关",
        key_relevance=("石墨烯",),
        title_zh="启发性论文",
        summary_zh="摘要",
        used_ai=True,
        priority_score=84,
        method_value_score=4,
        evidence_score=3,
    )

    selected = app._select_deep_read_candidate(
        [routine, inspiring],
        enabled=True,
        minimum_priority_score=82,
    )

    assert selected is inspiring
    assert (
        app._select_deep_read_candidate(
            [inspiring],
            enabled=False,
            minimum_priority_score=82,
        )
        is None
    )


def test_deep_read_skips_publisher_pdf_for_next_arxiv_candidate() -> None:
    publisher_paper = replace(
        _paper(),
        paper_id="publisher",
        source="Crossref",
        abstract_url="https://pubs.acs.org/doi/10.1021/example",
        pdf_url="https://pubs.acs.org/doi/pdf/10.1021/example",
    )
    arxiv_paper = replace(_paper(), paper_id="arxiv")

    def recommendation(paper: Paper) -> app.Recommendation:
        return app.Recommendation(
            paper=paper,
            relevance_score=3,
            reason="高度相关",
            key_relevance=("石墨烯",),
            title_zh="启发性论文",
            summary_zh="摘要",
            used_ai=True,
            priority_score=84,
            method_value_score=4,
            evidence_score=3,
        )

    selected = app._select_deep_read_candidate(
        [recommendation(publisher_paper), recommendation(arxiv_paper)],
        enabled=True,
        minimum_priority_score=82,
    )

    assert selected is not None
    assert selected.paper is arxiv_paper


def test_evaluate_reuses_cached_ai_recommendations(tmp_path) -> None:
    cached_paper = _paper()
    fresh_paper = Paper(
        paper_id="2608.00002",
        title="Fresh graphene transport",
        authors=cached_paper.authors,
        abstract=cached_paper.abstract,
        published=cached_paper.published,
        updated=cached_paper.updated,
        categories=cached_paper.categories,
        abstract_url="https://arxiv.org/abs/2608.00002",
        pdf_url="https://arxiv.org/pdf/2608.00002",
    )
    matches = {
        cached_paper.paper_id: MatchResult(3, ("graphene",), ()),
        fresh_paper.paper_id: MatchResult(3, ("graphene",), ()),
    }
    state = StateStore(tmp_path / "seen.json")
    state.cache_ai_evaluation(
        "cache:2608.00001",
        app.Recommendation(
            paper=cached_paper,
            relevance_score=3,
            reason="cached",
            key_relevance=("graphene",),
            title_zh="cached",
            summary_zh="cached",
            used_ai=True,
            priority_score=88,
        ),
    )

    class FakeRecommender:
        def __init__(self) -> None:
            self.calls = []

        def evaluation_cache_key(self, paper, *, profile_name):
            return f"cache:{paper.paper_id}"

        def evaluate(self, papers, matches):
            self.calls.append([paper.paper_id for paper in papers])
            return [
                app.Recommendation(
                    paper=paper,
                    relevance_score=2,
                    reason="fresh",
                    key_relevance=("graphene",),
                    title_zh="fresh",
                    summary_zh="fresh",
                    used_ai=True,
                    priority_score=68,
                )
                for paper in papers
            ]

    recommender = FakeRecommender()

    recommendations, notices = app._evaluate(
        [cached_paper, fresh_paper],
        matches,
        recommender=recommender,
        state=state,
        profile_name="test",
    )

    assert notices == ()
    assert recommender.calls == [["2608.00002"]]
    assert [recommendation.reason for recommendation in recommendations] == [
        "cached",
        "fresh",
    ]
    assert state.cached_ai_evaluation(fresh_paper, "cache:2608.00002") is not None


def test_delivery_limit_marks_remaining_paper_deferred(tmp_path, monkeypatch) -> None:
    older = _paper()
    newer = Paper(
        paper_id="2608.00002",
        title="New graphene transport",
        authors=older.authors,
        abstract=older.abstract,
        published=older.published + timedelta(seconds=1),
        updated=older.updated,
        categories=older.categories,
        abstract_url="https://arxiv.org/abs/2608.00002",
        pdf_url="https://arxiv.org/pdf/2608.00002",
    )

    class FakeClient:
        def send_digest(
            self,
            recommendations,
            matches,
            *,
            notices=(),
            digest_title="",
            digest_intro="",
        ) -> None:
            return None

    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config, **kwargs: FetchResult(papers=[newer, older], warnings=()),
    )
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(3, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_enrich_candidates",
        lambda candidates, config: (candidates, ()),
    )
    monkeypatch.setattr(app, "_feishu_client", lambda: FakeClient())
    state_path = tmp_path / "seen.json"
    args = app._arguments(
        ["--state", str(state_path), "--no-ai", "--max-papers", "1"]
    )

    assert app.run(args) == 0

    state = StateStore(state_path)
    assert state.status(newer.paper_id) == "sent"
    assert state.status(older.paper_id) == "deferred"
    assert state.should_consider(older.paper_id)


def test_ai_failure_defers_then_recovers_without_duplicate_delivery(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from paper_radar.models import Recommendation

    paper = _paper()
    state_path = tmp_path / 'state.json'
    run_date = datetime(2026, 9, 30, tzinfo=UTC).date()
    sent = []
    alerts = []
    failing = True

    def evaluate(*args, **kwargs):
        if failing:
            raise RuntimeError('temporary outage')
        return [Recommendation(paper=paper, relevance_score=3, reason='相关',
                               key_relevance=(), title_zh='石墨烯输运', summary_zh='研究量子输运。',
                               used_ai=True)]

    recommender = SimpleNamespace(evaluate=evaluate,
                                  evaluation_cache_key=lambda *a, **kw: 'cache')
    monkeypatch.setattr(app, '_beijing_date', lambda: run_date)
    monkeypatch.setattr(app, 'fetch_all_papers',
                        lambda *a, **kw: FetchResult(papers=[paper], warnings=()))
    monkeypatch.setattr(app, 'match_paper', lambda *a: MatchResult(5, ('graphene',), ()))
    monkeypatch.setattr(app, '_enrich_candidates', lambda ps, c: (ps, ()))
    monkeypatch.setattr(app.AIRecommender, 'from_environment', lambda p: recommender)
    monkeypatch.setattr(app, '_send_alert', lambda *a, **kw: alerts.append(a) or True)
    monkeypatch.setattr(app, '_feishu_client', lambda: SimpleNamespace(
        send_digest=lambda *a, **kw: sent.append(a)))
    monkeypatch.setattr(app, '_generate_deep_read', lambda *a, **kw: None)
    args = app._arguments(['--state', str(state_path), '--once-per-beijing-day'])
    assert app.run(args) == 1
    assert app.run(args) == 1
    state = StateStore(state_path)
    assert state.status(paper.paper_id) == 'deferred'
    assert not state.completed_on(run_date)
    assert len(alerts) == 1
    assert not sent
    failing = False
    assert app.run(args) == 0
    assert StateStore(state_path).completed_on(run_date)
    assert StateStore(state_path).status(paper.paper_id) == 'sent'
    assert app.run(args) == 0
    assert len(sent) == 1


def test_pending_deep_read_survives_failure_and_retries_before_daily_guard(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    path = tmp_path / 'state.json'
    state = StateStore(path)
    state.mark(_paper(), 'sent')
    state.mark_completed(app._beijing_date())
    card = {'schema': '2.0', 'body': {'elements': []}}
    state.data['pending_deep_reads'] = {'paper': card}
    state.save()
    attempts = []
    failing = True

    def send(card):
        attempts.append(card)
        if failing:
            raise RuntimeError('temporary failure')

    monkeypatch.setattr(app, '_feishu_client', lambda: SimpleNamespace(send_deep_read_card=send))
    monkeypatch.setattr(app, 'fetch_all_papers',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('digest rerun')))
    args = app._arguments(['--state', str(path), '--once-per-beijing-day'])
    assert app.run(args) == 1
    assert StateStore(path).data['pending_deep_reads'] == {'paper': card}
    assert StateStore(path).status(_paper().paper_id) == 'sent'
    failing = False
    assert app.run(args) == 0
    assert StateStore(path).data['pending_deep_reads'] == {}
    assert app.run(args) == 0
    assert len(attempts) == 2


def test_pending_deep_read_not_sent_by_preview(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    state = StateStore(path)
    state.mark_completed(app._beijing_date())
    state.data['pending_deep_reads'] = {'paper': {'schema': '2.0'}}
    state.save()
    original = path.read_bytes()
    monkeypatch.setattr(app, '_feishu_client',
                        lambda: (_ for _ in ()).throw(AssertionError('preview sent message')))
    args = app._arguments(['--state', str(path), '--once-per-beijing-day', '--dry-run'])
    assert app.run(args) == 0
    assert path.read_bytes() == original


def test_digest_sent_then_deep_read_failure_persists_card(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from paper_radar.models import Recommendation

    paper = _paper()
    path = tmp_path / 'state.json'
    sent = []
    failing = True
    card = {'schema': '2.0', 'body': {'elements': []}}
    recommendation = Recommendation(paper, 3, '相关', (), '标题', '摘要', True)
    monkeypatch.setattr(app, 'fetch_all_papers',
                        lambda *a, **kw: FetchResult(papers=[paper], warnings=()))
    monkeypatch.setattr(app, 'match_paper', lambda *a: MatchResult(5, ('graphene',), ()))
    monkeypatch.setattr(app, '_enrich_candidates', lambda ps, c: (ps, ()))
    monkeypatch.setattr(app.AIRecommender, 'from_environment', lambda p: object())
    monkeypatch.setattr(app, '_evaluate', lambda *a, **kw: ([recommendation], ()))
    monkeypatch.setattr(app, '_generate_deep_read',
                        lambda *a, **kw: SimpleNamespace(paper=paper))
    monkeypatch.setattr(app, 'build_deep_read_card', lambda d: card)

    def deep_send(c):
        # Verify durable digest status and queued content exist BEFORE sending.
        disk = StateStore(path)
        assert disk.status(paper.paper_id) == 'sent'
        assert disk.data['pending_deep_reads'][paper.paper_id] == card
        if failing:
            raise RuntimeError('rejected')
        sent.append('deep')

    monkeypatch.setattr(app, '_feishu_client', lambda: SimpleNamespace(
        send_digest=lambda *a, **kw: sent.append('digest'), send_deep_read_card=deep_send))
    args = app._arguments(['--state', str(path), '--once-per-beijing-day'])
    assert app.run(args) == 1
    assert sent == ['digest']
    assert not StateStore(path).completed_on(app._beijing_date())
    failing = False
    assert app.run(args) == 0
    assert sent == ['digest', 'deep']
    assert app.run(args) == 0
    assert sent == ['digest', 'deep']
