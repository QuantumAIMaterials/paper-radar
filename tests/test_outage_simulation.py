"""Fault injection through the real SDK; no network or real sleeping."""
import json

import httpx
import pytest
from openai import APIStatusError, OpenAI

from paper_radar.recommender import AIRecommender


def completed():
    events = [
        {'type': 'response.output_text.delta', 'delta': '{"ok":true}'},
        {'type': 'response.completed', 'response': {
            'id': 'resp_simulation', 'object': 'response', 'created_at': 0,
            'status': 'completed', 'model': 'test', 'output': [],
        }},
    ]
    return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                          content=''.join('data: ' + json.dumps(e) + '\n\n' for e in events))


def setup(monkeypatch, handler):
    recommender = object.__new__(AIRecommender)
    recommender.model = 'test'
    recommender.reasoning_effort = 'medium'
    recommender.client = OpenAI(api_key='fake-test-key', base_url='https://test.invalid/v1',
                               max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    sleeps = []
    monkeypatch.setattr('paper_radar.recommender.time.sleep', sleeps.append)
    return recommender, sleeps


def invoke(recommender):
    return recommender._request_content([], schema=None, schema_name='simulation',
                                        max_output_tokens=100)


@pytest.mark.parametrize('status', [408, 409, 429, 500, 502, 503, 504])
@pytest.mark.parametrize('failures', [1, 2, 4])
def test_http_outage_recovers(monkeypatch, status, failures):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= failures:
            return httpx.Response(status, json={'error': {'message': 'simulated outage'}})
        return completed()

    recommender, sleeps = setup(monkeypatch, handler)
    with recommender.client:
        assert invoke(recommender) == '{"ok":true}'
    assert len(calls) == failures + 1
    assert sleeps == [30, 60, 120, 240][:failures]


@pytest.mark.parametrize('status', [401, 403, 404])
def test_permanent_failure_stops_immediately(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={'error': {'message': 'permanent'}})

    recommender, sleeps = setup(monkeypatch, handler)
    with recommender.client, pytest.raises(APIStatusError):
        invoke(recommender)
    assert len(calls) == 1
    assert not sleeps


@pytest.mark.parametrize('header,expected', [('90', 90), ('99999', 300),
                                            ('garbage', 30), ('-3', 30)])
def test_rate_limit_retry_after(monkeypatch, header, expected):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={'retry-after': header},
                                  json={'error': {'message': 'rate limited'}})
        return completed()

    recommender, sleeps = setup(monkeypatch, handler)
    with recommender.client:
        assert invoke(recommender) == '{"ok":true}'
    assert sleeps == [expected]


@pytest.mark.parametrize('error', [httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError])
def test_transport_recovers(monkeypatch, error):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            raise error('simulated network fault', request=request)
        return completed()

    recommender, sleeps = setup(monkeypatch, handler)
    with recommender.client:
        assert invoke(recommender) == '{"ok":true}'
    assert len(calls) == 3
    assert sleeps == [30, 60]


def test_stream_ends_without_completion(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  content='data: {"type":"response.output_text.delta",'
                                          '"delta":"discard this partial output"}\n\n')
        return completed()

    recommender, sleeps = setup(monkeypatch, handler)
    with recommender.client:
        assert invoke(recommender) == '{"ok":true}'
    assert sleeps == [30]


def test_sustained_outage_has_finite_budget(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={'error': {'message': 'still unavailable'}})

    recommender, sleeps = setup(monkeypatch, handler)
    with recommender.client, pytest.raises(APIStatusError):
        invoke(recommender)
    assert len(calls) == 5
    assert sum(sleeps) == 450


@pytest.mark.parametrize('mode', ['dry-run', 'resend-latest'])
def test_ai_failure_does_not_mutate_preview_or_resend_state(tmp_path, monkeypatch, mode):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from paper_radar import app
    from paper_radar.models import MatchResult, Paper
    from paper_radar.sources import FetchResult
    from paper_radar.state import StateStore

    now = datetime.now(UTC)
    paper = Paper(paper_id='test:1', title='Graphene', abstract='Transport', authors=(),
                  published=now, updated=now, categories=(),
                  abstract_url='https://test.invalid/paper', pdf_url='')
    path = tmp_path / 'state.json'
    state = StateStore(path)
    state.mark(paper, 'deferred')
    state.save()
    before = path.read_bytes()
    monkeypatch.setattr(app, 'fetch_all_papers',
                        lambda *a, **kw: FetchResult(papers=[paper], warnings=()))
    monkeypatch.setattr(app, 'match_paper', lambda *a: MatchResult(5, ('graphene',), ()))
    monkeypatch.setattr(app, '_enrich_candidates', lambda ps, c: (ps, ()))

    evaluation_calls = []

    def fail(*a, **kw):
        evaluation_calls.append(a)
        raise RuntimeError('injected outage')

    recommender = SimpleNamespace(evaluate=fail, evaluation_cache_key=lambda *a, **kw: 'key')
    monkeypatch.setattr(app.AIRecommender, 'from_environment', lambda p: recommender)
    monkeypatch.setattr(app, '_send_alert', lambda *a, **kw: pytest.fail('unexpected message'))
    monkeypatch.setattr(app, '_feishu_client', lambda: SimpleNamespace(
        send_digest=lambda *a, **kw: pytest.fail('unexpected delivery'),
    ))
    args = app._arguments(['--state', str(path), '--' + mode])
    assert app.run(args) == 1
    assert path.read_bytes() == before
    assert len(evaluation_calls) == 1
