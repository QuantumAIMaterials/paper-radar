import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from paper_radar.models import Paper
from paper_radar.recommender import AIRecommender, _extract_json, calculate_priority


class FakeStream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self.events = events

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.events)


class FakeResponses:
    def __init__(self, response: SimpleNamespace | None = None) -> None:
        self.request = None
        self.response = response

    def create(self, **request: object) -> FakeStream:
        self.request = request
        response = self.response or SimpleNamespace(
            status="completed",
            output_text='{"evaluations": []}',
        )
        return FakeStream(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta='{"evaluations":',
                ),
                SimpleNamespace(type="response.output_text.delta", delta=" []}"),
                SimpleNamespace(type="response.completed", response=response),
            ]
        )


def make_recommender() -> AIRecommender:
    recommender = object.__new__(AIRecommender)
    recommender.model = "gpt-5.6-sol"
    recommender.fallback_models = ()
    recommender.unavailable_models = set()
    recommender.reasoning_effort = "high"
    recommender.prompt_template = "Evaluate these papers:\n\n{PAPER_INFO}"
    return recommender


def make_paper() -> Paper:
    now = datetime.now(UTC)
    return Paper(
        paper_id="2508.00001",
        title="Excitons in a two-dimensional semiconductor",
        authors=("Test Author",),
        abstract="We study excitons in a monolayer semiconductor.",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2508.00001",
        pdf_url="https://arxiv.org/pdf/2508.00001",
    )


def valid_evaluation() -> dict[str, object]:
    return {
        "paper_id": "2508.00001",
        "group_fit_score": 5,
        "novelty_score": 4,
        "method_value_score": 4,
        "evidence_score": 4,
        "study_type": "实验",
        "reason": "与课题组的二维材料激子研究直接相关。",
        "key_relevance": ["二维半导体", "激子"],
        "quality_signals": ["给出了明确实验方法", "报告了具体激子调控结果"],
        "title_zh": "二维半导体中的激子",
        "summary_zh": "本文研究单层半导体中的激子性质。该工作与二维材料光学直接相关。",
    }


def valid_deep_read() -> dict[str, object]:
    return {
        "title_zh": "二维半导体激子论文速读",
        "selection_reason": "方法具有直接实验启发，且证据链完整。",
        "one_sentence_summary": "论文结合光谱与输运测量研究单层半导体中的激子。",
        "technical_route": ["样品制备：单层材料 -> 器件加工 -> 低温测量"],
        "takeaways": ["激子响应可以被外场连续调控。"],
        "advances": ["把光谱观测与输运证据放在同一器件中比较。"],
        "limitations": ["结论目前只在一种材料和有限温区验证。"],
        "group_inspirations": ["可在组内二维器件上复现实验对照流程。"],
        "author_context": [],
    }


def test_request_uses_streaming_message_input() -> None:
    responses = FakeResponses()
    recommender = make_recommender()
    recommender.client = SimpleNamespace(responses=responses)

    result = recommender._request("Evaluate this paper", structured=True)

    assert result == '{"evaluations": []}'
    assert responses.request is not None
    assert responses.request["stream"] is True
    assert responses.request["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Evaluate this paper"}],
        }
    ]
    assert responses.request["reasoning"] == {"effort": "high"}
    assert responses.request["text"]["format"]["type"] == "json_schema"
    assert responses.request["text"]["format"]["strict"] is True


def test_request_uses_json_mode_for_retry() -> None:
    responses = FakeResponses()
    recommender = make_recommender()
    recommender.client = SimpleNamespace(responses=responses)

    recommender._request("Return JSON", structured=False)

    assert responses.request is not None
    assert responses.request["text"] == {"format": {"type": "json_object"}}


def test_request_logs_token_usage(caplog) -> None:
    response = SimpleNamespace(
        status="completed",
        output_text='{"evaluations": []}',
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=34,
            total_tokens=154,
            input_tokens_details=SimpleNamespace(cached_tokens=20),
            output_tokens_details=SimpleNamespace(reasoning_tokens=12),
        ),
    )
    responses = FakeResponses(response)
    recommender = make_recommender()
    recommender.client = SimpleNamespace(responses=responses)

    with caplog.at_level(logging.INFO, logger="paper_radar.recommender"):
        recommender._request("Evaluate this paper", structured=True)

    assert "LLM usage for paper_recommendations" in caplog.text
    assert "input=120" in caplog.text
    assert "output=34" in caplog.text
    assert "reasoning=12" in caplog.text
    assert "cached_input=20" in caplog.text
    assert "total=154" in caplog.text


def test_evaluation_cache_key_changes_with_prompt() -> None:
    paper = make_paper()
    recommender = make_recommender()
    original_key = recommender.evaluation_cache_key(
        paper,
        profile_name="2D Quantum Materials",
    )
    recommender.prompt_template += "\nNew rubric."

    updated_key = recommender.evaluation_cache_key(
        paper,
        profile_name="2D Quantum Materials",
    )

    assert updated_key != original_key


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"evaluations": []}', {"evaluations": []}),
        ("```json\n{\"evaluations\": []}\n```", {"evaluations": []}),
        ("Result: {\"evaluations\": []}", {"evaluations": []}),
        ("[]", {"evaluations": []}),
        ('"{\\"evaluations\\": []}"', {"evaluations": []}),
    ],
)
def test_extract_json_accepts_relay_variants(
    response: str, expected: dict[str, object]
) -> None:
    assert _extract_json(response) == expected


def test_evaluate_retries_invalid_structured_output_with_json_mode() -> None:
    recommender = make_recommender()
    outputs = iter(["", json.dumps({"evaluations": [valid_evaluation()]})])
    request_modes: list[bool] = []

    def fake_request(prompt: str, *, structured: bool) -> str:
        request_modes.append(structured)
        return next(outputs)

    recommender._request = fake_request
    result = recommender.evaluate([make_paper()], {})[0]

    assert request_modes == [True, False]
    assert result.used_ai is True
    assert result.title_zh == "二维半导体中的激子"
    assert result.summary_zh
    assert result.relevance_score == 3
    assert result.priority_score == 88
    assert result.reading_action == "精读"


def test_evaluate_retries_incomplete_evaluation() -> None:
    recommender = make_recommender()
    incomplete = valid_evaluation()
    incomplete["summary_zh"] = ""
    outputs = iter(
        [
            json.dumps({"evaluations": [incomplete]}),
            json.dumps([valid_evaluation()]),
        ]
    )
    request_modes: list[bool] = []

    def fake_request(prompt: str, *, structured: bool) -> str:
        request_modes.append(structured)
        return next(outputs)

    recommender._request = fake_request
    result = recommender.evaluate([make_paper()], {})[0]

    assert request_modes == [True, False]
    assert result.used_ai is True
    assert result.key_relevance == ("二维半导体", "激子")


def test_evaluate_accepts_relay_field_aliases() -> None:
    recommender = make_recommender()
    relay_evaluation = valid_evaluation()
    relay_evaluation["key_relevance_items"] = relay_evaluation.pop("key_relevance")
    relay_evaluation["chinese_title"] = relay_evaluation.pop("title_zh")
    relay_evaluation["chinese_summary"] = relay_evaluation.pop("summary_zh")
    recommender._request = lambda prompt, *, structured: json.dumps(
        {"evaluations": [relay_evaluation]}
    )

    result = recommender.evaluate([make_paper()], {})[0]

    assert result.used_ai is True
    assert result.title_zh == "二维半导体中的激子"
    assert result.summary_zh
    assert result.key_relevance == ("二维半导体", "激子")


def test_evaluate_retries_when_reason_is_not_chinese() -> None:
    recommender = make_recommender()
    english = valid_evaluation()
    english["reason"] = "Directly relevant to two-dimensional exciton research."
    outputs = iter(
        [
            json.dumps({"evaluations": [english]}),
            json.dumps({"evaluations": [valid_evaluation()]}),
        ]
    )
    request_modes: list[bool] = []

    def fake_request(prompt: str, *, structured: bool) -> str:
        request_modes.append(structured)
        return next(outputs)

    recommender._request = fake_request

    result = recommender.evaluate([make_paper()], {})[0]

    assert request_modes == [True, False]
    assert result.reason == "与课题组的二维材料激子研究直接相关。"


def test_evaluate_rejects_incomplete_json_mode_result() -> None:
    recommender = make_recommender()
    recommender._request = lambda prompt, *, structured: '{"evaluations": []}'

    with pytest.raises(ValueError, match="1 paper evaluation"):
        recommender.evaluate([make_paper()], {})


def test_generate_deep_read_uses_pdf_file_input(tmp_path) -> None:
    recommender = make_recommender()
    request: dict[str, object] = {}

    def fake_request_content(
        content,
        *,
        schema,
        schema_name,
        max_output_tokens,
    ) -> str:
        request.update(
            content=content,
            schema=schema,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        )
        return json.dumps(valid_deep_read())

    recommender._request_content = fake_request_content
    prompt_path = tmp_path / "deep_read.txt"
    prompt_path.write_text(
        "Profile: {PROFILE_NAME}\n\n{PAPER_METADATA}",
        encoding="utf-8",
    )

    result = recommender.generate_deep_read(
        make_paper(),
        profile_name="二维量子材料",
        prompt_path=prompt_path,
    )

    content = request["content"]
    assert content[0] == {
        "type": "input_file",
        "file_url": "https://arxiv.org/pdf/2508.00001",
        "detail": "low",
    }
    assert "二维量子材料" in content[1]["text"]
    assert request["schema_name"] == "paper_deep_read"
    assert request["max_output_tokens"] == 12000
    assert result.paper.paper_id == "2508.00001"
    assert result.technical_route


def test_priority_requires_direct_fit_and_actionable_value() -> None:
    assert calculate_priority(5, 4, 4, 4) == (3, 88)
    assert calculate_priority(4, 3, 3, 3) == (2, 68)
    assert calculate_priority(2, 5, 5, 5) == (1, 76)


def test_transient_failure_waits_then_recovers(monkeypatch):
    import httpx
    from openai import InternalServerError

    recommender = make_recommender()
    calls = []
    sleeps = []
    error = InternalServerError('unavailable', response=httpx.Response(
        503, request=httpx.Request('POST', 'https://example.com/responses')), body=None)

    def request(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise error
        return 'ok'

    monkeypatch.setattr(recommender, '_request_content_once', request)
    monkeypatch.setattr('paper_radar.recommender.time.sleep', sleeps.append)
    assert recommender._request_content([], schema=None, schema_name='test',
                                        max_output_tokens=100) == 'ok'
    assert sleeps == [30, 60]


def test_exhausted_503_does_not_restart_budget_in_json_mode(monkeypatch):
    import httpx
    from openai import InternalServerError

    recommender = make_recommender()
    calls = []
    sleeps = []
    error = InternalServerError('unavailable', response=httpx.Response(
        503, request=httpx.Request('POST', 'https://example.com/responses')), body=None)

    def request(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(recommender, '_request_content_once', request)
    monkeypatch.setattr('paper_radar.recommender.time.sleep', sleeps.append)
    with pytest.raises(InternalServerError):
        recommender.evaluate([make_paper()], {})
    assert len(calls) == 5
    assert sleeps == [30, 60, 120, 240]


def test_auth_failure_is_not_retried(monkeypatch):
    import httpx
    from openai import AuthenticationError

    recommender = make_recommender()
    calls = []
    error = AuthenticationError('bad key', response=httpx.Response(
        401, request=httpx.Request('POST', 'https://example.com/responses')), body=None)

    def request(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(recommender, '_request_content_once', request)
    monkeypatch.setattr('paper_radar.recommender.time.sleep',
                        lambda _: pytest.fail('must not sleep'))
    with pytest.raises(AuthenticationError):
        recommender.evaluate([make_paper()], {})
    assert len(calls) == 1


def test_interrupted_stream_restarts_with_fresh_output(monkeypatch):
    import httpx

    recommender = make_recommender()
    calls = []
    sleeps = []

    class InterruptedStream(FakeStream):
        def __iter__(self):
            yield SimpleNamespace(type='response.output_text.delta', delta='partial')
            raise httpx.ReadError('connection dropped')

    def create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return InterruptedStream([])
        return FakeStream([
            SimpleNamespace(type='response.output_text.delta', delta='{"ok":true}'),
            SimpleNamespace(type='response.completed', response=SimpleNamespace(
                status='completed', output_text='{"ok":true}')),
        ])

    recommender.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr('paper_radar.recommender.time.sleep', sleeps.append)
    result = recommender._request_content([], schema=None, schema_name='test',
                                          max_output_tokens=100)
    assert result == '{"ok":true}'
    assert sleeps == [30]


def test_model_withdrawal_switches_once_and_keeps_fallback_for_later_calls(monkeypatch):
    import httpx
    from openai import NotFoundError

    recommender = make_recommender()
    recommender.fallback_models = ('backup-a', 'backup-b')
    calls = []

    def request(*args, **kwargs):
        calls.append(recommender.model)
        if recommender.model == 'gpt-5.6-sol':
            raise NotFoundError('model withdrawn', response=httpx.Response(
                404, request=httpx.Request('POST', 'https://example.com/responses')),
                body={'code': 'model_not_found'})
        return 'ok'

    monkeypatch.setattr(recommender, '_request_content_once', request)
    monkeypatch.setattr('paper_radar.recommender.time.sleep',
                        lambda _: pytest.fail('must not retry a withdrawn model'))
    for _ in range(2):
        assert recommender._request_content([], schema=None, schema_name='test',
                                            max_output_tokens=100) == 'ok'
    assert calls == ['gpt-5.6-sol', 'backup-a', 'backup-a']
    assert recommender.model == 'backup-a'


@pytest.mark.parametrize(('status', 'code'), [
    (401, 'invalid_api_key'), (403, 'permission_denied'), (404, 'not_found'),
])
def test_unrelated_api_failures_do_not_switch_models(monkeypatch, status, code):
    import httpx
    from openai import APIStatusError

    recommender = make_recommender()
    recommender.fallback_models = ('backup',)
    calls = []

    def request(*args, **kwargs):
        calls.append(recommender.model)
        raise APIStatusError('failure', response=httpx.Response(
            status, request=httpx.Request('POST', 'https://example.com/responses')),
            body={'code': code})

    monkeypatch.setattr(recommender, '_request_content_once', request)
    with pytest.raises(APIStatusError):
        recommender.evaluate([make_paper()], {})
    assert calls == ['gpt-5.6-sol']


def test_exhausted_model_fallback_does_not_retry_in_json_mode(monkeypatch):
    import httpx
    from openai import BadRequestError

    recommender = make_recommender()
    recommender.fallback_models = ('backup-a', 'backup-b')
    calls = []

    def request(*args, **kwargs):
        calls.append(recommender.model)
        raise BadRequestError('model unavailable', response=httpx.Response(
            400, request=httpx.Request('POST', 'https://example.com/responses')),
            body={'code': 'model_not_found'})

    monkeypatch.setattr(recommender, '_request_content_once', request)
    with pytest.raises(BadRequestError):
        recommender.evaluate([make_paper()], {})
    assert calls == ['gpt-5.6-sol', 'backup-a', 'backup-b']


def test_model_switch_does_not_reset_transient_retry_budget(monkeypatch):
    import httpx
    from openai import InternalServerError, NotFoundError

    recommender = make_recommender()
    recommender.fallback_models = ('backup',)
    sleeps = []
    calls = []

    def request(*args, **kwargs):
        calls.append(recommender.model)
        request = httpx.Request('POST', 'https://example.com/responses')
        if len(calls) == 3:
            raise NotFoundError('withdrawn', response=httpx.Response(404, request=request),
                                body={'code': 'model_not_found'})
        raise InternalServerError('temporary', response=httpx.Response(503, request=request),
                                  body=None)

    monkeypatch.setattr(recommender, '_request_content_once', request)
    monkeypatch.setattr('paper_radar.recommender.time.sleep', sleeps.append)
    with pytest.raises(InternalServerError):
        recommender.evaluate([make_paper()], {})
    assert calls == ['gpt-5.6-sol'] * 3 + ['backup'] * 3
    assert sleeps == [30, 60, 120, 240]


def test_environment_fallbacks_are_opt_in_and_deduplicated(monkeypatch, tmp_path):
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('Evaluate {PAPER_INFO}')
    monkeypatch.setenv('LLM_API_KEY', 'test')
    monkeypatch.setenv('LLM_MODEL', 'primary')
    monkeypatch.setenv('LLM_FALLBACK_MODELS', ' primary, backup-a, , backup-b, backup-a ')
    recommender = AIRecommender.from_environment(prompt)
    assert recommender.fallback_models == ('backup-a', 'backup-b')
    monkeypatch.delenv('LLM_FALLBACK_MODELS')
    assert AIRecommender.from_environment(prompt).fallback_models == ()


def test_pdf_model_unavailable_is_not_repeated_in_json_mode(monkeypatch, tmp_path):
    import httpx
    from openai import NotFoundError

    recommender = make_recommender()
    calls = []
    prompt = tmp_path / 'deep-read.txt'
    prompt.write_text('Read {PAPER_METADATA}')

    def request(*args, **kwargs):
        calls.append(1)
        raise NotFoundError('model withdrawn', response=httpx.Response(
            404, request=httpx.Request('POST', 'https://example.com/responses')),
            body={'code': 'model_not_found'})

    monkeypatch.setattr(recommender, '_request_content', request)
    with pytest.raises(NotFoundError):
        recommender.generate_deep_read(make_paper(), profile_name='test', prompt_path=prompt)
    assert calls == [1]
