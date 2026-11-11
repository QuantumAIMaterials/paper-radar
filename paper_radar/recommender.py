from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

from paper_radar.models import DeepRead, MatchResult, Paper, Recommendation

LOGGER = logging.getLogger(__name__)


def _model_unavailable(error: Exception) -> bool:
    return (
        isinstance(error, APIStatusError)
        and error.status_code in {400, 403, 404}
        and error.code in {"model_not_found", "model_not_available"}
    )


def _transient_error(error: Exception) -> bool:
    return (
        isinstance(error, (APIConnectionError, httpx.TransportError))
        or isinstance(error, APIStatusError)
        and (error.status_code in {408, 409, 429} or error.status_code >= 500)
        or isinstance(error, RuntimeError)
        and str(error).startswith("AI response was not completed: stream ended early")
    )


RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "group_fit_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "novelty_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "method_value_score": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                    },
                    "evidence_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "study_type": {
                        "type": "string",
                        "enum": ["实验", "理论", "计算", "综述", "混合", "其他"],
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "A concise Simplified Chinese explanation of how the paper "
                            "relates to the laboratory"
                        ),
                    },
                    "key_relevance": {"type": "array", "items": {"type": "string"}},
                    "quality_signals": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "title_zh": {"type": "string"},
                    "summary_zh": {"type": "string"},
                },
                "required": [
                    "paper_id",
                    "group_fit_score",
                    "novelty_score",
                    "method_value_score",
                    "evidence_score",
                    "study_type",
                    "reason",
                    "key_relevance",
                    "quality_signals",
                    "title_zh",
                    "summary_zh",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}

DEEP_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title_zh": {"type": "string"},
        "selection_reason": {"type": "string"},
        "one_sentence_summary": {"type": "string"},
        "technical_route": {"type": "array", "items": {"type": "string"}},
        "takeaways": {"type": "array", "items": {"type": "string"}},
        "advances": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "group_inspirations": {"type": "array", "items": {"type": "string"}},
        "author_context": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title_zh",
        "selection_reason",
        "one_sentence_summary",
        "technical_route",
        "takeaways",
        "advances",
        "limitations",
        "group_inspirations",
        "author_context",
    ],
    "additionalProperties": False,
}


def calculate_priority(
    group_fit_score: int,
    novelty_score: int,
    method_value_score: int,
    evidence_score: int,
) -> tuple[int, int]:
    priority_score = round(
        (
            group_fit_score * 40
            + novelty_score * 25
            + method_value_score * 20
            + evidence_score * 15
        )
        / 5
    )
    if (
        group_fit_score >= 4
        and priority_score >= 78
        and (novelty_score >= 4 or method_value_score >= 4)
        and evidence_score >= 3
    ):
        relevance_score = 3
    elif group_fit_score >= 3 and priority_score >= 50:
        relevance_score = 2
    else:
        relevance_score = 1
    return relevance_score, priority_score


def _reading_action(relevance_score: int) -> str:
    return {3: "精读", 2: "速读", 1: "收藏"}[relevance_score]


def fallback_recommendation(paper: Paper, match: MatchResult) -> Recommendation:
    group_fit_score = 5 if len(match.core_terms) >= 2 else 4
    novelty_score = 2
    method_value_score = 3 if len(match.supporting_terms) >= 2 else 2
    evidence_score = 2
    relevance_score, priority_score = calculate_priority(
        group_fit_score,
        novelty_score,
        method_value_score,
        evidence_score,
    )
    matched = match.matched_terms[:6]
    reason = "命中本雷达关注的核心与交叉研究关键词：" + ", ".join(matched)
    return Recommendation(
        paper=paper,
        relevance_score=relevance_score,
        reason=reason,
        key_relevance=matched,
        title_zh="",
        summary_zh="",
        used_ai=False,
        priority_score=priority_score,
        group_fit_score=group_fit_score,
        novelty_score=novelty_score,
        method_value_score=method_value_score,
        evidence_score=evidence_score,
        study_type="未判断",
        reading_action=_reading_action(relevance_score),
        quality_signals=("AI 不可用，仅完成关键词相关性判断",),
    )


def _extract_json(value: str) -> dict[str, Any]:
    text = value.strip()
    if not text:
        raise ValueError("AI response was empty")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as original_error:
        parsed = None
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            raise ValueError("AI response did not contain valid JSON") from original_error

    # Some compatible relays JSON-encode the result twice or return the array directly.
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        parsed = json.loads(parsed.strip())
    if isinstance(parsed, list):
        parsed = {"evaluations": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("AI response must be a JSON object")
    return parsed


def _contains_chinese(value: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", value) is not None


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _usage_value(value: Any) -> str:
    return str(value) if value is not None else "unknown"


def _log_response_usage(response: Any, schema_name: str) -> None:
    usage = _get_field(response, "usage")
    if usage is None:
        return

    input_details = _get_field(usage, "input_tokens_details")
    output_details = _get_field(usage, "output_tokens_details")
    LOGGER.info(
        "LLM usage for %s: input=%s output=%s reasoning=%s cached_input=%s total=%s",
        schema_name,
        _usage_value(_get_field(usage, "input_tokens")),
        _usage_value(_get_field(usage, "output_tokens")),
        _usage_value(_get_field(output_details, "reasoning_tokens")),
        _usage_value(_get_field(input_details, "cached_tokens")),
        _usage_value(_get_field(usage, "total_tokens")),
    )


class AIRecommender:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        reasoning_effort: str,
        prompt_path: Path,
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 120.0, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.fallback_models = tuple(dict.fromkeys(
            item.strip() for item in fallback_models if item.strip() and item.strip() != model
        ))
        self.unavailable_models: set[str] = set()
        self.reasoning_effort = reasoning_effort
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    @classmethod
    def from_environment(cls, prompt_path: Path) -> AIRecommender | None:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            model=os.getenv("LLM_MODEL", "gpt-6.1-sol").strip(),
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low").strip(),
            prompt_path=prompt_path,
            fallback_models=tuple(os.getenv("LLM_FALLBACK_MODELS", "").split(",")),
        )

    def _request_content(self, *args: Any, **kwargs: Any) -> str:
        # Own the retry budget, including interrupted streams; SDK retries are disabled.
        delays = (30, 60, 120, 240)
        attempt = 0
        while True:
            try:
                return self._request_content_once(*args, **kwargs)
            except Exception as error:
                exhausted_transient = _transient_error(error) and attempt == len(delays)
                if _model_unavailable(error) or exhausted_transient:
                    self.unavailable_models.add(self.model)
                    replacement = next((
                        model for model in self.fallback_models
                        if model not in self.unavailable_models
                    ), None)
                    if replacement is None:
                        raise
                    LOGGER.warning(
                        "AI model %s failed (%s); switching to configured fallback %s",
                        self.model, type(error).__name__, replacement,
                    )
                    self.model = replacement
                    continue
                if not _transient_error(error) or attempt == len(delays):
                    raise
                delay = delays[attempt]
                if isinstance(error, APIStatusError):
                    try:
                        retry_after = float(error.response.headers.get("retry-after", "0"))
                        delay = max(delay, min(retry_after, 300))
                    except ValueError:
                        pass
                LOGGER.warning(
                    "Temporary AI failure (%s); retrying in %ss (%s/%s)",
                    type(error).__name__, delay, attempt + 1, len(delays),
                )
                time.sleep(delay)
                attempt += 1

    def _request_content_once(
        self,
        content: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None,
        schema_name: str,
        max_output_tokens: int,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
            "stream": True,
        }
        if schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            }
        else:
            request["text"] = {"format": {"type": "json_object"}}
        output_parts: list[str] = []
        refusal_parts: list[str] = []
        completed_response = None
        with self.client.responses.create(**request) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    output_parts.append(event.delta)
                elif event.type == "response.refusal.delta":
                    refusal_parts.append(event.delta)
                elif event.type == "response.completed":
                    completed_response = event.response

        if refusal_parts:
            raise RuntimeError("AI refused the paper evaluation request")
        if completed_response is None or completed_response.status != "completed":
            status = getattr(completed_response, "status", "stream ended early")
            raise RuntimeError(f"AI response was not completed: {status}")
        output_text = "".join(output_parts) or completed_response.output_text
        if not output_text.strip():
            raise ValueError("AI response contained no output text")
        _log_response_usage(completed_response, schema_name)
        return output_text

    def evaluation_cache_key(self, paper: Paper, *, profile_name: str) -> str:
        payload = {
            "version": 1,
            "profile_name": profile_name,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_sha256": hashlib.sha256(
                self.prompt_template.encode("utf-8")
            ).hexdigest(),
            "paper_id": paper.paper_id,
            "paper_prompt_sha256": hashlib.sha256(
                paper.prompt_text().encode("utf-8")
            ).hexdigest(),
        }
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _request(self, prompt: str, *, structured: bool) -> str:
        return self._request_content(
            [{"type": "input_text", "text": prompt}],
            schema=RECOMMENDATION_SCHEMA if structured else None,
            schema_name="paper_recommendations",
            max_output_tokens=10000,
        )

    @staticmethod
    def _parse_recommendations(
        output_text: str,
        papers: list[Paper],
    ) -> dict[str, Recommendation]:
        raw = _extract_json(output_text)
        evaluations = raw.get("evaluations")
        if not isinstance(evaluations, list):
            raise ValueError("AI response does not contain an evaluations array")

        by_id = {paper.paper_id: paper for paper in papers}
        recommendations: dict[str, Recommendation] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            paper_id = str(evaluation.get("paper_id", "")).strip()
            paper = by_id.get(paper_id)
            if paper is None:
                continue
            try:
                dimensions = tuple(
                    int(evaluation.get(field, 0))
                    for field in (
                        "group_fit_score",
                        "novelty_score",
                        "method_value_score",
                        "evidence_score",
                    )
                )
            except (TypeError, ValueError):
                continue
            if any(score not in {1, 2, 3, 4, 5} for score in dimensions):
                continue
            group_fit, novelty, method_value, evidence = dimensions
            relevance_score, priority_score = calculate_priority(*dimensions)

            relevance = evaluation.get("key_relevance") or evaluation.get(
                "key_relevance_items", []
            )
            if not isinstance(relevance, list):
                continue
            key_relevance = tuple(
                str(item).strip() for item in relevance if str(item).strip()
            )
            signals = evaluation.get("quality_signals", [])
            if not isinstance(signals, list):
                continue
            quality_signals = tuple(
                str(item).strip() for item in signals if str(item).strip()
            )
            study_type = str(evaluation.get("study_type", "")).strip()
            reason = str(evaluation.get("reason", "")).strip()
            title_zh = str(
                evaluation.get("title_zh") or evaluation.get("chinese_title", "")
            ).strip()
            summary_zh = str(
                evaluation.get("summary_zh") or evaluation.get("chinese_summary", "")
            ).strip()
            if (
                not reason
                or not _contains_chinese(reason)
                or not key_relevance
                or not quality_signals
                or study_type not in {"实验", "理论", "计算", "综述", "混合", "其他"}
                or not title_zh
                or not summary_zh
            ):
                continue

            recommendations[paper_id] = Recommendation(
                paper=paper,
                relevance_score=relevance_score,
                reason=reason,
                key_relevance=key_relevance,
                title_zh=title_zh,
                summary_zh=summary_zh,
                used_ai=True,
                priority_score=priority_score,
                group_fit_score=group_fit,
                novelty_score=novelty,
                method_value_score=method_value,
                evidence_score=evidence,
                study_type=study_type,
                reading_action=_reading_action(relevance_score),
                quality_signals=quality_signals,
            )

        missing_ids = set(by_id) - set(recommendations)
        if missing_ids:
            raise ValueError(
                f"AI response omitted or invalidated {len(missing_ids)} paper evaluation(s)"
            )
        return recommendations

    def evaluate(
        self,
        papers: list[Paper],
        matches: dict[str, MatchResult],
    ) -> list[Recommendation]:
        paper_info = "\n\n---\n\n".join(paper.prompt_text() for paper in papers)
        prompt = self.prompt_template.replace("{PAPER_INFO}", paper_info)
        try:
            output_text = self._request(prompt, structured=True)
            recommendations = self._parse_recommendations(output_text, papers)
        except Exception as error:
            if (_transient_error(error) or isinstance(error, APIConnectionError)
                    or isinstance(error, APIStatusError) and error.status_code not in {400, 422}
                    or _model_unavailable(error)):
                raise
            LOGGER.warning("Structured AI evaluation failed; retrying with JSON mode: %s", error)
            fallback_prompt = (
                prompt
                + "\n\nReturn exactly one JSON object with an evaluations array. "
                "Include exactly one complete evaluation for every supplied paper. "
                "The reason for each paper must be written in Simplified Chinese. "
                "Do not return a bare array or wrap the JSON in commentary."
            )
            output_text = self._request(fallback_prompt, structured=False)
            recommendations = self._parse_recommendations(output_text, papers)

        return [recommendations[paper.paper_id] for paper in papers]

    @staticmethod
    def _parse_deep_read(output_text: str, paper: Paper) -> DeepRead:
        raw = _extract_json(output_text)

        def required_text(field: str) -> str:
            value = str(raw.get(field, "")).strip()
            if not value or not _contains_chinese(value):
                raise ValueError(f"Deep-read field {field} must contain Chinese text")
            return value

        def string_items(field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
            value = raw.get(field)
            if not isinstance(value, list):
                raise ValueError(f"Deep-read field {field} must be an array")
            items = tuple(str(item).strip() for item in value if str(item).strip())
            if not items and not allow_empty:
                raise ValueError(f"Deep-read field {field} cannot be empty")
            if items and not any(_contains_chinese(item) for item in items):
                raise ValueError(f"Deep-read field {field} must contain Chinese text")
            return items

        return DeepRead(
            paper=paper,
            title_zh=required_text("title_zh"),
            selection_reason=required_text("selection_reason"),
            one_sentence_summary=required_text("one_sentence_summary"),
            technical_route=string_items("technical_route"),
            takeaways=string_items("takeaways"),
            advances=string_items("advances"),
            limitations=string_items("limitations"),
            group_inspirations=string_items("group_inspirations"),
            author_context=string_items("author_context", allow_empty=True),
        )

    def generate_deep_read(
        self,
        paper: Paper,
        *,
        profile_name: str,
        prompt_path: Path,
    ) -> DeepRead:
        if not paper.pdf_url or paper.pdf_url == paper.abstract_url:
            raise ValueError("A distinct PDF URL is required for a deep read")

        prompt = prompt_path.read_text(encoding="utf-8")
        prompt = prompt.replace("{PROFILE_NAME}", profile_name)
        prompt = prompt.replace("{PAPER_METADATA}", paper.prompt_text())
        file_input = {
            "type": "input_file",
            "file_url": paper.pdf_url,
            "detail": "low",
        }
        content = [
            file_input,
            {"type": "input_text", "text": prompt},
        ]
        try:
            output_text = self._request_content(
                content,
                schema=DEEP_READ_SCHEMA,
                schema_name="paper_deep_read",
                max_output_tokens=12000,
            )
            return self._parse_deep_read(output_text, paper)
        except Exception as error:
            if (_transient_error(error) or isinstance(error, APIConnectionError)
                    or isinstance(error, APIStatusError) and error.status_code not in {400, 422}
                    or _model_unavailable(error)):
                raise
            LOGGER.warning(
                "Structured PDF deep read failed; retrying with JSON mode: %s",
                error,
            )
            fallback_content = [
                file_input,
                {
                    "type": "input_text",
                    "text": (
                        prompt
                        + "\n\nReturn only the requested complete JSON object. "
                        "Do not wrap it in Markdown or commentary."
                    ),
                },
            ]
            output_text = self._request_content(
                fallback_content,
                schema=None,
                schema_name="paper_deep_read",
                max_output_tokens=12000,
            )
            return self._parse_deep_read(output_text, paper)
