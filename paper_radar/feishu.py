from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from paper_radar.models import DeepRead, MatchResult, Recommendation

BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
LOGGER = logging.getLogger(__name__)


def build_signature(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _markdown_link(label: str, url: str) -> str:
    escaped_label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{escaped_label}]({url})"


def build_card(recommendation: Recommendation, match: MatchResult) -> dict[str, Any]:
    paper = recommendation.paper
    display_title = recommendation.title_zh or paper.title
    summary = recommendation.summary_zh or _truncate(paper.abstract, 700)
    authors = _truncate(", ".join(paper.authors), 300)
    matched_terms = ", ".join(match.matched_terms[:8])
    relevance = ", ".join(recommendation.key_relevance[:6]) or matched_terms
    score_icons = "*" * recommendation.relevance_score
    content = (
        f"**Original title:** {paper.title}\n"
        f"**Authors:** {authors}\n"
        f"**Published:** {paper.published.date().isoformat()}\n"
        f"**Priority:** {recommendation.relevance_score}/3 {score_icons}\n"
        f"**Matched terms:** {matched_terms}\n\n"
        f"**Summary**\n{summary}\n\n"
        f"**Why it matters**\n{recommendation.reason}\n\n"
        f"**Key relevance:** {relevance}"
    )
    actions = [
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "打开原文"},
            "url": paper.abstract_url,
            "type": "primary",
        }
    ]
    if paper.pdf_url and paper.pdf_url != paper.abstract_url:
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "打开 PDF"},
                "url": paper.pdf_url,
                "type": "default",
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue" if recommendation.relevance_score == 2 else "red",
            "title": {"tag": "plain_text", "content": _truncate(display_title, 120)},
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": content}},
            {
                "tag": "action",
                "actions": actions,
            },
        ],
    }


def build_digest_card(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    *,
    generated_at: datetime | None = None,
    notices: tuple[str, ...] = (),
    digest_title: str = "二维量子材料论文速递",
    digest_intro: str = "二维量子材料、量子器件和纳米加工",
) -> dict[str, Any]:
    if not recommendations:
        raise ValueError("Cannot build an empty paper digest")

    digest_date = (generated_at or datetime.now(UTC)).astimezone(BEIJING_TIMEZONE).date()
    high_priority_count = sum(item.relevance_score == 3 for item in recommendations)
    digest_topic = (
        digest_title.removesuffix("论文速递").strip() or digest_intro or digest_title
    )

    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": (
                    f"今天 Top {len(recommendations)} 里对 "
                    f"{digest_topic} 比较值得看的："
                ),
            },
        }
    ]
    if notices:
        notice_text = "；".join(notices)
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**提示：** {_truncate(notice_text, 300)}",
                },
            }
        )

    for index, recommendation in enumerate(recommendations, start=1):
        paper = recommendation.paper
        match = matches[paper.paper_id]
        display_title = recommendation.title_zh or paper.title
        summary = recommendation.summary_zh or paper.abstract
        source_parts = [paper.source]
        if paper.venue:
            source_parts.append(paper.venue)
        source_parts.append(paper.published.date().isoformat())
        if paper.citation_count is not None and paper.citation_count > 0:
            source_parts.append(f"引用 {paper.citation_count}")
        source_line = _truncate(" / ".join(source_parts), 140)
        reason = recommendation.reason or (
            "命中本雷达关注方向：" + ", ".join(match.matched_terms[:5])
        )
        original_title_link = _markdown_link(
            _truncate(paper.title, 300), paper.abstract_url
        )
        pdf_link = ""
        if paper.pdf_url and paper.pdf_url != paper.abstract_url:
            pdf_link = f" / {_markdown_link('PDF', paper.pdf_url)}"
        content = (
            f"**{_truncate(display_title, 180)}（推荐 #{index}）**\n"
            f"{original_title_link}\n"
            f"{source_line}{pdf_link}\n\n"
            f"**做什么：** {_truncate(summary, 360)}\n\n"
            f"**和我们组的关系：** {_truncate(reason, 260)}"
        )
        elements.extend(
            [
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": content}},
            ]
        )

    elements.extend(
        [
            {"tag": "hr"},
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "如果明天没有新增值得看的论文，我会保持安静。",
                },
            },
        ]
    )
    panel_title = f"今日精选 {len(recommendations)} 篇 · 展开全文"
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange" if notices else ("red" if high_priority_count else "blue"),
            "title": {
                "tag": "plain_text",
                "content": f"{digest_title} | {digest_date.isoformat()}",
            },
        },
        "body": {
            "elements": [
                {
                    "tag": "collapsible_panel",
                    "expanded": False,
                    "header": {
                        "title": {"tag": "plain_text", "content": panel_title},
                        "background_color": "grey",
                        "vertical_align": "center",
                        "icon": {
                            "tag": "standard_icon",
                            "token": "down-small-ccm_outlined",
                        },
                        "icon_position": "right",
                        "icon_expanded_angle": -180,
                    },
                    "border": {"color": "grey", "corner_radius": "5px"},
                    "vertical_spacing": "8px",
                    "padding": "10px 10px 10px 10px",
                    "elements": elements,
                }
            ]
        },
    }


def _deep_read_section(title: str, items: tuple[str, ...], limit: int) -> dict[str, Any]:
    content = "\n".join(f"- {_truncate(item, limit)}" for item in items)
    return {"tag": "markdown", "content": f"**{title}**\n{content}"}


def build_deep_read_card(deep_read: DeepRead) -> dict[str, Any]:
    paper = deep_read.paper
    links = f"[原文]({paper.abstract_url})"
    if paper.pdf_url and paper.pdf_url != paper.abstract_url:
        links += f" · [PDF]({paper.pdf_url})"
    authors = _truncate(", ".join(paper.authors), 700)
    overview = (
        f"**英文题目：** {_truncate(paper.title, 240)}\n"
        f"**链接：** {links}\n"
        f"**作者：** {authors}\n\n"
        f"**入选理由：** {_truncate(deep_read.selection_reason, 360)}\n\n"
        f"**一句话概括：** {_truncate(deep_read.one_sentence_summary, 420)}"
    )
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": overview},
        {"tag": "hr"},
        _deep_read_section("技术路线", deep_read.technical_route[:6], 520),
        {"tag": "hr"},
        _deep_read_section("Takeaway", deep_read.takeaways[:5], 420),
        {"tag": "hr"},
        _deep_read_section("先进在哪里", deep_read.advances[:5], 420),
        {"tag": "hr"},
        _deep_read_section("我对局限的判断", deep_read.limitations[:4], 420),
    ]
    if deep_read.author_context:
        elements.extend(
            [
                {"tag": "hr"},
                _deep_read_section("作者信息", deep_read.author_context[:4], 420),
            ]
        )
    elements.extend(
        [
            {"tag": "hr"},
            _deep_read_section("对我们组的启发", deep_read.group_inspirations[:4], 460),
            {
                "tag": "markdown",
                "content": "速读基于论文 PDF 与元数据生成；关键结论请以原文为准。",
            },
        ]
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red",
            "title": {
                "tag": "plain_text",
                "content": f"论文速读｜{_truncate(deep_read.title_zh, 90)}",
            },
        },
        "body": {
            "elements": [
                {
                    "tag": "collapsible_panel",
                    "expanded": False,
                    "header": {
                        "title": {
                            "tag": "plain_text",
                            "content": "技术路线、结论与启发 · 展开全文",
                        },
                        "background_color": "grey",
                        "vertical_align": "center",
                        "icon": {
                            "tag": "standard_icon",
                            "token": "down-small-ccm_outlined",
                        },
                        "icon_position": "right",
                        "icon_expanded_angle": -180,
                    },
                    "border": {"color": "grey", "corner_radius": "5px"},
                    "vertical_spacing": "8px",
                    "padding": "10px 10px 10px 10px",
                    "elements": elements,
                }
            ]
        },
    }


def build_alert_card(title: str, message: str, *, fatal: bool = False) -> dict[str, Any]:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red" if fatal else "orange",
            "title": {"tag": "plain_text", "content": _truncate(title, 80)},
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": _truncate(message, 1800)},
            }
        ],
    }


@dataclass(slots=True)
class FeishuClient:
    webhook_url: str
    signing_secret: str = ""
    timeout: float = 30.0
    retry_delays: tuple[float, ...] = (10.0, 30.0, 60.0)

    def _send_card(self, card: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": card,
        }
        if self.signing_secret:
            timestamp = int(time.time())
            payload["timestamp"] = str(timestamp)
            payload["sign"] = build_signature(self.signing_secret, timestamp)

        for attempt in range(len(self.retry_delays) + 1):
            response: httpx.Response | None = None
            try:
                response = httpx.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                code = body.get("code", body.get("StatusCode", 0))
                if code in {0, "0", None}:
                    return
                message = body.get(
                    "msg", body.get("StatusMessage", "unknown Feishu error")
                )
                if str(code) != "11232" or attempt == len(self.retry_delays):
                    raise RuntimeError(
                        f"Feishu rejected the message: {code} {message}"
                    )
            except (httpx.RequestError, httpx.HTTPStatusError) as error:
                retryable = not isinstance(error, httpx.HTTPStatusError) or (
                    error.response.status_code == 429
                    or error.response.status_code >= 500
                )
                if not retryable or attempt == len(self.retry_delays):
                    raise

            delay = self.retry_delays[attempt]
            retry_after = (
                response.headers.get("Retry-After", "").strip()
                if response is not None
                else ""
            )
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
            LOGGER.warning(
                "Feishu delivery was rate limited or temporarily unavailable; "
                "retrying in %.0f second(s) (%d/%d)",
                delay,
                attempt + 1,
                len(self.retry_delays),
            )
            time.sleep(delay)

    def send(self, recommendation: Recommendation, match: MatchResult) -> None:
        self._send_card(build_card(recommendation, match))

    def send_digest(
        self,
        recommendations: list[Recommendation],
        matches: dict[str, MatchResult],
        *,
        notices: tuple[str, ...] = (),
        digest_title: str = "二维量子材料论文速递",
        digest_intro: str = "二维量子材料、量子器件和纳米加工",
    ) -> None:
        self._send_card(
            build_digest_card(
                recommendations,
                matches,
                notices=notices,
                digest_title=digest_title,
                digest_intro=digest_intro,
            )
        )

    def send_deep_read_card(self, card: dict[str, Any]) -> None:
        self._send_card(card)

    def send_deep_read(self, deep_read: DeepRead) -> None:
        self.send_deep_read_card(build_deep_read_card(deep_read))

    def send_alert(self, title: str, message: str, *, fatal: bool = False) -> None:
        self._send_card(build_alert_card(title, message, fatal=fatal))
