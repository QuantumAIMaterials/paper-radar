from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as time_module
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from paper_radar.config import RadarConfig, load_config
from paper_radar.feishu import FeishuClient, build_deep_read_card, build_digest_card
from paper_radar.matcher import match_paper
from paper_radar.models import DeepRead, MatchResult, Paper, Recommendation
from paper_radar.recommender import AIRecommender, fallback_recommendation
from paper_radar.semantic_scholar import EnrichmentError, describe_error, enrich_papers
from paper_radar.sources import fetch_all_papers
from paper_radar.state import StateStore

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find relevant research papers and send them to Feishu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "keywords.yml",
        help="Keyword configuration file",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "config" / "recommender_prompt.txt",
        help="AI recommendation prompt",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_ROOT / "state" / "seen.json",
        help="Persistent deduplication state",
    )
    parser.add_argument(
        "--deep-read-prompt",
        type=Path,
        default=PROJECT_ROOT / "config" / "deep_read_prompt.txt",
        help="AI prompt used for the optional Top 1 PDF deep read",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not send or update state")
    parser.add_argument("--no-ai", action="store_true", help="Disable optional AI evaluation")
    parser.add_argument(
        "--resend-latest",
        action="store_true",
        help="Ignore deduplication for this run and leave existing state unchanged",
    )
    parser.add_argument(
        "--deliver-not-before",
        metavar="HH:MM",
        help="Delay Feishu delivery until this Beijing time if the run finishes early",
    )
    parser.add_argument(
        "--start-not-before",
        metavar="HH:MM",
        help="Delay scheduled work until this Beijing time",
    )
    parser.add_argument(
        "--max-start-wait-minutes",
        type=int,
        help="Skip an early trigger instead of waiting longer than this many minutes",
    )
    parser.add_argument(
        "--once-per-beijing-day",
        action="store_true",
        help="Skip work after this state file has completed successfully today",
    )
    parser.add_argument("--lookback-days", type=int, help="Override the lookback window")
    parser.add_argument("--max-results", type=int, help="Override the arXiv result limit")
    parser.add_argument("--max-papers", type=int, help="Override the delivery limit")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _rank_matches(
    papers: list[Paper],
    matches: dict[str, MatchResult],
    state: StateStore,
) -> list[Paper]:
    return sorted(
        papers,
        key=lambda paper: (
            state.status(paper.paper_id) == "deferred",
            paper.published,
            matches[paper.paper_id].score,
        ),
        reverse=True,
    )


def _evaluate(
    papers: list[Paper],
    matches: dict[str, MatchResult],
    *,
    recommender: AIRecommender | None,
    state: StateStore | None = None,
    profile_name: str = "",
) -> tuple[list[Recommendation], tuple[str, ...]]:
    if recommender is None:
        LOGGER.info("AI evaluation is disabled; using deterministic keyword ranking")
        return (
            [fallback_recommendation(paper, matches[paper.paper_id]) for paper in papers],
            ("AI 评估未启用，本期使用关键词兜底排序。",),
        )

    cached_by_id: dict[str, Recommendation] = {}
    cache_keys: dict[str, str] = {}
    pending_papers: list[Paper] = []
    if state is not None and profile_name:
        for paper in papers:
            cache_key = recommender.evaluation_cache_key(
                paper,
                profile_name=profile_name,
            )
            cache_keys[paper.paper_id] = cache_key
            cached = state.cached_ai_evaluation(paper, cache_key)
            if cached is None:
                pending_papers.append(paper)
            else:
                cached_by_id[paper.paper_id] = cached
    else:
        pending_papers = papers

    if cached_by_id:
        LOGGER.info("Reused %d cached AI paper evaluation(s)", len(cached_by_id))
    if not pending_papers:
        return [cached_by_id[paper.paper_id] for paper in papers], ()

    try:
        recommendations = recommender.evaluate(pending_papers, matches)
        LOGGER.info("AI evaluated %d new paper(s)", len(recommendations))
        if state is not None:
            for recommendation in recommendations:
                cache_key = cache_keys.get(recommendation.paper.paper_id)
                if cache_key:
                    state.cache_ai_evaluation(cache_key, recommendation)
        recommendations_by_id = {
            recommendation.paper.paper_id: recommendation
            for recommendation in recommendations
        }
        recommendations_by_id.update(cached_by_id)
        return [recommendations_by_id[paper.paper_id] for paper in papers], ()
    except Exception as error:
        LOGGER.exception("AI evaluation failed; using deterministic keyword ranking")
        recommendations_by_id = {
            paper_id: recommendation
            for paper_id, recommendation in cached_by_id.items()
        }
        recommendations_by_id.update(
            {
                paper.paper_id: fallback_recommendation(paper, matches[paper.paper_id])
                for paper in pending_papers
            }
        )
        return (
            [recommendations_by_id[paper.paper_id] for paper in papers],
            (f"AI 评估失败（{type(error).__name__}），本期使用关键词兜底排序。",),
        )


def _calibrate_priorities(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    *,
    max_high_priority: int,
) -> list[Recommendation]:
    recommendations.sort(
        key=lambda item: (
            item.priority_score,
            item.paper.published,
            matches[item.paper.paper_id].score,
        ),
        reverse=True,
    )
    calibrated = []
    high_priority_count = 0
    for recommendation in recommendations:
        if recommendation.relevance_score == 3:
            if high_priority_count >= max_high_priority:
                recommendation = replace(
                    recommendation,
                    relevance_score=2,
                    reading_action="速读",
                )
            else:
                high_priority_count += 1
        calibrated.append(recommendation)
    return calibrated


def _select_for_delivery(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    *,
    minimum_relevance: int,
    max_papers: int,
    max_non_preferred_papers: int,
) -> tuple[list[Recommendation], set[str]]:
    eligible = [
        item for item in recommendations if item.relevance_score >= minimum_relevance
    ]
    selected: list[Recommendation] = []
    quota_skipped: set[str] = set()
    non_preferred_count = 0
    for recommendation in eligible:
        if len(selected) >= max_papers:
            break
        paper_id = recommendation.paper.paper_id
        is_preferred = bool(matches[paper_id].preferred_terms)
        if not is_preferred and non_preferred_count >= max_non_preferred_papers:
            quota_skipped.add(paper_id)
            continue
        selected.append(recommendation)
        if not is_preferred:
            non_preferred_count += 1
    return selected, quota_skipped


def _select_deep_read_candidate(
    recommendations: list[Recommendation],
    *,
    enabled: bool,
    minimum_priority_score: int,
) -> Recommendation | None:
    if not enabled:
        return None
    return next(
        (
            item
            for item in recommendations
            if item.used_ai
            and item.relevance_score == 3
            and item.priority_score >= minimum_priority_score
            and item.method_value_score >= 4
            and item.evidence_score >= 3
            and _is_reliable_deep_read_pdf(item.paper)
        ),
        None,
    )


def _is_reliable_deep_read_pdf(paper: Paper) -> bool:
    if not paper.source.startswith("arXiv") or not paper.pdf_url:
        return False
    pdf_host = (urlsplit(paper.pdf_url).hostname or "").lower()
    return pdf_host in {"arxiv.org", "export.arxiv.org"}


def _generate_deep_read(
    candidate: Recommendation | None,
    recommender: AIRecommender | None,
    *,
    profile_name: str,
    prompt_path: Path,
) -> DeepRead | None:
    if candidate is None or recommender is None:
        return None
    try:
        deep_read = recommender.generate_deep_read(
            candidate.paper,
            profile_name=profile_name,
            prompt_path=prompt_path,
        )
        LOGGER.info("Generated a PDF deep read for %s", candidate.paper.paper_id)
        return deep_read
    except Exception:
        LOGGER.exception(
            "Optional PDF deep read failed for %s; continuing with the digest",
            candidate.paper.paper_id,
        )
        return None


def _enrich_candidates(
    candidates: list[Paper],
    config: RadarConfig,
) -> tuple[list[Paper], tuple[str, ...]]:
    if not config.semantic_scholar.enabled or not candidates:
        return candidates, ()
    try:
        enriched = enrich_papers(candidates, config.semantic_scholar)
        return enriched, ()
    except Exception as error:
        LOGGER.exception("Semantic Scholar candidate enrichment failed")
        if isinstance(error, EnrichmentError):
            candidates = error.papers
        return (
            candidates,
            (
                "Semantic Scholar 附加元数据暂不可用"
                f"（{describe_error(error)}），已保留现有数据；部分附加字段可能缺失。",
            ),
        )


def _seconds_until_beijing_time(
    target_time: str,
    *,
    now: datetime | None = None,
) -> float:
    try:
        hour_text, minute_text = target_time.split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as error:
        raise ValueError("--deliver-not-before must use HH:MM format") from error
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("--deliver-not-before must use a valid 24-hour time")

    current_time = (now or datetime.now(UTC)).astimezone(BEIJING_TIMEZONE)
    target = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if current_time >= target:
        return 0.0
    return (target - current_time).total_seconds()


def _wait_for_delivery_window(target_time: str | None) -> None:
    if not target_time:
        return
    seconds = _seconds_until_beijing_time(target_time)
    if seconds <= 0:
        return
    LOGGER.info(
        "Waiting %.0f second(s) until %s Beijing time before Feishu delivery",
        seconds,
        target_time,
    )
    time_module.sleep(seconds)


def _wait_for_start_window(
    target_time: str | None,
    *,
    max_wait_minutes: int | None,
) -> bool:
    if not target_time:
        return True
    if max_wait_minutes is not None and max_wait_minutes <= 0:
        raise ValueError("--max-start-wait-minutes must be greater than zero")

    seconds = _seconds_until_beijing_time(target_time)
    if seconds <= 0:
        return True
    if max_wait_minutes is not None and seconds > max_wait_minutes * 60:
        LOGGER.info(
            "Skipping an early trigger %.0f seconds before %s Beijing time",
            seconds,
            target_time,
        )
        return False
    LOGGER.info(
        "Waiting %.0f second(s) to start at %s Beijing time",
        seconds,
        target_time,
    )
    time_module.sleep(seconds)
    return True


def _beijing_date(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(BEIJING_TIMEZONE).date()


def _print_preview(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    notices: tuple[str, ...],
    *,
    deep_read: DeepRead | None,
    digest_title: str,
    digest_intro: str,
) -> None:
    digest = (
        build_digest_card(
            recommendations,
            matches,
            notices=notices,
            digest_title=digest_title,
            digest_intro=digest_intro,
        )
        if recommendations
        else None
    )
    preview: Any = digest
    if deep_read is not None:
        preview = {
            "digest": digest,
            "deep_read": build_deep_read_card(deep_read),
        }
    output = json.dumps(preview, ensure_ascii=False, indent=2) + "\n"
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(output.encode("utf-8"))
    else:
        print(output, end="")


def _feishu_client() -> FeishuClient | None:
    webhook_url = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return None
    return FeishuClient(
        webhook_url=webhook_url,
        signing_secret=os.getenv("FEISHU_SIGNING_SECRET", "").strip(),
    )


def _send_alert(title: str, message: str, *, fatal: bool = False) -> bool:
    client = _feishu_client()
    if client is None:
        LOGGER.warning("Cannot send Feishu alert because FEISHU_WEBHOOK_URL is missing")
        return False
    try:
        client.send_alert(title, message, fatal=fatal)
        return True
    except Exception:
        LOGGER.exception("Failed to send the Feishu runtime alert")
        return False


def _send_alert_once(
    state: StateStore,
    run_date: date,
    kind: str,
    title: str,
    message: str,
    *,
    fatal: bool = False,
) -> bool:
    if state.alerted_on(run_date, kind):
        LOGGER.info("A %s alert is already recorded for %s; suppressing it", kind, run_date)
        return True
    if not _send_alert(title, message, fatal=fatal):
        return False
    state.mark_alerted(run_date, kind)
    state.save()
    return True


def _deliver_pending_deep_reads(state: StateStore, client: FeishuClient) -> bool:
    pending = state.data.setdefault("pending_deep_reads", {})
    success = True
    for paper_id, card in list(pending.items()):
        try:
            client.send_deep_read_card(card)
        except Exception:
            LOGGER.exception("Pending PDF deep read delivery failed for %s", paper_id)
            success = False
            continue
        del pending[paper_id]
        state.save()
        LOGGER.info("Sent pending PDF deep read for %s", paper_id)
    return success


def run(args: argparse.Namespace) -> int:
    config: RadarConfig = load_config(args.config)
    if args.lookback_days is not None or args.max_results is not None:
        config = replace(
            config,
            arxiv=replace(
                config.arxiv,
                lookback_days=args.lookback_days or config.arxiv.lookback_days,
                max_results=args.max_results or config.arxiv.max_results,
            ),
            crossref=replace(
                config.crossref,
                lookback_days=args.lookback_days or config.crossref.lookback_days,
            ),
        )

    if not _wait_for_start_window(
        args.start_not_before,
        max_wait_minutes=args.max_start_wait_minutes,
    ):
        return 0

    state = StateStore(args.state)
    run_date = _beijing_date()
    # Retry persisted cards before the daily digest guard, without regenerating AI output.
    if not args.dry_run and not args.resend_latest and state.data.get("pending_deep_reads"):
        pending_client = _feishu_client()
        if pending_client is None or not _deliver_pending_deep_reads(state, pending_client):
            return 1
    if args.once_per_beijing_day and state.completed_on(run_date):
        LOGGER.info("A successful run is already recorded for %s; skipping", run_date)
        return 0

    fetch_result = fetch_all_papers(config, enrich_semantic_scholar=False)
    papers = fetch_result.papers
    notices = list(fetch_result.warnings)
    source_failures = list(fetch_result.failures)
    standalone_alert_notices = list(source_failures)
    matches: dict[str, MatchResult] = {}
    matched_papers: list[Paper] = []
    for paper in papers:
        match = match_paper(paper, config.matching)
        if match is not None and (
            args.resend_latest or state.should_consider(paper)
        ):
            matches[paper.paper_id] = match
            matched_papers.append(paper)

    if args.resend_latest:
        LOGGER.info(
            "%d paper(s) matched the %s rules with deduplication bypassed",
            len(matched_papers),
            config.profile.name,
        )
    else:
        LOGGER.info(
            "%d pending paper(s) matched the %s rules",
            len(matched_papers),
            config.profile.name,
        )
    if not matched_papers:
        if source_failures and not args.dry_run:
            _send_alert_once(
                state,
                run_date,
                "source_failure",
                f"{config.profile.name}雷达降级运行",
                "\n".join(f"- {item}" for item in source_failures),
            )
        if not args.dry_run and not args.resend_latest:
            if args.once_per_beijing_day and not source_failures:
                state.mark_completed(run_date)
            if state.migrated or (args.once_per_beijing_day and not source_failures):
                state.save()
        return 1 if source_failures else 0

    ranked = _rank_matches(matched_papers, matches, state)
    candidates = ranked[: config.run.ai_candidate_limit]
    candidates, enrichment_notices = _enrich_candidates(candidates, config)
    notices.extend(enrichment_notices)
    recommender = None if args.no_ai else AIRecommender.from_environment(args.prompt)
    recommendations, evaluation_notices = _evaluate(
        candidates,
        matches,
        recommender=recommender,
        state=state,
        profile_name=config.profile.name,
    )
    # Explicit --no-ai remains supported. Unplanned AI degradation must not
    # consume candidates or mark a day complete; scheduled runs can recover.
    if not args.no_ai and evaluation_notices:
        LOGGER.error("AI evaluation unavailable; retaining candidates for a later run")
        if not args.dry_run and not args.resend_latest:
            for paper in matched_papers:
                state.mark(paper, "deferred")
            state.save()
            _send_alert_once(
                state, run_date, "evaluation_failure",
                f"{config.profile.name}AI 评估暂缓",
                "AI 评估暂未完成，论文已保留待重试；本次不发送关键词兜底日报。"
                "请查看 Actions 日志；后续定时任务会再次尝试。",
            )
        return 1
    notices.extend(evaluation_notices)
    standalone_alert_notices.extend(evaluation_notices)
    recommendations = _calibrate_priorities(
        recommendations,
        matches,
        max_high_priority=config.run.max_high_priority_per_run,
    )
    max_papers = args.max_papers or config.run.max_papers_per_run
    selected, quota_skipped_ids = _select_for_delivery(
        recommendations,
        matches,
        minimum_relevance=config.run.minimum_ai_relevance,
        max_papers=max_papers,
        max_non_preferred_papers=config.run.max_non_preferred_papers,
    )
    deep_read_candidate = _select_deep_read_candidate(
        selected,
        enabled=config.run.deep_read_enabled,
        minimum_priority_score=config.run.deep_read_min_priority_score,
    )
    deep_read = _generate_deep_read(
        deep_read_candidate,
        recommender,
        profile_name=config.profile.name,
        prompt_path=args.deep_read_prompt,
    )

    client = _feishu_client()
    effective_dry_run = args.dry_run or client is None
    if client is None and not args.dry_run:
        LOGGER.warning("FEISHU_WEBHOOK_URL is missing; switching to dry-run mode")
    if effective_dry_run:
        _print_preview(
            selected,
            matches,
            tuple(notices),
            deep_read=deep_read,
            digest_title=config.profile.digest_title,
            digest_intro=config.profile.digest_intro,
        )
        return 1 if source_failures else 0

    if selected or standalone_alert_notices:
        _wait_for_delivery_window(args.deliver_not_before)

    sent_ids: set[str] = set()
    failures = int(bool(source_failures))
    if selected:
        try:
            client.send_digest(
                selected,
                matches,
                notices=tuple(notices),
                digest_title=config.profile.digest_title,
                digest_intro=config.profile.digest_intro,
            )
            sent_ids.update(item.paper.paper_id for item in selected)
            if source_failures:
                state.mark_alerted(run_date, "source_failure")
            LOGGER.info("Sent a Feishu digest containing %d paper(s)", len(selected))
            # Persist digest delivery before attempting its separate optional card.
            if not args.resend_latest:
                for item in selected:
                    state.mark(item.paper, "sent")
                if deep_read is not None:
                    state.data.setdefault("pending_deep_reads", {})[
                        deep_read.paper.paper_id
                    ] = build_deep_read_card(deep_read)
                state.save()
            if deep_read is not None:
                if args.resend_latest:
                    try:
                        client.send_deep_read(deep_read)
                    except Exception:
                        failures = 1
                        LOGGER.exception("Failed to send resend-mode PDF deep read")
                elif not _deliver_pending_deep_reads(state, client):
                    failures = 1
        except Exception:
            failures = 1
            LOGGER.exception("Failed to send the Feishu paper digest")
    else:
        LOGGER.info("No papers met the delivery threshold; no digest was sent")
        if standalone_alert_notices:
            alert_kind = (
                "source_failure" if source_failures else "evaluation_failure"
            )
            _send_alert_once(
                state,
                run_date,
                alert_kind,
                f"{config.profile.name}雷达降级运行",
                "\n".join(f"- {item}" for item in standalone_alert_notices),
            )
        elif notices:
            LOGGER.info(
                "Suppressed a standalone alert for optional metadata enrichment"
            )

    if args.resend_latest:
        LOGGER.info("Resend mode left the deduplication state unchanged")
        return 1 if failures else 0

    now = datetime.now(UTC)
    selected_ids = {item.paper.paper_id for item in selected}
    recommendations_by_id = {item.paper.paper_id: item for item in recommendations}
    for paper in matched_papers:
        if paper.paper_id in sent_ids:
            state.mark(paper, "sent", now=now)
        elif paper.paper_id in selected_ids:
            continue
        elif (
            paper.paper_id in recommendations_by_id
            and (
                recommendations_by_id[paper.paper_id].relevance_score
                < config.run.minimum_ai_relevance
                or paper.paper_id in quota_skipped_ids
            )
        ):
            state.mark(paper, "skipped", now=now)
        else:
            state.mark(paper, "deferred", now=now)
    state.prune(config.run.state_retention_days, now=now)
    if args.once_per_beijing_day and not failures:
        state.mark_completed(run_date)
    state.save()
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except Exception as error:
        LOGGER.exception("Paper Radar failed")
        if not args.dry_run:
            _send_alert(
                "论文雷达运行失败",
                f"任务未完成：{type(error).__name__}。请检查 GitHub Actions 日志。",
                fatal=True,
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
