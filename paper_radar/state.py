from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from paper_radar.identity import identity_keys, normalize_doi, paper_keys
from paper_radar.models import Paper, Recommendation

CURRENT_VERSION = 2


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.migrated = False
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": CURRENT_VERSION, "papers": {}, "ai_evaluations": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data.get("papers"), dict):
            raise ValueError(f"Unsupported or malformed state file: {self.path}")
        if not isinstance(data.get("ai_evaluations", {}), dict):
            raise ValueError(f"Unsupported or malformed state file: {self.path}")
        if data.get("version") == 1:
            for record in data["papers"].values():
                if isinstance(record, dict) and record.get("status") == "skipped":
                    record["status"] = "deferred"
            data["version"] = CURRENT_VERSION
            self.migrated = True
        if data.get("version") != CURRENT_VERSION:
            raise ValueError(f"Unsupported or malformed state file: {self.path}")
        data.setdefault("ai_evaluations", {})
        data.setdefault("last_alerted_beijing_date", {})
        return data

    def contains(self, paper_id: str) -> bool:
        return paper_id in self.data["papers"]

    def status(self, paper_id: str) -> str | None:
        record = self.data["papers"].get(paper_id)
        if not isinstance(record, dict):
            return None
        value = str(record.get("status", "")).strip()
        return value or None

    def should_consider(self, paper: Paper | str) -> bool:
        if isinstance(paper, str):
            return self.status(paper) in {None, "deferred"}
        if self.status(paper.paper_id) not in {None, "deferred"}:
            return False
        keys = paper_keys(paper)
        for paper_id, record in self.data["papers"].items():
            if not isinstance(record, dict) or record.get("status") != "sent":
                continue
            if keys & identity_keys(
                paper_id, str(record.get("title") or ""),
                record.get("doi"), record.get("aliases"),
            ):
                return False
        return True

    def completed_on(self, run_date: date) -> bool:
        return self.data.get("last_completed_beijing_date") == run_date.isoformat()

    def mark_completed(self, run_date: date) -> None:
        self.data["last_completed_beijing_date"] = run_date.isoformat()

    def alerted_on(self, run_date: date, kind: str) -> bool:
        alerts = self.data.get("last_alerted_beijing_date", {})
        return isinstance(alerts, dict) and alerts.get(kind) == run_date.isoformat()

    def mark_alerted(self, run_date: date, kind: str) -> None:
        self.data.setdefault("last_alerted_beijing_date", {})[kind] = (
            run_date.isoformat()
        )

    def mark(self, paper: Paper, status: str, *, now: datetime | None = None) -> None:
        timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        existing = self.data["papers"].get(paper.paper_id, {})
        if not isinstance(existing, dict):
            existing = {}
        self.data["papers"][paper.paper_id] = {
            "status": status,
            "seen_at": timestamp,
            "first_seen_at": existing.get("first_seen_at", timestamp),
            "attempts": int(existing.get("attempts", 0)) + 1,
            "published": paper.published.date().isoformat(),
            "title": paper.title,
            "doi": normalize_doi(paper.doi) or normalize_doi(existing.get("doi")),
            "aliases": sorted(paper_keys(paper) | identity_keys(
                paper.paper_id, str(existing.get("title") or ""),
                existing.get("doi"), existing.get("aliases"),
            )),
            "source": paper.source,
        }

    def cached_ai_evaluation(
        self, paper: Paper, cache_key: str
    ) -> Recommendation | None:
        record = self.data.get("ai_evaluations", {}).get(cache_key)
        if not isinstance(record, dict) or record.get("paper_id") != paper.paper_id:
            return None
        try:
            return Recommendation(
                paper=paper,
                relevance_score=int(record["relevance_score"]),
                reason=str(record["reason"]),
                key_relevance=_strings(record.get("key_relevance")),
                title_zh=str(record.get("title_zh", "")),
                summary_zh=str(record.get("summary_zh", "")),
                used_ai=True,
                priority_score=int(record.get("priority_score", 0)),
                group_fit_score=int(record.get("group_fit_score", 0)),
                novelty_score=int(record.get("novelty_score", 0)),
                method_value_score=int(record.get("method_value_score", 0)),
                evidence_score=int(record.get("evidence_score", 0)),
                study_type=str(record.get("study_type", "未判断")),
                reading_action=str(record.get("reading_action", "速读")),
                quality_signals=_strings(record.get("quality_signals")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def cache_ai_evaluation(
        self,
        cache_key: str,
        recommendation: Recommendation,
        *,
        now: datetime | None = None,
    ) -> None:
        timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        self.data.setdefault("ai_evaluations", {})[cache_key] = {
            "paper_id": recommendation.paper.paper_id,
            "created_at": timestamp,
            "relevance_score": recommendation.relevance_score,
            "reason": recommendation.reason,
            "key_relevance": list(recommendation.key_relevance),
            "title_zh": recommendation.title_zh,
            "summary_zh": recommendation.summary_zh,
            "priority_score": recommendation.priority_score,
            "group_fit_score": recommendation.group_fit_score,
            "novelty_score": recommendation.novelty_score,
            "method_value_score": recommendation.method_value_score,
            "evidence_score": recommendation.evidence_score,
            "study_type": recommendation.study_type,
            "reading_action": recommendation.reading_action,
            "quality_signals": list(recommendation.quality_signals),
        }

    def prune(self, retention_days: int, *, now: datetime | None = None) -> None:
        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = current_time - timedelta(days=retention_days)
        retained = {}
        for paper_id, record in self.data["papers"].items():
            try:
                seen_at = datetime.fromisoformat(record["seen_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if seen_at.astimezone(UTC) >= cutoff:
                retained[paper_id] = record
        self.data["papers"] = retained
        retained_paper_ids = set(retained)
        self.data["ai_evaluations"] = {
            key: record
            for key, record in self.data.get("ai_evaluations", {}).items()
            if isinstance(record, dict) and record.get("paper_id") in retained_paper_ids
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(self.path)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())
