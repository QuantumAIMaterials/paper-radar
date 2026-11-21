"""Durable webhook delivery claims. Unknown acceptance is never automatically replayed."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any


class DeliveryUncertain(RuntimeError):
    """The request may have been accepted; reconcile before another send."""


class DeliveryRejected(RuntimeError):
    """There is positive evidence that this request was not accepted."""


class DeliveryJournal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS deliveries (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, effects TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0, error TEXT)""")
            db.commit()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        return db

    def assert_clear(self) -> None:
        with closing(self.connect()) as db:
            row = db.execute("""SELECT id FROM deliveries WHERE status IN ('sending','unknown')
                AND (id LIKE 'digest:%' OR id LIKE 'resend:%') LIMIT 1""").fetchone()
            if row:
                raise DeliveryUncertain(f"Delivery {row['id']} needs reconciliation; no resend")

    def send(self, ident: str, effects: dict[str, Any], send: Callable[[], Any]) -> bool:
        with closing(self.connect()) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT status FROM deliveries WHERE id=?", (ident,)).fetchone()
                if row and row["status"] == "sent":
                    return False
                unknown = db.execute(
                    """SELECT id FROM deliveries WHERE status IN ('sending','unknown')
                    AND (id=? OR (? AND (id LIKE 'digest:%' OR id LIKE 'resend:%'))) LIMIT 1""",
                    (ident, ident.startswith(("digest:", "resend:"))),
                ).fetchone()
                if unknown:
                    raise DeliveryUncertain(
                        f"Delivery {unknown['id']} needs reconciliation; no resend"
                    )
                now = time.time()
                db.execute(
                    """INSERT INTO deliveries(id,status,effects,created_at,updated_at)
                    VALUES(?,'sending',?,?,?) ON CONFLICT(id) DO UPDATE SET
                    status='sending',effects=excluded.effects,updated_at=excluded.updated_at,error=NULL""",
                    (ident, json.dumps(effects), now, now),
                )
            # The claim is durable before any external request, including process death.
            try:
                send()
            except BaseException as exc:
                status = "rejected" if isinstance(exc, DeliveryRejected) else "unknown"
                with db:
                    db.execute(
                        "UPDATE deliveries SET status=?,error=?,updated_at=? WHERE id=?",
                        (status, type(exc).__name__, time.time(), ident),
                    )
                raise
            with db:
                db.execute(
                    "UPDATE deliveries SET status='sent',updated_at=? WHERE id=?",
                    (time.time(), ident),
                )
        return True

    def recover(self, apply: Callable[[dict[str, Any]], None]) -> None:
        with closing(self.connect()) as db:
            rows = db.execute(
                "SELECT id,effects FROM deliveries WHERE status='sent' AND applied=0 "
                "ORDER BY created_at,id"
            ).fetchall()
            for row in rows:
                apply(json.loads(row["effects"]))
                with db:
                    db.execute("UPDATE deliveries SET applied=1 WHERE id=?", (row["id"],))
