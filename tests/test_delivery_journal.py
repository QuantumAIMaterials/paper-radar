import sqlite3

import httpx
import pytest

from paper_radar import app
from paper_radar.delivery import DeliveryJournal, DeliveryRejected, DeliveryUncertain
from paper_radar.feishu import FeishuClient
from paper_radar.state import StateStore


def test_server_accepts_but_loses_response_never_posts_twice_after_restart(tmp_path, monkeypatch):
    accepted = []

    def post(url, **kwargs):
        accepted.append(kwargs)
        raise httpx.ReadTimeout("response lost", request=httpx.Request("POST", url))

    monkeypatch.setattr("paper_radar.feishu.httpx.post", post)
    monkeypatch.setattr("paper_radar.feishu.time.sleep", lambda _: pytest.fail("must not retry"))
    path = tmp_path / "delivery.db"
    client = FeishuClient("https://example.invalid/webhook")
    with pytest.raises(DeliveryUncertain):
        DeliveryJournal(path).send("digest:day1", {}, lambda: client.send_deep_read_card({}))
    for ident in ["digest:day1", "digest:day2", "resend:manual"]:
        with pytest.raises(DeliveryUncertain):
            DeliveryJournal(path).send(ident, {}, lambda: client.send_deep_read_card({}))
    assert len(accepted) == 1


def test_process_died_with_durable_sending_claim_is_not_replayed(tmp_path):
    journal = DeliveryJournal(tmp_path / "delivery.db")
    with sqlite3.connect(journal.path) as db:
        db.execute("INSERT INTO deliveries VALUES('digest:day','sending','{}',1,1,0,NULL)")
    with pytest.raises(DeliveryUncertain):
        DeliveryJournal(journal.path).send("digest:day", {}, lambda: pytest.fail("replayed"))


def test_known_rejection_can_retry_and_sent_receipt_is_idempotent(tmp_path):
    journal = DeliveryJournal(tmp_path / "delivery.db")
    with pytest.raises(DeliveryRejected):
        journal.send("digest:day", {}, lambda: (_ for _ in ()).throw(DeliveryRejected("429")))
    sent = []
    assert journal.send("digest:day", {}, lambda: sent.append(1))
    assert not DeliveryJournal(journal.path).send("digest:day", {}, lambda: sent.append(1))
    assert sent == [1]


def test_acknowledged_digest_recovers_exact_records_after_state_write_crash(tmp_path):
    state = StateStore(tmp_path / "state.json")
    journal = app._delivery_journal(state)
    effects = {
        "papers": {"original": {"status": "sent", "title": "Original accepted paper"}},
        "pending_deep_reads": {"original": {"accepted": "card"}},
        "completed_date": "2026-11-22",
    }
    journal.send("digest:2026-11-22", effects, lambda: None)
    # Process died after the acknowledged send, before JSON was written.
    restored = StateStore(state.path)
    DeliveryJournal(journal.path).recover(lambda e: app._apply_delivery_effects(restored, e))
    assert StateStore(state.path).data["papers"] == effects["papers"]
    assert restored.data["pending_deep_reads"] == effects["pending_deep_reads"]
    journal.recover(lambda _: pytest.fail("already applied"))


def test_deep_read_ack_recovered_without_duplicate_or_lost_pending_card(tmp_path):
    state = StateStore(tmp_path / "state.json")
    state.data["pending_deep_reads"] = {"p": {"original": "card"}}
    state.save()
    journal = app._delivery_journal(state)
    journal.send("deep-read:p", {"delivered_deep_reads": ["p"]}, lambda: None)
    restored = StateStore(state.path)
    journal.recover(lambda e: app._apply_delivery_effects(restored, e))
    assert restored.data["pending_deep_reads"] == {}
    assert not journal.send("deep-read:p", {}, lambda: pytest.fail("duplicate"))


def test_unknown_optional_card_does_not_block_unrelated_digest(tmp_path):
    journal = DeliveryJournal(tmp_path / "delivery.db")
    with pytest.raises(DeliveryUncertain):
        journal.send("deep-read:p", {}, lambda: (_ for _ in ()).throw(DeliveryUncertain("lost")))
    journal.assert_clear()
    assert journal.send("digest:day", {}, lambda: None)
    with pytest.raises(DeliveryUncertain):
        journal.send("deep-read:p", {}, lambda: pytest.fail("duplicate"))


def test_recovery_holds_the_write_lock_until_all_effects_are_applied(tmp_path):
    journal = DeliveryJournal(tmp_path / 'delivery.db')
    for ident in ['digest:first', 'digest:second']:
        journal.send(ident, {'id': ident}, lambda: None)
    applied = []

    def apply(effects):
        applied.append(effects['id'])
        with (sqlite3.connect(journal.path, timeout=0) as other,
              pytest.raises(sqlite3.OperationalError, match='locked')):
            other.execute('BEGIN IMMEDIATE')

    journal.recover(apply)
    assert applied == ['digest:first', 'digest:second']


@pytest.mark.parametrize("code", [500, 502, 503])
def test_server_error_is_ambiguous_not_safe_to_replay(code, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(1)
        return httpx.Response(code, request=httpx.Request("POST", url))

    monkeypatch.setattr("paper_radar.feishu.httpx.post", post)
    with pytest.raises(DeliveryUncertain):
        FeishuClient("https://example.invalid").send_deep_read_card({})
    assert len(calls) == 1
