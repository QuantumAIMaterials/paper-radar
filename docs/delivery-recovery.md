# Webhook delivery recovery

Each profile stores a private SQLite journal next to its deduplication state:
`seen.json.delivery.sqlite3` or `quantum_ai_seen.json.delivery.sqlite3`.
Keep these journals with the state when migrating. Never commit or delete them
to force a retry.

The journal commits a sending claim before posting, then records an acknowledged
send before applying paper and deep-read state. Restarting after acknowledgement
recovers the exact accepted records and pending card without another HTTP request.
Read/write timeouts, invalid acknowledgements and HTTP 5xx mean unknown acceptance.
They stop automatic resend, including later schedules and explicit resend mode.
Connect failures and explicit rate-limit rejection retain bounded retry.

An unknown optional deep-read card does not stop unrelated daily digests. An
unknown digest pauses further digests for that profile until reconciliation, so
the same papers cannot be selected again under a different day or regenerated text.
Failures remain visible in the scheduler receipt and log.

For an unknown or interrupted sending record, inspect the original destination
and compare the recorded paper identities, date and stored effects. Preserve the
external evidence before resolving it as sent or positively rejected. A timeout,
an empty search, or a missing local acknowledgement does not prove rejection.
The existing webhook supplies no idempotency key or message readback; automatic
exactly-once delivery is not claimed. The safe fallback is to retain unknown.

Preview creates no journal and sends nothing. Isolated tests cover response loss,
process death before acknowledgement, crash before JSON state application,
explicit rejection, deep-read recovery and independent profile state.
