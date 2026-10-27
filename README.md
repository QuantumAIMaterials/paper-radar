# Paper Radar

Paper Radar runs two independent research feeds from one GitHub repository:

- **2D Quantum Materials** watches quantum phenomena in genuinely two-dimensional
  materials.
- **Quantum AI Materials** watches AI for materials and quantum science,
  scientific agents, autonomous laboratories, and quantum machine learning with
  a concrete scientific application.

Both feeds query arXiv and Crossref, automatically fall back to OpenAlex's arXiv
index when the official Atom API is rate limited, enrich available metadata
through Semantic Scholar, rank matching papers, and send a Chinese digest to a
dedicated Feishu group bot. GitHub Actions hosts the scheduled jobs, so Windows,
WSL, and Docker do not need to remain online.

## Pipeline

1. Query the official arXiv Atom API and Crossref Works API with polite rate limits;
   switch promptly to OpenAlex when arXiv is unavailable or rate limited.
2. Merge duplicate preprints and journal records by DOI and normalized title,
   and skip previously delivered papers across sources and days using the same
   identifiers. Merged records retain alternate IDs, DOI forms, and title aliases;
   Greek scientific letters remain distinct. Deduplication is checked again when
   optional metadata reveals a previously unknown DOI. Old state files remain
   readable without a reset.
3. Require both a profile-specific core term and a profile-specific focus term.
4. Reconsider papers marked `deferred` before newly discovered papers.
5. Add venue, publication type, and citation metadata from Semantic Scholar for the shortlisted candidates when available.
6. Evaluate up to twenty candidates with a strict Responses API JSON schema,
   reusing cached AI evaluations when possible.
7. Send one concise Chinese daily digest containing up to ten recommended papers.
8. When a PDF-backed paper clears the strict Top 1 threshold, append one structured
   deep read covering its route, findings, advances, limitations, and lab takeaways.
   The entire deep read is collapsed by default and expands with one click.
9. Commit that profile's independent delivery state back to the repository.

AI evaluation is optional. If it is disabled or fails, deterministic keyword
ranking keeps the pipeline running and the digest reports the degraded mode.
Source and fatal runtime failures are also reported to the corresponding Feishu
group.

## Radar profiles

| Profile | Keywords | AI prompt | State | GitHub workflow |
| --- | --- | --- | --- | --- |
| 2D Quantum Materials | `config/keywords.yml` | `config/recommender_prompt.txt` | `state/seen.json` | `2D Quantum Materials Radar` |
| Quantum AI Materials | `config/quantum_ai_keywords.yml` | `config/quantum_ai_recommender_prompt.txt` | `state/quantum_ai_seen.json` | `Quantum AI Materials Radar` |

Both profiles share `config/deep_read_prompt.txt` for the optional PDF-grounded
Top 1 paper deep read.

The 2D profile requires a 2D-material term plus a quantum-physics term such as
quantum transport, superconductivity, topology, correlation, magnetism,
spin/valley physics, excitons, or moire physics. It explicitly excludes AI and
materials-informatics papers.

The Quantum AI profile requires an AI-method term plus a materials, chemistry,
condensed-matter, or quantum-science term. Generic business, medical, finance,
traffic, and consumer AI is excluded. A plain materials paper without a
substantive AI method is also rejected, so the two feeds do not duplicate one
another.

Each YAML profile also controls the Feishu title and introduction through its
`profile` section. This keeps presentation, matching, AI review, and state
isolated while sharing the source and delivery code.

## GitHub and Feishu setup

Create one Feishu custom bot in each target group and enable signature
verification. Then open:

`Repository Settings > Secrets and variables > Actions`

Add the following delivery secrets. The existing 2D values stay unchanged; the
Quantum AI values must come from the new group's bot.

| Secret | Required | Purpose |
| --- | --- | --- |
| `FEISHU_WEBHOOK_URL` | Yes for 2D delivery | 2D group's custom-bot webhook |
| `FEISHU_SIGNING_SECRET` | Recommended | 2D group's signature secret |
| `QUANTUM_AI_FEISHU_WEBHOOK_URL` | Yes for Quantum AI delivery | Quantum AI group's custom-bot webhook |
| `QUANTUM_AI_FEISHU_SIGNING_SECRET` | Recommended | Quantum AI group's signature secret |
| `LLM_API_KEY` | No | Shared AI recommendation and Chinese summaries |
| `LLM_BASE_URL` | No | Shared Responses-compatible API base URL |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Raises Semantic Scholar rate limits if available |
| `OPENALEX_API_KEY` | No | Raises the limit of the automatic arXiv backup |

Optional repository variables are shared by both feeds:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-5.6-sol` | Responses API model |
| `LLM_REASONING_EFFORT` | `low` | Evaluation reasoning effort |
| `ARXIV_CONTACT` | empty | Contact email sent to arXiv and Crossref |

Never commit webhook URLs, signing secrets, or API keys to the repository.

The arXiv Atom API gets one attempt because GitHub-hosted runners are frequently
rate limited there. On failure, OpenAlex promptly supplies arXiv records with one
combined OR query per profile and `Retry-After`-aware backoff. If both paths fail,
the workflow stays failed and does not record that Beijing day as completed,
allowing the next scheduled attempt to retry. Source alerts are persisted and
sent at most once per profile and day.

Source responses must contain a valid Atom feed or the expected JSON results
list. Malformed HTTP 200 responses fail the source instead of recording an empty,
completed day. Crossref ignores invalid DOI records, tolerates null optional
fields, and excludes future publication dates.

Semantic Scholar is an optional enrichment source. Rate limits and temporary
server or transport errors use exponential backoff starting at ten seconds.
Every request, including recursive batch splits, observes the configured request
interval. `Retry-After` supports seconds and HTTP dates; if its requested cooldown
exceeds the configured maximum retry delay, enrichment stops for that run instead
of retrying too early or delaying the digest indefinitely.

A `No valid paper ids given` response is treated as no indexed metadata, without
retrying or splitting those same identifiers. Other rejected batches are split
to isolate an invalid identifier. If a later batch or split fails, successful
earlier metadata is retained. Null fields never become literal `None` values or
erase existing metadata, and malformed batch lengths are rejected before pairing
results with papers. Crossref and OpenAlex citation counts remain available even
when Semantic Scholar cannot supplement them. Logs count actual metadata matches,
not merely the number of input papers. An enrichment-only issue is logged and is
not sent as a standalone Feishu alert when there are no recommended papers.

## First Quantum AI run

1. Open the repository's **Actions** tab.
2. Select **Quantum AI Materials Radar** and choose **Run workflow**.
3. Keep `dry_run` enabled, `no_ai` disabled, and enable `resend_latest` for a
   complete preview that ignores the empty or existing state.
4. Inspect the JSON preview in the workflow log.
5. Add the two `QUANTUM_AI_FEISHU_*` secrets from the new Feishu group.
6. Run again with `dry_run` disabled and confirm the digest card in that group.

`resend_latest` leaves state unchanged. Keep it disabled for scheduled and
normal manual runs.

GitHub cron is a best-effort scheduler, so each profile has four staggered daily
triggers instead of one. Delayed early triggers wait and begin preparing the 2D
digest at 18:30 Beijing time and the Quantum AI digest at 18:40. This offset
avoids making both profiles compete for source API rate limits. Completed 2D
digests wait at a delivery gate until 19:00; Quantum AI delivers at 19:08 to
avoid Feishu application-level frequency limiting. A trigger that arrives too
early to wait safely exits, leaving the later attempts as fallbacks. Each profile
records one successful completion per Beijing day in its state file, so redundant
triggers do not create duplicate digests. Failed or incomplete source runs do not
record completion and remain eligible for a later attempt. Feishu rate limits and
temporary server failures are retried automatically. Manual delivery still runs
immediately unless a timing option is passed explicitly.

## Optional AI evaluation

The evaluator calls the Responses API and requests schema-constrained JSON for
group fit, novelty, method value, abstract evidence, study type, concrete
quality signals, Chinese title, summary, and recommendation reason. The final
priority is computed in application code rather than accepted directly from the
model. Successful evaluations are cached in the selected state file using the
paper metadata, profile, model, reasoning effort, and prompt hash, so deferred
papers are not re-evaluated unless the prompt or inputs change. Each completed
model response logs token usage when the relay returns usage metadata. At most
three papers per digest can retain the `3/3` label.

The optional deep read is restricted to a directly accessible arXiv PDF and
requires a successful AI evaluation,
`3/3` priority, a composite score of at least 82, method value of at least 4/5,
abstract evidence of at least 3/5, and a distinct PDF URL. Only the highest
qualifying paper is processed. PDF or relay failures skip the deep read without
blocking the daily digest. Author context is restricted to affiliations and
contribution information stated in the PDF; the model is not asked to invent
biographies or recent publication lists.

Daily Feishu digests use one Card JSON 2.0 collapsible panel for the entire
delivery. The card defaults to one compact summary row; expanding it once
reveals every recommendation in the same concise format: recommendation rank,
linked original title, source/date, “做什么”, and “和我们组的关系”. Internal scores,
authors, and keyword diagnostics stay out of the group-facing digest. Individual
papers are not nested in separate panels, and a day with no new qualifying paper
remains silent.

The implementation follows the official OpenAI Structured Outputs guidance:

<https://developers.openai.com/api/docs/guides/structured-outputs>

PDF deep reads use Responses API file inputs:

<https://developers.openai.com/api/docs/guides/file-inputs>

For the configured Responses-compatible relay, use:

```text
LLM_BASE_URL=https://bench.physcai.com/openai
LLM_MODEL=gpt-5.6-sol
```

Store the URL as a GitHub secret and the model as a GitHub variable. Do not add
the local Codex `auth.json` file to this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# 2D Quantum Materials preview
python -m paper_radar --dry-run --no-ai --max-results 20

# Quantum AI Materials preview
python -m paper_radar \
  --config config/quantum_ai_keywords.yml \
  --prompt config/quantum_ai_recommender_prompt.txt \
  --state state/quantum_ai_seen.json \
  --dry-run --no-ai --max-results 20

ruff check .
pytest -q
```

If the selected workflow's `FEISHU_WEBHOOK_URL` is missing, the command
returns a failure when papers need delivery and leaves the candidates available
for retry. Use explicit `--dry-run` for a preview without a webhook.

## State behavior

Each radar has its own small, reviewable JSON state. Successfully delivered
papers are marked `sent`; papers explicitly rated below the threshold are
marked `skipped`; papers that only missed a candidate or delivery limit are
marked `deferred` and receive priority in the next run. Failed deliveries
remain eligible for retry. Records older than the configured retention window
are removed. Cached AI evaluations are pruned with their corresponding paper
records. The state also stores the last successfully completed Beijing date to
make redundant scheduled attempts idempotent.

Each GitHub Actions workflow serializes runs for its own profile. State commits
pull and push with retry so the two profiles can prepare in parallel and still
land their independent state files safely. State-free manual previews use an
isolated concurrency group, so a scheduled run waiting for its delivery window
does not block or get cancelled by a dry run.
Manual dry runs check out the branch selected in Actions so fixes can be tested
before merging. Production runs always check out the latest `main`. Dry runs
and explicit resend runs skip the workflow's state-commit step, including when
a source fails. Resend-mode alerts also leave the on-disk state unchanged.

## AI outage recovery

Transient AI errors (connection/stream transport failures, HTTP 408/409/429 and
5xx) are retried with 30, 60, 120 and 240 second delays. Numeric Retry-After
headers are honored up to 300 seconds. SDK retries are disabled to keep this
budget bounded. Authentication failures are not retried. A server outage does
not trigger another complete retry cycle in JSON fallback mode.

When AI evaluation remains unavailable, normal runs return a failure status,
retain papers as deferred, preserve valid AI caches, and leave the day
incomplete so a later scheduled run can retry. No keyword-only digest is sent
in this case, and an outage alert is deduplicated per Beijing day. Explicit
`--no-ai` still permits keyword-only delivery. Dry runs and resend previews do
not write recovery state. Previously sent papers remain deduplicated; this
change does not automatically resend historical degraded digests.

This handles temporary outages, but a sustained provider outage can still delay
the digest. Inspect Actions logs for authentication, quota or prolonged service
errors instead of assuming that retrying can fix them.

PDF deep reads use schema 2.0 Markdown components, including the footer (no
legacy `note` or `div` components). After successful digest delivery, the sent
paper IDs and generated deep-read card are persisted before attempting the
separate card. Failed cards remain in `pending_deep_reads` in the profile's state
file and are retried before the next run's daily-completion check, without
repeating AI generation or the digest. A failed delivery returns a failure
status. Pending-card retries respect the configured delivery time. Dry-run and
explicit resend mode do not drain this queue. Previously
failed cards from runs before this change were not stored and cannot be
recovered automatically. A webhook accepted just before a process crash or an
ambiguous network timeout can still be duplicated; webhooks provide no
transaction with the local state file.
