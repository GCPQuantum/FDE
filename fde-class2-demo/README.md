# Meridian Retail Support Draft Review Tool

A small classroom demo tool that drafts a candidate customer support reply
for a human reviewer, based only on a supplied case record (`cases.json`)
and a supplied policy (`policy.md`).

**This tool drafts for human review only.** It never sends a message,
approves a refund, or changes any record. It has no web UI, no database, no
retrieval system, no login flow, and no autonomous agent behavior.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Configure your API key and model:
   ```
   python setup_env.py
   ```
   This prompts for your Anthropic API key (input hidden, never printed)
   and a model id (defaults to `claude-haiku-4-5-20251001`), and writes a
   project-local `.env` file. See `.env.example` for the two variable names
   it uses: `APP_ANTHROPIC_API_KEY` and `APP_ANTHROPIC_MODEL`. This app
   never reads the global `ANTHROPIC_API_KEY`.

## Usage

Run a live case (calls the Anthropic API):
```
python app.py --case C1
```
`--case` accepts `C1`, `C2`, or `C3` (see `cases.json`).

Simulate an unavailable/timeout state with no API call:
```
python app.py --case C1 --simulate-timeout
```

Run entirely offline using a prerecorded, labelled synthetic result from
`demo_outputs.json` (no API call is made):
```
python app.py --case C1 --offline-demo
```

Each run prints three labelled sections:

- **SUPPLIED FACTS** — the case record used.
- **SUPPLIED POLICY** — the full policy text used.
- **DRAFT RESULT FOR HUMAN REVIEW** — the model's structured output:
  - `draft_reply` — drafted reply text.
  - `evidence_refs` — policy IDs (e.g. `P1`) the draft relies on.
  - `missing_information` — field names still needed before proceeding.
  - `review_status` — one of `READY_FOR_HUMAN_REVIEW`, `NEEDS_INFORMATION`,
    `BLOCKED`, or `UNAVAILABLE` (used for simulated timeouts and any
    provider/model failure).

If the API call fails, times out, or returns output that doesn't match the
required structure, the tool shows a safe unavailable state and a manual
fallback message instead of a traceback or a stale draft.

## Tests

No test calls an external API; all model interaction is replaced with a
fake function.
```
pytest
```
