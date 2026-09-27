# Meridian Retail Support-Draft Review Tool

A tiny classroom demo. It drafts a support reply **for human review only**,
using a fixed policy (`policy.md`) and fixed cases (`cases.json`).

The model runs **locally via [Ollama](https://ollama.com)** — no API key, no
account, and nothing leaves the machine. The only network traffic is to
`localhost:11434`.

It **never** sends a message, approves a refund, or changes a record. It only
reads local files, asks the local model for a draft, checks the shape of the
reply, and prints it.

## What it does

For a chosen case it prints three labelled sections:

- **SUPPLIED FACTS** — the case record from `cases.json`
- **SUPPLIED POLICY** — the text of `policy.md`
- **DRAFT RESULT FOR HUMAN REVIEW** — the model's JSON draft, after validation

The model is asked to return only JSON with exactly these fields, and Ollama
**enforces that schema** on the model's output (`format=DRAFT_SCHEMA` in
`app.py`), so malformed JSON is prevented at the source rather than only caught
afterwards:

| field | type | meaning |
|-------|------|---------|
| `draft_reply` | string | the drafted reply text |
| `evidence_refs` | list of strings | policy IDs supporting the draft (e.g. `["P1"]`) |
| `missing_information` | list of strings | field names that are missing and needed |
| `review_status` | string | `READY_FOR_HUMAN_REVIEW`, `NEEDS_INFORMATION`, or `BLOCKED` |

> **Note:** validation checks *structure and types only*. Correct structure
> does **not** prove the content is correct — a human must still review it.
> Model size changes how often that bites: `llama3:8b` returns the right
> *shape* every time but will cite the wrong policy ID and ask for a
> `delivery_date` the facts already supply. `qwen3.5:9b` gets both right. Run
> the same case against each to show that passing validation and being
> sendable are different things:
>
> ```bash
> APP_OLLAMA_MODEL=llama3:8b  python app.py --case C3   # cites P2, asks for a date it has
> APP_OLLAMA_MODEL=qwen3.5:9b python app.py --case C3   # cites P3, refuses to confirm resolution
> ```

`validate_result` re-checks every field even though the schema was enforced
server-side — the app never trusts the server to have done it.

## Setup

**1. Install Ollama and pull the model** (one-time, ~6.6 GB):

```bash
brew install --cask ollama    # or download from https://ollama.com/download
ollama serve                  # skip if the Ollama app is already running
ollama pull qwen3.5:9b
```

Check it: `curl http://localhost:11434/api/version`

**2. Install the Python side:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: write the project-only .env (model + host; no key to enter):
python setup_env.py
```

### Configuration

All three variables have working defaults, so the app runs with no
configuration at all:

| variable | default | meaning |
|----------|---------|---------|
| `APP_OLLAMA_MODEL` | `qwen3.5:9b` | any model shown by `ollama list` |
| `APP_OLLAMA_HOST` | `http://localhost:11434` | where the Ollama server listens |
| `APP_OLLAMA_TIMEOUT` | `60` | seconds; the first call after a cold start is the slow one |

There is **no API key**. Sampling is fixed at `temperature: 0` so classroom runs
are reproducible, and `think=False` is passed on every call: reasoning models
otherwise spend the token budget thinking and return empty content, which the
app would report as `UNAVAILABLE`. Models without a thinking mode ignore the
flag, so any model in `ollama list` works.

If you use a `.env` file, load it into your shell before running (e.g. with
`set -a; source .env; set +a`), or export the variables yourself.

## Usage

```bash
# Normal run (calls the local model; ~2-3s warm, longer on a cold start):
python app.py --case C1        # accepts C1, C2, or C3

# No model call; show the safe "unavailable" state and a manual fallback:
python app.py --case C1 --simulate-timeout

# No model call; print a labelled prerecorded synthetic result:
python app.py --case C3 --offline-demo

# Try a different local model without editing anything
# (after: ollama pull llama3:8b):
APP_OLLAMA_MODEL=llama3:8b python app.py --case C2
```

### Safe states

- `--simulate-timeout`: makes no model call, clears `draft_reply`, sets
  `review_status` to `UNAVAILABLE`, and prints a manual-fallback message.
- `--offline-demo`: makes no model call, loads a labelled synthetic result from
  `demo_outputs.json`, and prints an explicit line that no model API was called.
- A stopped Ollama server, a model that was never pulled, a timeout, invalid
  JSON, or a wrong shape all fall back to the same `UNAVAILABLE` state — never
  a traceback and never a stale draft. The result also carries an
  `unavailable_reason` diagnostic so you can tell *which* of those happened.
- `UNAVAILABLE` is deliberately **not** an allowed `review_status`, so the model
  can never claim it; only the app's own error path produces it.

## Tests

```bash
pip install -r requirements.txt
pytest
```

Tests never call a real model, local or remote; it is injected as a fake
function. They cover a successful C1 result, invalid model output,
timeout/failure behaviour, an unreachable server (pointed at a dead loopback
port on purpose), agreement between the enforced schema and the validator,
loading all input files, offline outputs, and `setup_env.py`.

Ollama does **not** need to be running to run the tests.

## Files

- `app.py` — the tool
- `setup_env.py` — writes the project-only `.env` and checks that Ollama is
  reachable and the model is installed
- `policy.md`, `cases.json` — supplied inputs (pre-existing)
- `demo_outputs.json` — labelled synthetic results for `--offline-demo`
- `.env.example` — defaults only; no secrets exist in this project
- `requirements.txt` — pinned, tested versions
- `test_app.py`, `test_setup_env.py` — API-free tests
