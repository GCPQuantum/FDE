"""Meridian Retail support-draft review tool.

Drafts a structured case review for HUMAN REVIEW ONLY. It never sends a
message, approves a return/refund, or changes a record. It only reads local
files -- the case data, the policy, and the "context pack" of task.md,
definitions.md and contractDefinitions.md -- calls a LOCAL model served by
Ollama to produce a draft, validates the shape of the response (including
every source_id it cites), and prints the result.

No API key and no network egress: the model runs on this machine and is
reached over localhost.

Usage:
    python app.py --case C1
    python app.py --case C2 --simulate-timeout
    python app.py --case C3 --offline-demo
"""

import argparse                 # parses the --case / --simulate-timeout flags
import json                     # reads the input files, parses the model's reply
import os                       # reads APP_OLLAMA_* configuration
import sys                      # writes the usage error to stderr
from dataclasses import dataclass  # bundles the context-pack files into one value
from datetime import datetime   # computes the delivery window in code, not in the model
from pathlib import Path        # builds file paths that don't depend on the cwd

# Anchor every path to the folder holding app.py, NOT the folder you happen to
# be standing in. Without this, `python /some/where/app.py` would look for
# cases.json in the wrong place.
BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "cases.json"                    # the supplied case facts
POLICY_PATH = BASE_DIR / "policy.md"                    # the supplied policy (P1..P4)
DEMO_PATH = BASE_DIR / "demo_outputs.json"               # prerecorded answers for --offline-demo
EVIDENCE_MAP_PATH = BASE_DIR / "evidence_map.json"       # source_id -> artefact, per case
# The "context pack": classroom-authored task/definitions/contract files that
# drive the prompt. They are read verbatim, the same way policy.md always
# has been -- app.py does not re-word or summarise them.
TASK_PATH = BASE_DIR / "task.md"
DEFINITIONS_PATH = BASE_DIR / "definitions.md"
CONTRACT_PATH = BASE_DIR / "contractDefinitions.md"

VALID_CASES = ("C1", "C2", "C3")  # argparse rejects anything else for free

# The only three verdicts the model is allowed to return. Note what is NOT in
# this set: UNAVAILABLE. That one belongs to the app alone (see
# unavailable_result), so the model can never claim its own failure state.
VALID_REVIEW_STATUS = {
    "READY_FOR_HUMAN_REVIEW",   # facts + policy support a draft a human can check
    "NEEDS_INFORMATION",        # a required fact is missing; ask for it
    "BLOCKED",                  # the request cannot be met from this record
}

# The return window from P1, in hours. The model is never asked to do date
# arithmetic -- Python computes the verdict and hands it over as a fact.
DELIVERY_WINDOW_HOURS = 78.0

# The three possible outcomes of that computation. Strings rather than a
# boolean because "no delivery date" is a third state, not a false.
WINDOW_UNKNOWN = "UNKNOWN_DELIVERY_DATE"  # no date on record -> P2, ask for it
WINDOW_WITHIN = "WITHIN_WINDOW"           # inside the window -> review for refund
WINDOW_OUTSIDE = "OUTSIDE_WINDOW"         # past the window -> reject, out of time policy

DEFAULT_MODEL = "qwen3.5:9b"                # any name from `ollama list`
DEFAULT_HOST = "http://localhost:11434"     # where the Ollama server listens
DEFAULT_TIMEOUT = 60.0                      # seconds; a cold model load is slow

# Fields the model itself must produce. case_id, known_facts and
# missing_information are the OTHER three fields of the Appendix E / Class 2
# contract (contractDefinitions.md) -- the app fills those in from cases.json
# and evidence_map.json rather than trusting the model to copy them, because
# their correctness (which source_id backs which fact) is exactly the thing
# that must never depend on the model getting a citation right.
#
# Ollama enforces this schema on the model's output, so malformed JSON is
# prevented at the source rather than only caught afterwards. validate_model_result
# still re-checks everything -- the app never trusts the server to have done it.
MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "case_summary": {"type": "string"},                                # summarises only supplied evidence
        "evidence_refs": {"type": "array", "items": {"type": "string"}},   # supplied policy IDs, e.g. ["P1"]
        "evidence_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source_id": {"type": "string"},
                },
                "required": ["claim", "source_id"],
                "additionalProperties": False,
            },
        },
        "conflicting_information": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description", "source_ids"],
                "additionalProperties": False,
            },
        },
        "draft_reply": {"type": "string"},                                 # proposed wording only
        "review_status": {"type": "string", "enum": sorted(VALID_REVIEW_STATUS)},
        "human_action_required": {"type": "string"},                       # next human-owned step
    },
    "required": sorted(
        (
            "case_summary",
            "evidence_refs",
            "evidence_links",
            "conflicting_information",
            "draft_reply",
            "review_status",
            "human_action_required",
        )
    ),
    "additionalProperties": False,  # no extra keys; the model's part of the contract is exactly these seven
}

# One source of truth for "which keys the model must produce", derived from
# the schema rather than typed out a second time.
REQUIRED_MODEL_KEYS = frozenset(MODEL_SCHEMA["properties"])

# The full Appendix E / Class 2 contract (contractDefinitions.md): the seven
# model keys above plus the three the app fills in itself.
REQUIRED_RESULT_KEYS = frozenset(REQUIRED_MODEL_KEYS | {"case_id", "known_facts", "missing_information"})

# Shown whenever no usable draft exists, so the operator is told what to do
# next instead of being left with an empty screen.
MANUAL_FALLBACK_MESSAGE = (
    "The drafting model is unavailable. No draft was produced. "
    "A human agent must handle this case manually using the supplied facts "
    "and policy above."
)


# --------------------------------------------------------------------------
# Loading supplied inputs (relative to app.py, not the current directory)
# --------------------------------------------------------------------------
def load_text(path):
    """Return a text file's contents verbatim."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_case(case_id, cases_path=CASES_PATH):
    """Return the case dict for case_id from cases.json."""
    # cases_path is a parameter (not a hardcoded constant) so tests can point
    # this at a temporary fixture file.
    with open(cases_path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    if case_id not in cases:
        # Fail loudly here: a typo'd case ID is a programmer error, not a
        # model failure, so it must NOT be swallowed into the safe state.
        raise KeyError(f"Case {case_id!r} not found in {cases_path}")
    return cases[case_id]


def load_policy(policy_path=POLICY_PATH):
    """Return the policy text from policy.md."""
    # Passed to the model verbatim as text -- the policy is never parsed or
    # summarised, so what the model sees is exactly what a human would read.
    return load_text(policy_path)


def load_evidence_map(evidence_map_path=EVIDENCE_MAP_PATH):
    """Return the source_id -> artefact map from evidence_map.json."""
    with open(evidence_map_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class ContextPack:
    """The classroom-authored files that govern every run, loaded once.

    task.md, definitions.md and contractDefinitions.md are read verbatim, the
    same way policy.md always has been -- this is the "context pack", not a
    paraphrase of it written into app.py.
    """

    policy_text: str
    task_text: str
    definitions_text: str
    contract_text: str
    evidence_map: dict


def load_context_pack(
    policy_path=POLICY_PATH,
    task_path=TASK_PATH,
    definitions_path=DEFINITIONS_PATH,
    contract_path=CONTRACT_PATH,
    evidence_map_path=EVIDENCE_MAP_PATH,
):
    """Load every file in the context pack, all paths overridable for tests."""
    return ContextPack(
        policy_text=load_policy(policy_path),
        task_text=load_text(task_path),
        definitions_text=load_text(definitions_path),
        contract_text=load_text(contract_path),
        evidence_map=load_evidence_map(evidence_map_path),
    )


# --------------------------------------------------------------------------
# Evidence: source IDs, the known facts they back, and what's missing
# --------------------------------------------------------------------------
# task.md's Evidence section names four source IDs for one specific case
# (support_queue:C2, order_record:MR-1088, issue_record:C2,
# policy_register:policy.md). task.md is a TEMPLATE shared by every case, so
# the real IDs for whichever case was requested are built here rather than
# read literally off the page -- build_system_prompt tells the model to use
# these, not the ones printed in the task text.
def source_ids_for_case(case_id, order_id):
    """Return the four source IDs task.md's Evidence section describes."""
    return {
        "support_queue": f"support_queue:{case_id}",
        "order_record": f"order_record:{order_id}",
        "issue_record": f"issue_record:{case_id}",
        "policy_register": "policy_register:policy.md",
    }


def valid_source_ids_for_case(case_id, source_ids, evidence_map):
    """Return the set of source IDs the model may cite for this case.

    Also checks that evidence_map.json actually defines the four IDs this
    case needs -- a config mistake there must surface as a clear failure, not
    let every citation for this case silently pass validation.
    """
    known = evidence_map.get(case_id, {})
    for source_id in source_ids.values():
        if source_id not in known:
            raise RuntimeError(
                f"evidence_map.json has no entry for {source_id!r} (case {case_id!r})."
            )
    return frozenset(known)


def known_facts_for_case(case, source_ids):
    """Return the known_facts list: {field, value, source_id}, code-derived.

    Pairing a fact with its source_id is exactly the kind of citation a model
    gets wrong under load, so it is computed here from cases.json and never
    left to the model to copy.
    """
    facts = [
        {"field": "order_id", "value": case["order_id"], "source_id": source_ids["order_record"]},
        {"field": "product", "value": case["product"], "source_id": source_ids["order_record"]},
    ]
    # delivery_date is a known fact only when present; when it is missing it
    # belongs in missing_information instead (see missing_information_for_case).
    if case.get("delivery_date"):
        facts.append(
            {"field": "delivery_date", "value": case["delivery_date"], "source_id": source_ids["order_record"]}
        )
    facts.append({"field": "issue_status", "value": case["issue_status"], "source_id": source_ids["issue_record"]})
    facts.append(
        {"field": "issue_details", "value": case["issue_details"], "source_id": source_ids["issue_record"]}
    )
    facts.append(
        {
            "field": "customer_question",
            "value": case["customer_question"],
            "source_id": source_ids["support_queue"],
        }
    )
    return facts


def missing_information_for_case(case):
    """Return the missing_information list. Only delivery_date can be missing."""
    return [] if case.get("delivery_date") else ["delivery_date"]


# --------------------------------------------------------------------------
# Deterministic facts computed in code, never by the model
# --------------------------------------------------------------------------
# An LLM cannot reliably subtract two dates, and it has no idea what "now" is.
# So the elapsed time and the verdict are calculated here, exactly and
# repeatably, then handed to the model as facts it only has to read.
def hours_since_delivery(case, now=None):
    """Hours elapsed since delivery, or None when the record has no date."""
    raw = case.get("delivery_date")
    if not raw:
        # Covers null, "" and a missing key alike -- all mean "no date".
        return None
    now = now or datetime.now()
    # fromisoformat accepts "2026-09-12" (midnight) and a full timestamp. A
    # malformed non-empty date raises, which becomes the UNAVAILABLE state --
    # a corrupt record must not produce a confident customer reply.
    delivered = datetime.fromisoformat(raw)
    return (now - delivered).total_seconds() / 3600.0


def derive_time_facts(case, now=None):
    """Return the delivery-window facts handed to the model as COMPUTED FACTS.

    now is injectable so tests are not at the mercy of the system clock.
    """
    now = now or datetime.now()
    elapsed = hours_since_delivery(case, now)
    if elapsed is None:
        verdict = WINDOW_UNKNOWN
    elif elapsed < DELIVERY_WINDOW_HOURS:
        verdict = WINDOW_WITHIN
    else:
        verdict = WINDOW_OUTSIDE
    return {
        # Stated explicitly: without it the model has no reference for "now".
        "current_datetime": now.isoformat(timespec="seconds"),
        "delivery_window_hours": DELIVERY_WINDOW_HOURS,
        "hours_since_delivery": None if elapsed is None else round(elapsed, 1),
        # The verdict the model must act on. One value, already decided.
        "window_verdict": verdict,
    }


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_system_prompt(context, source_ids):
    # The context pack (task.md, definitions.md, contractDefinitions.md) is
    # embedded verbatim -- app.py states the hard limits and the window rule
    # in its own words, but does not rewrite what the classroom files already
    # say about the contract or the vocabulary.
    return (
        "You are a support-draft assistant for Meridian Retail. You draft "
        "case reviews for HUMAN REVIEW ONLY. You cannot approve returns or "
        "refunds, send customer messages, change records, or invent facts.\n\n"
        "Follow the TASK, DEFINITIONS and OUTPUT CONTRACT supplied verbatim "
        "below.\n\n"
        "TASK (task.md):\n"
        f"{context.task_text}\n\n"
        # task.md's Evidence section names one case literally. It is a
        # template shared by every case, so this run's real IDs are given
        # here and MUST be used instead of whatever case letter or order
        # number appears in the TASK text above.
        "FOR THIS RUN, use these evidence source IDs -- ignore any case letter "
        "or order number written in the TASK's Evidence section above, which "
        "is a template:\n"
        f"  support_queue: {source_ids['support_queue']}\n"
        f"  order_record: {source_ids['order_record']}\n"
        f"  issue_record: {source_ids['issue_record']}\n"
        f"  policy_register: {source_ids['policy_register']}\n\n"
        "DEFINITIONS (definitions.md):\n"
        f"{context.definitions_text}\n\n"
        "OUTPUT CONTRACT (contractDefinitions.md):\n"
        f"{context.contract_text}\n\n"
        # The rule is spelled out per verdict so the model only has to match a
        # string, never to reason about dates or pick a status on its own.
        "The COMPUTED FACTS given in the next message carry a window_verdict "
        "already calculated in code. Use it exactly as given and do NOT do "
        "any date arithmetic yourself:\n"
        '  "UNKNOWN_DELIVERY_DATE": delivery_date is missing. Set review_status '
        "to NEEDS_INFORMATION. human_action_required must say a person needs "
        "to obtain and confirm the delivery date from the order_record source "
        "before the two-day window can be evaluated (P2).\n"
        '  "WITHIN_WINDOW": delivered within the policy window. Set review_status '
        "to READY_FOR_HUMAN_REVIEW and say clearly in draft_reply that this is "
        "not yet an approved return. human_action_required must say a person "
        "needs to confirm the delivery date and review the case for possible "
        "return approval (P1).\n"
        '  "OUTSIDE_WINDOW": delivered outside the policy window. Set review_status '
        "to BLOCKED. human_action_required must say a person needs to confirm "
        "the rejection is correct before it is communicated (P1).\n"
        "Whatever the verdict, if issue_status is open you must never say the "
        "issue is resolved (P3).\n\n"
        "The SUPPLIED KNOWN FACTS and SUPPLIED MISSING INFORMATION in the next "
        "message are already final -- do not repeat, alter or invent them; "
        "only cite their source_id values inside evidence_links.\n\n"
        "Return ONLY a single JSON object, with no surrounding text or "
        "markdown fences, containing exactly these keys. (case_id, known_facts "
        "and missing_information are the contract's other three fields; the "
        "system fills those in and you must not produce them.)\n"
        '  "case_summary": a string summarising only the supplied evidence.\n'
        '  "evidence_refs": a list of supplied policy IDs (e.g. ["P1", "P2"]).\n'
        '  "evidence_links": a list of {"claim": string, "source_id": string} '
        "pairs, using only the source IDs given above.\n"
        '  "conflicting_information": a list of {"description": string, '
        '"source_ids": [string, ...]} pairs; an empty list when nothing conflicts.\n'
        '  "draft_reply": a string, proposed wording only -- never a claim that '
        "an action occurred.\n"
        '  "review_status": one of "READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", '
        'or "BLOCKED".\n'
        '  "human_action_required": a string naming the next human-owned step, '
        "without inventing a person."
    )


def build_user_prompt(case, policy_text, known_facts, missing_information, time_facts):
    # Dump every supplied/computed input as readable JSON rather than prose,
    # so the model sees the exact field names and source IDs it must cite.
    raw_case = json.dumps(case, indent=2, ensure_ascii=False)
    facts_block = json.dumps(
        {"known_facts": known_facts, "missing_information": missing_information},
        indent=2,
        ensure_ascii=False,
    )
    computed = json.dumps(time_facts, indent=2, ensure_ascii=False)
    return (
        "SUPPLIED POLICY (policy_register:policy.md):\n"
        f"{policy_text}\n\n"
        "SUPPLIED CASE RECORD (for context; cite SUPPLIED KNOWN FACTS below, "
        "not this block, since only that one carries source IDs):\n"
        f"{raw_case}\n\n"
        # Its own labelled section so it is unmistakably a given, not
        # something the model is being asked to work out or re-derive.
        "SUPPLIED KNOWN FACTS AND MISSING INFORMATION (already paired with "
        "source IDs; final -- do not recompute):\n"
        f"{facts_block}\n\n"
        "COMPUTED FACTS (already calculated in code -- trust these exactly "
        "and do not recompute them):\n"
        f"{computed}\n\n"
        "Draft the case review now and respond with only the JSON object."
    )


# --------------------------------------------------------------------------
# Local model call (injectable for tests)
# --------------------------------------------------------------------------
def call_ollama(system_prompt, user_prompt):
    """Call the local Ollama server and return the raw text response.

    Reads the model from APP_OLLAMA_MODEL, the server from APP_OLLAMA_HOST,
    and the timeout from APP_OLLAMA_TIMEOUT. There is no API key: the model
    runs locally and nothing leaves this machine.
    """
    # Imported inside the function, not at the top of the file, so the tests
    # (and --offline-demo) still run on a machine without the package.
    import ollama

    # Every setting has a working default, so the app runs with no config.
    model = os.environ.get("APP_OLLAMA_MODEL", DEFAULT_MODEL)
    host = os.environ.get("APP_OLLAMA_HOST", DEFAULT_HOST)
    try:
        timeout = float(os.environ.get("APP_OLLAMA_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        # A typo'd timeout is a config error worth naming, not a silent 60s.
        raise RuntimeError("APP_OLLAMA_TIMEOUT must be a number of seconds.")

    # A per-call timeout is the whole reason for building an explicit client:
    # a hung local model must not hang the tool forever.
    client = ollama.Client(host=host, timeout=timeout)
    try:
        response = client.chat(
            model=model,
            # Two roles: the rules, then this specific case.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Hand the schema to the server: Ollama constrains generation so
            # the model *cannot* emit malformed JSON or a bad status value.
            format=MODEL_SCHEMA,
            # Reasoning models (qwen3.5 and friends) default to thinking on,
            # spend the token budget on reasoning, and return empty content --
            # which this app would correctly but uselessly report as
            # UNAVAILABLE. Ask for the answer directly. Models without a
            # thinking mode ignore this flag.
            think=False,
            # temperature 0 keeps the classroom demo reproducible.
            options={"temperature": 0, "num_predict": 1024},
        )
    except ollama.ResponseError as exc:
        # The server answered, but with an error. 404 means "no such model",
        # by far the most common setup mistake -- so name the exact fix.
        if exc.status_code == 404:
            raise RuntimeError(
                f"Model {model!r} is not installed on the Ollama server at "
                f"{host}. Install it with: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"Ollama returned an error: {exc}") from exc
    except Exception as exc:
        # The server never answered: not running, wrong host, or timed out.
        # The exception's class name is kept so the cause stays identifiable.
        raise RuntimeError(
            f"Could not complete the request to Ollama at {host} "
            f"({type(exc).__name__}: {exc}). Is the Ollama server running?"
        ) from exc

    # `or ""` guards the empty-content case (a thinking model that spent its
    # budget reasoning) so the caller gets a string and fails in validation.
    return response.message.content or ""


# --------------------------------------------------------------------------
# Validation (structure only; correct structure does not prove correct content)
# --------------------------------------------------------------------------
def validate_model_result(raw_text, valid_source_ids):
    """Parse and validate the model's seven fields. Raise ValueError on any problem.

    valid_source_ids is the set this case's evidence_links and
    conflicting_information may cite -- the four IDs evidence_map.json
    defines for it. A citation outside that set is exactly the kind of wrong
    reference the schema alone cannot catch.
    """
    # Step 1: is it JSON at all? Everything below re-checks what the schema
    # already enforced -- deliberately. The server is not a trusted validator.
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Response was not valid JSON: {exc}")

    # Step 2: valid JSON can still be a list, a number, or a bare string.
    if not isinstance(data, dict):
        raise ValueError("Response JSON was not an object.")

    # Step 3: exactly the seven model keys -- no missing ones, no extras.
    if set(data.keys()) != set(REQUIRED_MODEL_KEYS):
        raise ValueError(
            f"Response keys {sorted(data.keys())} do not match required "
            f"keys {sorted(REQUIRED_MODEL_KEYS)}."
        )

    # Step 4: the simple string and string-list fields.
    if not isinstance(data["case_summary"], str):
        raise ValueError("case_summary must be a string.")
    if not isinstance(data["draft_reply"], str):
        raise ValueError("draft_reply must be a string.")
    if not isinstance(data["human_action_required"], str):
        raise ValueError("human_action_required must be a string.")
    if not isinstance(data["evidence_refs"], list) or not all(
        isinstance(item, str) for item in data["evidence_refs"]
    ):
        raise ValueError("evidence_refs must be a list of strings.")

    # Step 5: evidence_links -- {claim, source_id} pairs, source_id must
    # resolve to a supplied artefact for this case.
    if not isinstance(data["evidence_links"], list):
        raise ValueError("evidence_links must be a list.")
    for link in data["evidence_links"]:
        if (
            not isinstance(link, dict)
            or set(link.keys()) != {"claim", "source_id"}
            or not isinstance(link["claim"], str)
            or not isinstance(link["source_id"], str)
        ):
            raise ValueError('Each evidence_links item must be {"claim": str, "source_id": str}.')
        if link["source_id"] not in valid_source_ids:
            raise ValueError(f"evidence_links cites unknown source_id {link['source_id']!r}.")

    # Step 6: conflicting_information -- {description, source_ids} pairs,
    # every source_id must resolve the same way.
    if not isinstance(data["conflicting_information"], list):
        raise ValueError("conflicting_information must be a list.")
    for conflict in data["conflicting_information"]:
        if (
            not isinstance(conflict, dict)
            or set(conflict.keys()) != {"description", "source_ids"}
            or not isinstance(conflict["description"], str)
            or not isinstance(conflict["source_ids"], list)
            or not all(isinstance(item, str) for item in conflict["source_ids"])
        ):
            raise ValueError(
                'Each conflicting_information item must be '
                '{"description": str, "source_ids": [str, ...]}.'
            )
        for source_id in conflict["source_ids"]:
            if source_id not in valid_source_ids:
                raise ValueError(f"conflicting_information cites unknown source_id {source_id!r}.")

    # Step 7: the verdict must be one of the three allowed values. This is the
    # check that stops a model from inventing "APPROVED" or claiming
    # "UNAVAILABLE" on the app's behalf.
    if data["review_status"] not in VALID_REVIEW_STATUS:
        raise ValueError(
            f"review_status {data['review_status']!r} is not one of "
            f"{sorted(VALID_REVIEW_STATUS)}."
        )

    # Structurally sound. Says nothing about whether the CONTENT is right --
    # that is what the human reviewer is for.
    return data


def assemble_result(case_id, known_facts, missing_information, model_fields):
    """Merge the code-derived fields with the model's validated fields.

    case_id, known_facts and missing_information never pass through the
    model -- they are correct by construction from cases.json and
    evidence_map.json, so there is nothing for the model to get wrong here.
    """
    return {
        "case_id": case_id,
        "case_summary": model_fields["case_summary"],
        "known_facts": known_facts,
        "evidence_refs": model_fields["evidence_refs"],
        "evidence_links": model_fields["evidence_links"],
        "missing_information": missing_information,
        "conflicting_information": model_fields["conflicting_information"],
        "draft_reply": model_fields["draft_reply"],
        "review_status": model_fields["review_status"],
        "human_action_required": model_fields["human_action_required"],
    }


def unavailable_result(case_id=None, reason=None):
    """The safe state used for timeouts and any local-model failure.

    reason is a diagnostic string (server down, model not pulled, bad shape,
    unknown source_id). It is never a draft, so it is safe to show alongside
    the fallback message.
    """
    # Same shape as a real result, so downstream code and the printer need no
    # special case -- but with review_status set to a value the model is
    # structurally unable to produce, and every text/list field empty so no
    # partial or stale content can ever be mistaken for a sendable draft.
    result = {
        "case_id": case_id or "",
        "case_summary": "",
        "known_facts": [],
        "evidence_refs": [],
        "evidence_links": [],
        "missing_information": [],
        "conflicting_information": [],
        "draft_reply": "",
        "review_status": "UNAVAILABLE",
        "human_action_required": "",
        "manual_fallback": MANUAL_FALLBACK_MESSAGE,
    }
    # Only attached when there is something to say, so --simulate-timeout and
    # a real crash both read naturally.
    if reason:
        result["unavailable_reason"] = reason
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def generate_draft(case, context, call_model=call_ollama, time_facts=None):
    """Produce a validated result dict, or the safe unavailable state.

    call_model is injectable so tests never touch the network. time_facts is
    passed in by main() so the values printed for the reviewer are exactly
    the ones the model saw.
    """
    case_id = case["case_id"]
    # Everything lives inside the try, evidence and prompt building included:
    # a corrupt delivery_date, or a source_id evidence_map.json doesn't
    # define, must reach the safe state like any other failure rather than
    # escaping as a traceback.
    try:
        source_ids = source_ids_for_case(case_id, case["order_id"])
        valid_source_ids = valid_source_ids_for_case(case_id, source_ids, context.evidence_map)
        known_facts = known_facts_for_case(case, source_ids)
        missing_information = missing_information_for_case(case)
        if time_facts is None:
            time_facts = derive_time_facts(case)

        system_prompt = build_system_prompt(context, source_ids)
        user_prompt = build_user_prompt(case, context.policy_text, known_facts, missing_information, time_facts)
        # call_model defaults to the real Ollama call, but tests pass a fake
        # function with the same two-argument shape -- which is why the whole
        # pipeline can be tested without a server.
        raw_text = call_model(system_prompt, user_prompt)
        model_fields = validate_model_result(raw_text, valid_source_ids)
        return assemble_result(case_id, known_facts, missing_information, model_fields)
    except Exception as exc:
        # Any server error, timeout, bad JSON, bad shape, or unknown
        # source_id -> safe state. A blanket catch is the point: this tool
        # must never show a traceback or a half-finished draft. The reason
        # string keeps the cause visible.
        return unavailable_result(case_id, f"{type(exc).__name__}: {exc}")


def load_offline_result(case_id, demo_path=DEMO_PATH):
    """Load a labelled prerecorded synthetic result. No API call is made."""
    # Lets the demo run with no model at all -- on a plane, or as the known
    # policy-correct answer to compare a live run against.
    with open(demo_path, "r", encoding="utf-8") as fh:
        demo = json.load(fh)
    if case_id not in demo:
        raise KeyError(f"No offline demo result for case {case_id!r}.")
    return demo[case_id]


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def print_sections(case, context, result, note=None, time_facts=None):
    # Print the INPUTS before the output. A reviewer has to see the facts,
    # the context pack and the policy to judge the draft, so all of it is on
    # screen together.
    print("=" * 70)
    print("SUPPLIED FACTS")
    print("=" * 70)
    print(json.dumps(case, indent=2, ensure_ascii=False))
    print()

    print("=" * 70)
    print("SUPPLIED CONTEXT PACK (task.md, definitions.md, contractDefinitions.md)")
    print("=" * 70)
    print(context.task_text.rstrip())
    print()
    print(context.definitions_text.rstrip())
    print()
    print(context.contract_text.rstrip())
    print()

    # Show the code-computed values too. The reviewer needs the window verdict
    # to judge whether the draft actually followed it -- and it makes the split
    # visible: this block is arithmetic, the draft below is a model.
    if time_facts is not None:
        print("=" * 70)
        print("COMPUTED FACTS (calculated in code, not by the model)")
        print("=" * 70)
        print(json.dumps(time_facts, indent=2, ensure_ascii=False))
        print()

    print("=" * 70)
    print("SUPPLIED POLICY")
    print("=" * 70)
    print(context.policy_text.rstrip())  # rstrip avoids a trailing blank line
    print()

    print("=" * 70)
    print("DRAFT RESULT FOR HUMAN REVIEW")
    print("=" * 70)
    # The note labels a run that made no model call, so a prerecorded or
    # simulated result can never be mistaken for a live one.
    if note:
        print(note)
        print()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # On failure, repeat the fallback in plain prose under the JSON -- the
    # operator should not have to read a data structure to learn what to do.
    if result.get("review_status") == "UNAVAILABLE":
        print()
        print(result.get("manual_fallback", MANUAL_FALLBACK_MESSAGE))
    print()
    # Printed on every single run, success or failure, so the limits of the
    # tool are the last thing on screen.
    print(
        "Reminder: this is a DRAFT for human review only. Nothing was sent, "
        "no return or refund was approved, and no record was changed."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Meridian Retail support-draft review tool (drafts for human review only)."
    )
    # choices=VALID_CASES makes argparse reject a bad case ID with a usage
    # message before any file is read or any model is called.
    parser.add_argument("--case", choices=VALID_CASES, help="Case ID to draft for.")
    parser.add_argument(
        "--simulate-timeout",
        action="store_true",
        help="Make no model call; show the safe unavailable state.",
    )
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="Make no model call; use a prerecorded synthetic result.",
    )
    # argv=None means "read sys.argv"; tests pass a list instead.
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.case:
        # Return a code rather than raising: exit status 2 is the shell
        # convention for a usage error, and stderr keeps it out of piped output.
        print("Error: --case is required (choose C1, C2, or C3).", file=sys.stderr)
        return 2

    # Load the inputs before branching, so every mode prints the same
    # facts-and-context-pack around whatever result it produces.
    case = load_case(args.case)
    context = load_context_pack()
    # Computed once, then both shown to the reviewer and sent to the model, so
    # the two can never disagree about what time it is.
    time_facts = derive_time_facts(case)

    # Mode 1: pretend the model failed. Returns early, so there is no code
    # path from here to a model call -- the guarantee is structural, not a
    # promise in the docs.
    if args.simulate_timeout:
        result = unavailable_result(case["case_id"], "simulated timeout (no model call was made)")
        print_sections(
            case,
            context,
            result,
            note="MODE: --simulate-timeout. No model call was made.",
            time_facts=time_facts,
        )
        return 0

    # Mode 2: replay a known-good answer from disk. Also returns early, and
    # labels itself loudly so a demo result is never mistaken for a live one.
    if args.offline_demo:
        result = load_offline_result(args.case)
        print_sections(
            case,
            context,
            result,
            note="MODE: --offline-demo. NO MODEL API WAS CALLED. This is a "
            "labelled prerecorded synthetic result from demo_outputs.json.",
            time_facts=time_facts,
        )
        return 0

    # Mode 3: the real thing. generate_draft never raises, so there is no
    # try/except here -- a failure arrives as the UNAVAILABLE result and
    # prints through the same path as a success.
    result = generate_draft(case, context, time_facts=time_facts)
    print_sections(case, context, result, time_facts=time_facts)
    return 0


# Only runs when executed as a script, so `import app` in the tests does not
# trigger the CLI. SystemExit propagates main()'s return value as the exit code.
if __name__ == "__main__":
    raise SystemExit(main())
