#!/usr/bin/env python3
"""Meridian Retail support-draft review tool.

Drafts a candidate support reply for HUMAN REVIEW ONLY. This tool never
sends a message, approves a refund, or changes any record.
"""

import argparse
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CASES_PATH = APP_DIR / "cases.json"
POLICY_PATH = APP_DIR / "policy.md"
DEMO_OUTPUTS_PATH = APP_DIR / "demo_outputs.json"

VALID_CASE_IDS = ("C1", "C2", "C3")
VALID_REVIEW_STATUSES = ("READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", "BLOCKED")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

MANUAL_FALLBACK_MESSAGE = (
    "The drafting assistant is unavailable. Please write this reply manually "
    "using the supplied facts and policy above, and route it through normal "
    "human review."
)


class DraftGenerationError(Exception):
    """Raised when a model response cannot be turned into a valid draft result."""


def load_case(case_id):
    if case_id not in VALID_CASE_IDS:
        raise ValueError(f"Unknown case id: {case_id!r}. Expected one of {VALID_CASE_IDS}.")
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if case_id not in cases:
        raise ValueError(f"Case {case_id!r} not found in {CASES_PATH.name}.")
    return cases[case_id]


def load_policy():
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_offline_demo(case_id):
    with open(DEMO_OUTPUTS_PATH, "r", encoding="utf-8") as f:
        demos = json.load(f)
    if case_id not in demos:
        raise ValueError(f"No offline demo output for case {case_id!r}.")
    return demos[case_id]


def build_prompt(case, policy):
    return (
        "You are drafting a customer support reply for HUMAN REVIEW ONLY. "
        "You cannot approve refunds, send messages, or change records.\n\n"
        "Use ONLY the supplied facts and the supplied policy below. Do not "
        "assume any fact that is not supplied. Do not state or imply that a "
        "refund has been approved. Do not say an issue is resolved if its "
        "status is open.\n\n"
        "SUPPLIED POLICY:\n"
        f"{policy}\n\n"
        "SUPPLIED FACTS (case record):\n"
        f"{json.dumps(case, indent=2)}\n\n"
        "Respond with ONLY a JSON object (no markdown fences, no commentary) "
        "with exactly these keys:\n"
        '  "draft_reply": string - the drafted reply text\n'
        '  "evidence_refs": list of strings - policy IDs (e.g. "P1") that '
        "support the draft\n"
        '  "missing_information": list of strings - field names still needed '
        "before this case can proceed\n"
        '  "review_status": one of "READY_FOR_HUMAN_REVIEW", '
        '"NEEDS_INFORMATION", "BLOCKED"\n'
    )


def call_model_via_sdk(client, model, prompt):
    """Thin wrapper around the Anthropic SDK call."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def parse_and_validate(raw_text):
    """Parse model output text and enforce the exact result schema.

    Correct structure does not prove correct content; the caller is still
    responsible for human review of the returned draft.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DraftGenerationError(f"Model output was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise DraftGenerationError("Model output JSON was not an object.")

    required_keys = {"draft_reply", "evidence_refs", "missing_information", "review_status"}
    missing_keys = required_keys - data.keys()
    if missing_keys:
        raise DraftGenerationError(f"Model output missing keys: {sorted(missing_keys)}")

    if not isinstance(data["draft_reply"], str):
        raise DraftGenerationError("draft_reply must be a string.")

    if not (
        isinstance(data["evidence_refs"], list)
        and all(isinstance(x, str) for x in data["evidence_refs"])
    ):
        raise DraftGenerationError("evidence_refs must be a list of strings.")

    if not (
        isinstance(data["missing_information"], list)
        and all(isinstance(x, str) for x in data["missing_information"])
    ):
        raise DraftGenerationError("missing_information must be a list of strings.")

    if data["review_status"] not in VALID_REVIEW_STATUSES:
        raise DraftGenerationError(
            f"review_status must be one of {VALID_REVIEW_STATUSES}, "
            f"got {data['review_status']!r}."
        )

    return {
        "draft_reply": data["draft_reply"],
        "evidence_refs": list(data["evidence_refs"]),
        "missing_information": list(data["missing_information"]),
        "review_status": data["review_status"],
    }


def generate_draft_result(case, policy, model_fn):
    """Call model_fn(prompt) and validate its output.

    model_fn is any callable of the form text -> raw response text. This is
    decoupled from the Anthropic SDK so it can be replaced with a fake
    function in tests.
    """
    prompt = build_prompt(case, policy)
    raw_text = model_fn(prompt)
    return parse_and_validate(raw_text)


def has_missing_delivery_date(case):
    delivery_date = case.get("delivery_date")
    return delivery_date is None or delivery_date == ""


def missing_delivery_date_result():
    return {
        "draft_reply": (
            "Before Meridian Retail can assess whether this case falls "
            "within the two-day damage-reporting rule, please provide the "
            "delivery date. A human support agent will review the request "
            "once that information is available."
        ),
        "evidence_refs": ["P2", "P4"],
        "missing_information": ["delivery_date"],
        "review_status": "NEEDS_INFORMATION",
    }


def safe_unavailable_result(reason):
    return {
        "draft_reply": "",
        "evidence_refs": [],
        "missing_information": [],
        "review_status": "UNAVAILABLE",
        "reason": reason,
        "manual_fallback_message": MANUAL_FALLBACK_MESSAGE,
    }


def print_section(title):
    print()
    print(f"=== {title} ===")


def print_supplied_facts(case):
    print_section("SUPPLIED FACTS")
    for key, value in case.items():
        print(f"{key}: {value}")


def print_supplied_policy(policy):
    print_section("SUPPLIED POLICY")
    print(policy.strip())


def print_draft_result(result, offline_demo=False):
    print_section("DRAFT RESULT FOR HUMAN REVIEW")
    if offline_demo:
        print(
            "[OFFLINE DEMO MODE] No model API was called. This is a "
            "prerecorded, synthetic result for demonstration only."
        )
    print(f"review_status: {result['review_status']}")
    print(f"draft_reply: {result['draft_reply']}")
    print(f"evidence_refs: {result['evidence_refs']}")
    print(f"missing_information: {result['missing_information']}")
    if "reason" in result:
        print(f"reason: {result['reason']}")
    if "manual_fallback_message" in result:
        print(f"manual_fallback_message: {result['manual_fallback_message']}")
    print()
    print(
        "This draft is for human review only. It has not been sent, and no "
        "refund, message, or record change has been made."
    )


def run(case_id, simulate_timeout=False, offline_demo=False):
    case = load_case(case_id)
    policy = load_policy()

    print_supplied_facts(case)
    print_supplied_policy(policy)

    if simulate_timeout:
        result = safe_unavailable_result(
            "Simulated timeout (--simulate-timeout); no API call was made."
        )
        print_draft_result(result)
        return result

    if offline_demo:
        result = load_offline_demo(case_id)
        print_draft_result(result, offline_demo=True)
        return result

    if has_missing_delivery_date(case):
        result = missing_delivery_date_result()
        print_draft_result(result)
        return result

    try:
        api_key = os.environ.get("APP_ANTHROPIC_API_KEY")
        if not api_key:
            raise DraftGenerationError("APP_ANTHROPIC_API_KEY is not set.")
        model = os.environ.get("APP_ANTHROPIC_MODEL", DEFAULT_MODEL)

        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=20.0, max_retries=0)
        result = generate_draft_result(
            case, policy, lambda prompt: call_model_via_sdk(client, model, prompt)
        )
    except Exception as exc:
        result = safe_unavailable_result(f"Draft generation failed: {exc}")

    print_draft_result(result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Meridian Retail support-draft review tool.")
    parser.add_argument("--case", required=True, choices=VALID_CASE_IDS)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--simulate-timeout", action="store_true")
    mode_group.add_argument("--offline-demo", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(APP_DIR / ".env")
    except ImportError:
        pass
    run(args.case, simulate_timeout=args.simulate_timeout, offline_demo=args.offline_demo)


if __name__ == "__main__":
    main()
