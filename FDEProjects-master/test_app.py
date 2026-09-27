"""Tests for app.py. No test calls a real model, local or remote.

The model is always injected via a fake call_model function. The one test that
touches the real client points it at a dead local port on purpose, to prove the
failure path returns the safe state instead of a traceback.
"""

import json
from datetime import datetime

import app


def context_pack():
    return app.load_context_pack()


# A valid model response, generic across cases: the model's seven fields
# only. case_id, known_facts and missing_information are supplied by the app,
# not the model. Cites policy_register:policy.md, the one source_id every
# case's evidence map defines, so this fixture works for C1, C2 or C3.
VALID_C1_JSON = json.dumps(
    {
        "case_summary": "Customer reports a broken zip on a jacket; issue record is open.",
        "evidence_refs": ["P1"],
        "evidence_links": [
            {"claim": "Damage reported can be referred for human return review.", "source_id": "policy_register:policy.md"}
        ],
        "conflicting_information": [],
        "draft_reply": "Referring your jacket zip issue for human return review; this is not an approved return.",
        "review_status": "READY_FOR_HUMAN_REVIEW",
        "human_action_required": "A person should confirm the delivery date and review the case.",
    }
)


def fake_valid(system_prompt, user_prompt):
    return VALID_C1_JSON


def fake_invalid_json(system_prompt, user_prompt):
    return "sorry, here is your draft: not json at all"


def fake_wrong_shape(system_prompt, user_prompt):
    # Valid JSON, correct keys, but review_status is not an allowed value.
    data = json.loads(VALID_C1_JSON)
    data["review_status"] = "APPROVED"
    return json.dumps(data)


def fake_unknown_source_id(system_prompt, user_prompt):
    # Valid JSON and a valid review_status, but cites a source_id that does
    # not belong to this case.
    data = json.loads(VALID_C1_JSON)
    data["evidence_links"] = [{"claim": "made up", "source_id": "order_record:NOT-REAL"}]
    return json.dumps(data)


def fake_boom(system_prompt, user_prompt):
    raise TimeoutError("simulated provider timeout")


# --------------------------------------------------------------------------
# Successful C1 result
# --------------------------------------------------------------------------
def test_generate_draft_valid_c1():
    case = app.load_case("C1")
    result = app.generate_draft(case, context_pack(), call_model=fake_valid)
    assert result["case_id"] == "C1"
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["evidence_refs"] == ["P1"]
    assert result["missing_information"] == []
    assert isinstance(result["draft_reply"], str) and result["draft_reply"]
    # known_facts is code-derived, not copied from the model's response.
    fields = {fact["field"] for fact in result["known_facts"]}
    assert fields == {"order_id", "product", "delivery_date", "issue_status", "issue_details", "customer_question"}


def test_generate_draft_c2_missing_delivery_date():
    case = app.load_case("C2")
    result = app.generate_draft(case, context_pack(), call_model=fake_valid)
    assert result["missing_information"] == ["delivery_date"]
    fields = {fact["field"] for fact in result["known_facts"]}
    assert "delivery_date" not in fields


# --------------------------------------------------------------------------
# Invalid model output -> safe unavailable state
# --------------------------------------------------------------------------
def test_generate_draft_invalid_json_is_unavailable():
    case = app.load_case("C1")
    result = app.generate_draft(case, context_pack(), call_model=fake_invalid_json)
    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""
    assert result["case_id"] == "C1"


def test_generate_draft_wrong_enum_is_unavailable():
    case = app.load_case("C1")
    result = app.generate_draft(case, context_pack(), call_model=fake_wrong_shape)
    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""


def test_generate_draft_unknown_source_id_is_unavailable():
    case = app.load_case("C1")
    result = app.generate_draft(case, context_pack(), call_model=fake_unknown_source_id)
    assert result["review_status"] == "UNAVAILABLE"
    assert "source_id" in result["unavailable_reason"]


def test_validate_model_result_rejects_non_string_list():
    bad = json.dumps(
        {
            "case_summary": "x",
            "evidence_refs": [1, 2],
            "evidence_links": [],
            "conflicting_information": [],
            "draft_reply": "x",
            "review_status": "BLOCKED",
            "human_action_required": "x",
        }
    )
    try:
        app.validate_model_result(bad, valid_source_ids=frozenset())
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# Timeout / provider failure -> safe unavailable state, no draft
# --------------------------------------------------------------------------
def test_generate_draft_timeout_is_unavailable():
    case = app.load_case("C2")
    result = app.generate_draft(case, context_pack(), call_model=fake_boom)
    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""


def test_simulate_timeout_makes_no_call(capsys):
    # main() with --simulate-timeout must never call the local model.
    rc = app.main(["--case", "C1", "--simulate-timeout"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No model call was made" in out
    assert "UNAVAILABLE" in out


# --------------------------------------------------------------------------
# All input files (including the context pack) load for all three cases
# --------------------------------------------------------------------------
def test_all_cases_load():
    for case_id in ("C1", "C2", "C3"):
        case = app.load_case(case_id)
        assert case["case_id"] == case_id
    policy = app.load_policy()
    for pid in ("P1", "P2", "P3", "P4"):
        assert pid in policy


def test_context_pack_loads_all_files():
    context = context_pack()
    assert "Evidence" in context.task_text
    assert "source_id" in context.definitions_text
    assert "known_facts" in context.contract_text
    for case_id in ("C1", "C2", "C3"):
        assert case_id in context.evidence_map


# --------------------------------------------------------------------------
# Offline demo: no API call, labelled prerecorded result
# --------------------------------------------------------------------------
def test_offline_demo_loads_all_cases():
    for case_id in ("C1", "C2", "C3"):
        result = app.load_offline_result(case_id)
        assert result["review_status"] in app.VALID_REVIEW_STATUS
        assert isinstance(result["draft_reply"], str)
        assert result["case_id"] == case_id


def test_offline_demo_prints_no_api_line(capsys):
    rc = app.main(["--case", "C1", "--offline-demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NO MODEL API WAS CALLED" in out


# --------------------------------------------------------------------------
# The safe state explains itself (server down, model not pulled, bad shape)
# --------------------------------------------------------------------------
def test_unavailable_result_carries_reason():
    result = app.unavailable_result(case_id="C1", reason="ConnectionError: boom")
    assert result["review_status"] == "UNAVAILABLE"
    assert result["unavailable_reason"] == "ConnectionError: boom"
    assert result["draft_reply"] == ""
    assert result["case_id"] == "C1"


def test_unavailable_result_without_reason_omits_key():
    assert "unavailable_reason" not in app.unavailable_result()


def test_generate_draft_records_why_it_failed():
    case = app.load_case("C1")
    result = app.generate_draft(case, context_pack(), call_model=fake_boom)
    assert result["review_status"] == "UNAVAILABLE"
    assert "TimeoutError" in result["unavailable_reason"]


# --------------------------------------------------------------------------
# Unreachable Ollama server -> safe state, never a traceback
# --------------------------------------------------------------------------
def test_unreachable_server_is_unavailable(monkeypatch):
    # Port 1 on loopback refuses instantly; no network egress, no real model.
    monkeypatch.setenv("APP_OLLAMA_HOST", "http://127.0.0.1:1")
    monkeypatch.setenv("APP_OLLAMA_TIMEOUT", "2")
    case = app.load_case("C1")
    result = app.generate_draft(case, context_pack())
    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""
    assert "Ollama" in result["unavailable_reason"]


# --------------------------------------------------------------------------
# The schema handed to Ollama matches what validate_model_result demands
# --------------------------------------------------------------------------
def test_schema_and_validator_agree():
    assert set(app.MODEL_SCHEMA["properties"]) == set(app.REQUIRED_MODEL_KEYS)
    assert set(app.MODEL_SCHEMA["properties"]["review_status"]["enum"]) == app.VALID_REVIEW_STATUS
    # UNAVAILABLE is the app's own state; the model must never be able to claim it.
    assert "UNAVAILABLE" not in app.VALID_REVIEW_STATUS


def test_required_result_keys_match_the_contract():
    # case_id, known_facts and missing_information are app-supplied; the rest
    # come from the model. Together they are the full Appendix E contract.
    assert app.REQUIRED_RESULT_KEYS == app.REQUIRED_MODEL_KEYS | {
        "case_id",
        "known_facts",
        "missing_information",
    }
    result = app.generate_draft(app.load_case("C1"), context_pack(), call_model=fake_valid)
    assert set(result.keys()) == app.REQUIRED_RESULT_KEYS


# --------------------------------------------------------------------------
# Evidence: source IDs, known facts, and evidence_map.json integrity
# --------------------------------------------------------------------------
def test_source_ids_for_case():
    ids = app.source_ids_for_case("C1", "MR-1042")
    assert ids == {
        "support_queue": "support_queue:C1",
        "order_record": "order_record:MR-1042",
        "issue_record": "issue_record:C1",
        "policy_register": "policy_register:policy.md",
    }


def test_valid_source_ids_for_case_matches_evidence_map():
    context = context_pack()
    ids = app.source_ids_for_case("C1", "MR-1042")
    valid = app.valid_source_ids_for_case("C1", ids, context.evidence_map)
    assert valid == frozenset(context.evidence_map["C1"])


def test_valid_source_ids_raises_for_unknown_case():
    ids = app.source_ids_for_case("C1", "MR-1042")
    try:
        app.valid_source_ids_for_case("C1", ids, evidence_map={})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_known_facts_for_case_omits_missing_delivery_date():
    case = app.load_case("C2")
    ids = app.source_ids_for_case("C2", case["order_id"])
    facts = app.known_facts_for_case(case, ids)
    fields = {fact["field"] for fact in facts}
    assert "delivery_date" not in fields
    assert app.missing_information_for_case(case) == ["delivery_date"]


def test_known_facts_for_case_includes_delivery_date_when_present():
    case = app.load_case("C1")
    ids = app.source_ids_for_case("C1", case["order_id"])
    facts = app.known_facts_for_case(case, ids)
    delivery_facts = [fact for fact in facts if fact["field"] == "delivery_date"]
    assert delivery_facts == [{"field": "delivery_date", "value": "2026-09-12", "source_id": "order_record:MR-1042"}]
    assert app.missing_information_for_case(case) == []


# --------------------------------------------------------------------------
# The 78-hour delivery window is computed in code, with an injected clock so
# these tests never depend on the real date.
# --------------------------------------------------------------------------
FIXED_NOW = datetime(2026, 9, 14, 12, 0, 0)  # noon, two days after C1 delivery


def test_hours_since_delivery_exact():
    case = {"delivery_date": "2026-09-14"}
    assert app.hours_since_delivery(case, now=FIXED_NOW) == 12.0


def test_hours_since_delivery_is_none_without_a_date():
    for value in (None, ""):
        assert app.hours_since_delivery({"delivery_date": value}, now=FIXED_NOW) is None
    assert app.hours_since_delivery({}, now=FIXED_NOW) is None


def test_window_boundary_at_78_hours():
    # 77.9h in -> inside; exactly 78h and beyond -> outside.
    delivered = datetime(2026, 9, 12, 0, 0, 0)
    case = {"delivery_date": delivered.isoformat()}

    just_inside = delivered.replace(day=15, hour=5, minute=54)   # 77.9h
    assert app.derive_time_facts(case, now=just_inside)["window_verdict"] == app.WINDOW_WITHIN

    exactly_78 = delivered.replace(day=15, hour=6, minute=0)     # 78.0h
    assert app.derive_time_facts(case, now=exactly_78)["window_verdict"] == app.WINDOW_OUTSIDE


def test_verdict_for_each_real_case():
    expected = {
        "C1": app.WINDOW_OUTSIDE,   # delivered 2026-09-12
        "C2": app.WINDOW_UNKNOWN,   # delivery_date is null
        "C3": app.WINDOW_OUTSIDE,   # delivered 2026-09-10
    }
    now = datetime(2026, 9, 21, 22, 0, 0)
    for case_id, verdict in expected.items():
        facts = app.derive_time_facts(app.load_case(case_id), now=now)
        assert facts["window_verdict"] == verdict


def test_malformed_date_becomes_the_safe_state():
    # A corrupt record must not produce a confident reply. Uses a real case
    # ID (so evidence_map.json resolves normally) with a corrupted date.
    case = dict(app.load_case("C1"))
    case["delivery_date"] = "not-a-date"
    result = app.generate_draft(case, context_pack(), call_model=fake_valid)
    assert result["review_status"] == "UNAVAILABLE"
    assert "ValueError" in result["unavailable_reason"]


# --------------------------------------------------------------------------
# The computed facts and the rule both reach the model
# --------------------------------------------------------------------------
def test_computed_facts_are_in_the_user_prompt():
    case = app.load_case("C1")
    ids = app.source_ids_for_case("C1", case["order_id"])
    known_facts = app.known_facts_for_case(case, ids)
    missing_information = app.missing_information_for_case(case)
    facts = app.derive_time_facts(case, now=FIXED_NOW)
    prompt = app.build_user_prompt(case, app.load_policy(), known_facts, missing_information, facts)
    assert "COMPUTED FACTS" in prompt
    assert "window_verdict" in prompt
    assert facts["window_verdict"] in prompt
    assert "do not recompute" in prompt
    assert "SUPPLIED KNOWN FACTS" in prompt
    assert "order_record:MR-1042" in prompt


def test_system_prompt_states_the_rule_for_every_verdict():
    context = context_pack()
    ids = app.source_ids_for_case("C1", "MR-1042")
    system = app.build_system_prompt(context, ids)
    for verdict in (app.WINDOW_UNKNOWN, app.WINDOW_WITHIN, app.WINDOW_OUTSIDE):
        assert verdict in system
    assert "do NOT do any date arithmetic" in system
    # The context pack is embedded verbatim, not paraphrased.
    assert "Prepare a structured case review" in system
    assert "order_record: order_record:MR-1042" in system


def test_generate_draft_uses_the_time_facts_it_is_given():
    seen = {}

    def spy(system_prompt, user_prompt):
        seen["user"] = user_prompt
        return VALID_C1_JSON

    facts = app.derive_time_facts(app.load_case("C1"), now=FIXED_NOW)
    app.generate_draft(app.load_case("C1"), context_pack(), call_model=spy, time_facts=facts)
    # The injected clock, not the real one, must be what the model saw.
    assert "2026-09-14T12:00:00" in seen["user"]


def test_printed_facts_include_the_verdict(capsys):
    rc = app.main(["--case", "C1", "--simulate-timeout"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "COMPUTED FACTS (calculated in code, not by the model)" in out
    assert "window_verdict" in out
    assert "SUPPLIED CONTEXT PACK" in out
