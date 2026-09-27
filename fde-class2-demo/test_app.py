import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app


def test_parse_and_validate_accepts_well_formed_json():
    raw = json.dumps(
        {
            "draft_reply": "Thanks for reaching out.",
            "evidence_refs": ["P1", "P2"],
            "missing_information": [],
            "review_status": "READY_FOR_HUMAN_REVIEW",
        }
    )
    result = app.parse_and_validate(raw)
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["evidence_refs"] == ["P1", "P2"]


def test_generate_draft_result_success_for_c1():
    case = app.load_case("C1")
    policy = app.load_policy()

    def fake_model_fn(prompt):
        assert "MR-1042" in prompt
        return json.dumps(
            {
                "draft_reply": "We can refer your jacket for human return review.",
                "evidence_refs": ["P1"],
                "missing_information": [],
                "review_status": "READY_FOR_HUMAN_REVIEW",
            }
        )

    result = app.generate_draft_result(case, policy, fake_model_fn)
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["draft_reply"]


@pytest.mark.parametrize(
    "bad_raw",
    [
        "not json at all",
        json.dumps({"draft_reply": "x", "evidence_refs": [], "missing_information": []}),
        json.dumps(
            {
                "draft_reply": 123,
                "evidence_refs": [],
                "missing_information": [],
                "review_status": "BLOCKED",
            }
        ),
        json.dumps(
            {
                "draft_reply": "x",
                "evidence_refs": "P1",
                "missing_information": [],
                "review_status": "BLOCKED",
            }
        ),
        json.dumps(
            {
                "draft_reply": "x",
                "evidence_refs": [],
                "missing_information": [],
                "review_status": "APPROVED",
            }
        ),
    ],
)
def test_parse_and_validate_rejects_invalid_output(bad_raw):
    with pytest.raises(app.DraftGenerationError):
        app.parse_and_validate(bad_raw)


def test_timeout_behaviour_falls_back_safely():
    case = app.load_case("C1")
    policy = app.load_policy()

    def timing_out_model_fn(prompt):
        raise TimeoutError("simulated network timeout")

    with pytest.raises(TimeoutError):
        app.generate_draft_result(case, policy, timing_out_model_fn)

    # This mirrors the try/except safety net in app.run().
    try:
        app.generate_draft_result(case, policy, timing_out_model_fn)
        result = None
    except Exception as exc:
        result = app.safe_unavailable_result(f"Draft generation failed: {exc}")

    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""
    assert "manual_fallback_message" in result


def test_load_case_returns_expected_fields_for_all_cases():
    c1 = app.load_case("C1")
    assert c1["case_id"] == "C1"
    assert c1["order_id"] == "MR-1042"
    assert c1["delivery_date"] == "2026-09-15"

    c2 = app.load_case("C2")
    assert c2["case_id"] == "C2"
    assert c2["delivery_date"] is None

    c3 = app.load_case("C3")
    assert c3["case_id"] == "C3"
    assert c3["issue_status"] == "open"


def test_load_case_rejects_unknown_case_id():
    with pytest.raises(ValueError):
        app.load_case("C99")


def test_load_policy_contains_all_policy_ids():
    policy = app.load_policy()
    for policy_id in ("P1", "P2", "P3", "P4"):
        assert policy_id in policy


def test_load_offline_demo_returns_expected_shape_for_each_case():
    for case_id in ("C1", "C2", "C3"):
        demo = app.load_offline_demo(case_id)
        assert set(demo.keys()) == {
            "draft_reply",
            "evidence_refs",
            "missing_information",
            "review_status",
        }
        assert demo["review_status"] in app.VALID_REVIEW_STATUSES


def test_offline_demo_mode_never_calls_the_sdk(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("call_model_via_sdk must not be called in offline demo mode")

    monkeypatch.setattr(app, "call_model_via_sdk", boom)
    result = app.run("C1", offline_demo=True)
    assert result["review_status"] in app.VALID_REVIEW_STATUSES


def test_simulate_timeout_mode_never_calls_the_sdk(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("call_model_via_sdk must not be called when simulating a timeout")

    monkeypatch.setattr(app, "call_model_via_sdk", boom)
    result = app.run("C1", simulate_timeout=True)
    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""
    assert "manual_fallback_message" in result


def test_run_falls_back_safely_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("APP_ANTHROPIC_API_KEY", raising=False)
    result = app.run("C1")
    assert result["review_status"] == "UNAVAILABLE"
    assert result["draft_reply"] == ""


def test_missing_delivery_date_short_circuits_before_any_model_call(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("call_model_via_sdk must not be called when delivery_date is missing")

    monkeypatch.setattr(app, "call_model_via_sdk", boom)
    monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "sk-should-not-be-used")

    result = app.run("C2")  # C2's delivery_date is null in cases.json

    assert result == app.missing_delivery_date_result()
    assert result["review_status"] == "NEEDS_INFORMATION"
    assert result["evidence_refs"] == ["P2", "P4"]
    assert result["missing_information"] == ["delivery_date"]


def test_has_missing_delivery_date_treats_null_and_empty_string_as_missing():
    assert app.has_missing_delivery_date({"delivery_date": None}) is True
    assert app.has_missing_delivery_date({"delivery_date": ""}) is True
    assert app.has_missing_delivery_date({"delivery_date": "2026-09-15"}) is False


def test_present_delivery_date_does_not_short_circuit(monkeypatch):
    calls = {"count": 0}

    def fake_sdk_call(client, model, prompt):
        calls["count"] += 1
        return json.dumps(
            {
                "draft_reply": "Referring your case for human return review.",
                "evidence_refs": ["P1"],
                "missing_information": [],
                "review_status": "READY_FOR_HUMAN_REVIEW",
            }
        )

    class FakeClient:
        pass

    monkeypatch.setattr(app, "call_model_via_sdk", fake_sdk_call)
    monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "sk-test")

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: FakeClient())

    result = app.run("C1")  # C1 has a delivery_date

    assert calls["count"] == 1
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
