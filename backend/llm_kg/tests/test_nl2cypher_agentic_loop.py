from __future__ import annotations

import pytest

from backend.llm_kg import nl2cypher


def _schema() -> nl2cypher.SchemaSnapshot:
    return nl2cypher.SchemaSnapshot(
        labels=["Organization", "Product", "FinancialMetric", "FiscalPeriod"],
        relationship_types=["PRODUCES", "HAS_FINANCIAL_METRIC", "FOR_PERIOD"],
        properties={"Organization": ["name"]},
        schema_text="Node Labels:\nOrganization, Product\n\nRelationships:\n- (Organization)-[:PRODUCES]->(Product)",
    )


def _plan() -> dict[str, object]:
    return {
        "intent": "list_business",
        "strategy": "single_query",
        "must_have_paths": [],
        "forbidden_patterns": [],
        "output_contract": {"columns": ["organization", "product"]},
        "risk_hypotheses": [],
    }


def _critic_accept() -> dict[str, object]:
    return {"verdict": "accept", "issues": [], "repair_actions": [], "next_strategy": "single_query"}


def _critic_replan() -> dict[str, object]:
    return {
        "verdict": "replan",
        "issues": [{"code": "NEEDS_REPAIR", "message": "retry", "severity": "med"}],
        "repair_actions": ["retry"],
        "next_strategy": "single_query",
    }


def test_agentic_loop_round1_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nl2cypher, "_run_planner_agent", lambda **_kwargs: _plan())
    monkeypatch.setattr(
        nl2cypher,
        "_run_reactor_agent",
        lambda **_kwargs: {
            "cypher": "MATCH (o:Organization)-[:PRODUCES]->(p:Product) RETURN o.name AS organization, p.name AS product",
            "assumptions": [],
            "self_checks": {"schema_grounded": True, "projection_consistent": True},
        },
    )
    monkeypatch.setattr(nl2cypher, "_run_deterministic_checks", lambda **kwargs: (kwargs["cypher"], []))
    monkeypatch.setattr(nl2cypher, "execute_cypher", lambda _driver, _cypher: [{"organization": "鴻海", "product": "AI Server"}])
    monkeypatch.setattr(nl2cypher, "_run_critic_agent", lambda **_kwargs: _critic_accept())

    result = nl2cypher._run_agentic_query_loop(
        question="鴻海的事業有哪些",
        schema=_schema(),
        driver=object(),
        entity_names=["鴻海精密"],
        organization_names=["鴻海精密"],
        is_finance_question=False,
    )

    assert result["attempt"] == 1
    assert result["rows"] == [{"organization": "鴻海", "product": "AI Server"}]
    assert result["agentic_trace"]["replan_count"] == 0
    assert result["agentic_trace"]["stage"] == "done"


def test_agentic_loop_repairs_union_issue_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nl2cypher, "_run_planner_agent", lambda **_kwargs: _plan())

    reactor_outputs = iter(
        [
            {
                "cypher": "BAD_UNION",
                "assumptions": ["first try"],
                "self_checks": {"schema_grounded": True, "projection_consistent": False},
            },
            {
                "cypher": "GOOD_QUERY",
                "assumptions": ["second try"],
                "self_checks": {"schema_grounded": True, "projection_consistent": True},
            },
        ]
    )

    def fake_reactor(**_kwargs):
        return next(reactor_outputs)

    monkeypatch.setattr(nl2cypher, "_run_reactor_agent", fake_reactor)

    check_outputs = iter(
        [
            (
                "BAD_UNION",
                [nl2cypher.DeterministicIssue(code="UNION_RETURN_SHAPE", message="UNION mismatch", severity="high")],
            ),
            ("GOOD_QUERY", []),
        ]
    )

    def fake_checks(**_kwargs):
        return next(check_outputs)

    monkeypatch.setattr(nl2cypher, "_run_deterministic_checks", fake_checks)
    monkeypatch.setattr(
        nl2cypher,
        "execute_cypher",
        lambda _driver, cypher: [{"organization": "鴻海", "product": "EV"}] if cypher == "GOOD_QUERY" else [],
    )

    critic_outputs = iter([_critic_replan(), _critic_accept()])
    monkeypatch.setattr(nl2cypher, "_run_critic_agent", lambda **_kwargs: next(critic_outputs))
    monkeypatch.setattr(
        nl2cypher,
        "_run_replanner_agent",
        lambda **_kwargs: {
            "strategy": "single_query",
            "delta_actions": ["align alias"],
            "tightened_constraints": [],
            "stop_if": [],
        },
    )

    result = nl2cypher._run_agentic_query_loop(
        question="鴻海的事業有哪些",
        schema=_schema(),
        driver=object(),
        entity_names=[],
        organization_names=[],
        is_finance_question=False,
    )

    assert result["attempt"] == 2
    assert result["agentic_trace"]["replan_count"] == 1
    assert result["rows"] == [{"organization": "鴻海", "product": "EV"}]


def test_agentic_loop_handles_planner_failure_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nl2cypher,
        "_run_planner_agent",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("planner malformed json")),
    )
    monkeypatch.setattr(
        nl2cypher,
        "_run_reactor_agent",
        lambda **_kwargs: {
            "cypher": "MATCH (o:Organization) RETURN o.name AS organization",
            "assumptions": [],
            "self_checks": {"schema_grounded": True, "projection_consistent": True},
        },
    )
    monkeypatch.setattr(nl2cypher, "_run_deterministic_checks", lambda **kwargs: (kwargs["cypher"], []))
    monkeypatch.setattr(nl2cypher, "execute_cypher", lambda _driver, _cypher: [{"organization": "鴻海"}])
    monkeypatch.setattr(nl2cypher, "_run_critic_agent", lambda **_kwargs: _critic_accept())

    result = nl2cypher._run_agentic_query_loop(
        question="鴻海的事業有哪些",
        schema=_schema(),
        driver=object(),
        entity_names=[],
        organization_names=[],
        is_finance_question=False,
    )

    assert result["attempt"] == 1
    assert any("planner_error" in item for item in result["agentic_trace"]["failure_chain"])


def test_agentic_loop_fail_fast_on_repeated_cypher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nl2cypher, "_run_planner_agent", lambda **_kwargs: _plan())
    monkeypatch.setattr(
        nl2cypher,
        "_run_reactor_agent",
        lambda **_kwargs: {
            "cypher": "SAME_QUERY",
            "assumptions": [],
            "self_checks": {"schema_grounded": True, "projection_consistent": True},
        },
    )
    monkeypatch.setattr(nl2cypher, "_run_deterministic_checks", lambda **kwargs: (kwargs["cypher"], []))
    monkeypatch.setattr(nl2cypher, "execute_cypher", lambda _driver, _cypher: [])

    def fake_critic(*, repeated: bool, **_kwargs):
        if repeated:
            return {
                "verdict": "fail_fast",
                "issues": [{"code": "NO_PROGRESS", "message": "repeated cypher", "severity": "high"}],
                "repair_actions": [],
                "next_strategy": "single_query",
            }
        return _critic_replan()

    monkeypatch.setattr(nl2cypher, "_run_critic_agent", fake_critic)
    monkeypatch.setattr(
        nl2cypher,
        "_run_replanner_agent",
        lambda **_kwargs: {
            "strategy": "single_query",
            "delta_actions": ["retry"],
            "tightened_constraints": [],
            "stop_if": [],
        },
    )

    with pytest.raises(RuntimeError, match=r"Cypher generation failed after retries: "):
        nl2cypher._run_agentic_query_loop(
            question="鴻海的事業有哪些",
            schema=_schema(),
            driver=object(),
            entity_names=[],
            organization_names=[],
            is_finance_question=False,
        )


def test_agentic_loop_finance_fallback_after_exhausted_replans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nl2cypher, "NL2CYPHER_AGENTIC_MAX_ROUNDS", 2)
    monkeypatch.setattr(nl2cypher, "_run_planner_agent", lambda **_kwargs: _plan())
    monkeypatch.setattr(
        nl2cypher,
        "_run_reactor_agent",
        lambda **_kwargs: {
            "cypher": "INVALID_FINANCE_QUERY",
            "assumptions": [],
            "self_checks": {"schema_grounded": False, "projection_consistent": False},
        },
    )
    monkeypatch.setattr(
        nl2cypher,
        "_run_deterministic_checks",
        lambda **kwargs: (
            kwargs["cypher"],
            [
                nl2cypher.DeterministicIssue(
                    code="FINANCE_REQUIRED_PATH_MISSING",
                    message="Finance query must use HAS_FINANCIAL_METRIC and FOR_PERIOD relationships.",
                    severity="high",
                )
            ],
        ),
    )
    monkeypatch.setattr(nl2cypher, "_run_critic_agent", lambda **_kwargs: _critic_replan())
    monkeypatch.setattr(
        nl2cypher,
        "_run_replanner_agent",
        lambda **_kwargs: {
            "strategy": "single_query",
            "delta_actions": ["use finance path"],
            "tightened_constraints": [],
            "stop_if": [],
        },
    )
    monkeypatch.setattr(nl2cypher, "_build_finance_template_cypher", lambda _question, _entities: "FINANCE_TEMPLATE")
    monkeypatch.setattr(
        nl2cypher,
        "execute_cypher",
        lambda _driver, cypher: [{"organization": "鴻海", "metric": "REVENUE"}] if cypher == "FINANCE_TEMPLATE" else [],
    )

    result = nl2cypher._run_agentic_query_loop(
        question="鴻海 2025Q2 營收是多少",
        schema=_schema(),
        driver=object(),
        entity_names=[],
        organization_names=["鴻海"],
        is_finance_question=True,
    )

    assert result["cypher"] == "FINANCE_TEMPLATE"
    assert result["attempt"] == 2
    assert "after retries" in result["reason"]
    assert result["agentic_trace"]["stage"] == "exhausted"


def test_agentic_loop_failure_message_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nl2cypher, "_run_planner_agent", lambda **_kwargs: _plan())
    monkeypatch.setattr(
        nl2cypher,
        "_run_reactor_agent",
        lambda **_kwargs: {
            "cypher": "BAD_QUERY",
            "assumptions": [],
            "self_checks": {"schema_grounded": False, "projection_consistent": False},
        },
    )
    monkeypatch.setattr(
        nl2cypher,
        "_run_deterministic_checks",
        lambda **kwargs: (kwargs["cypher"], [nl2cypher.DeterministicIssue(code="BAD", message="bad query", severity="high")]),
    )
    monkeypatch.setattr(
        nl2cypher,
        "_run_critic_agent",
        lambda **_kwargs: {
            "verdict": "fail_fast",
            "issues": [{"code": "BAD", "message": "bad query", "severity": "high"}],
            "repair_actions": [],
            "next_strategy": "single_query",
        },
    )

    with pytest.raises(RuntimeError, match=r"Cypher generation failed after retries: bad query"):
        nl2cypher._run_agentic_query_loop(
            question="鴻海的事業有哪些",
            schema=_schema(),
            driver=object(),
            entity_names=[],
            organization_names=[],
            is_finance_question=False,
        )


def test_call_agent_json_retries_on_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_chat_json(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise nl2cypher.llm_client.LLMParseError("Invalid JSON from LLM: unterminated string")
        return {"ok": True}

    monkeypatch.setattr(nl2cypher.llm_client, "chat_json", fake_chat_json)

    result = nl2cypher._call_agent_json(
        system_prompt="system",
        user_prompt="user",
        max_tokens=128,
        nl2cypher_provider="gemini",
        nl2cypher_model="gemini-3-pro-preview",
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    assert int(calls[1]["max_tokens"]) > int(calls[0]["max_tokens"])
    retry_user_prompt = calls[1]["messages"][1]["content"]  # type: ignore[index]
    assert "前次輸出無法解析為合法 JSON object" in retry_user_prompt


def test_call_agent_json_fail_fast_when_gemini_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"value": 0}

    def fake_chat_json(**_kwargs):
        call_count["value"] += 1
        raise nl2cypher.llm_client.LLMResponseError("GEMINI_API_KEY is required when provider=gemini")

    monkeypatch.setattr(nl2cypher.llm_client, "chat_json", fake_chat_json)

    with pytest.raises(nl2cypher.llm_client.LLMResponseError, match=r"GEMINI_API_KEY is required"):
        nl2cypher._call_agent_json(
            system_prompt="system",
            user_prompt="user",
            max_tokens=128,
            nl2cypher_provider="gemini",
            nl2cypher_model="gemini-3-pro-preview",
        )

    assert call_count["value"] == 1
