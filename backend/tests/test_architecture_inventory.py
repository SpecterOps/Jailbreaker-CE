import json
import sys

sys.path.insert(0, "backend")

from app.attacklib.architecture_inventory import (
    ARCHITECTURE_SCHEMA_VERSION,
    build_architecture_artifact,
    stable_json_dumps,
)


def _phase_response(phase: str, **overrides):
    base = {
        "phase": phase,
        "agent": {
            "name": None,
            "profile": None,
            "model_aliases": [],
            "confidence": "unknown",
        },
        "tools": [],
        "data_stores": [],
        "workflow": {
            "framework": None,
            "nodes": [],
            "edges": [],
            "handoff_paths": [],
        },
        "limits": {
            "write_access": None,
            "network_access": None,
            "persistence": None,
            "cross_agent_handoff": None,
        },
        "unknowns": [],
    }
    base.update(overrides)
    return base


def test_build_architecture_artifact_merges_and_sorts_phase_outputs() -> None:
    rounds = [
        {
            "phase": "inventory",
            "parse_error": None,
            "judge_score": 7,
            "judge_rationale": "inventory found",
            "response": _phase_response(
                "inventory",
                agent={
                    "name": "ExampleCo Insights Agent",
                    "profile": "exampleco-insights-agent",
                    "model_aliases": ["exampleco-insights", "exampleco-insights-agent"],
                    "confidence": "confirmed",
                },
                tools=[
                    {
                        "name": "run_readonly_sql",
                        "type": "database",
                        "read_scope": ["SQLite"],
                        "write_scope": [],
                        "evidence": "available tools list",
                        "confidence": "confirmed",
                    },
                    {
                        "name": "describe_schema",
                        "type": "tool",
                        "read_scope": ["SQLite schema"],
                        "write_scope": [],
                        "evidence": "available tools list",
                        "confidence": "confirmed",
                    },
                ],
                limits={
                    "write_access": "read-only SQL only",
                    "network_access": None,
                    "persistence": None,
                    "cross_agent_handoff": None,
                },
            ),
        },
        {
            "phase": "data_layer",
            "parse_error": None,
            "judge_score": 7,
            "judge_rationale": "tables found",
            "response": _phase_response(
                "data_layer",
                data_stores=[
                    {
                        "name": "tickets",
                        "kind": "table",
                        "parent": "SQLite",
                        "fields": ["ticket_id", "user_id", "status"],
                        "join_keys": ["user_id"],
                        "description": "Support tickets.",
                        "confidence": "confirmed",
                    },
                    {
                        "name": "users",
                        "kind": "table",
                        "parent": "SQLite",
                        "fields": ["user_id", "email", "company"],
                        "join_keys": ["user_id"],
                        "description": "User records.",
                        "confidence": "confirmed",
                    },
                ],
            ),
        },
        {
            "phase": "architecture",
            "parse_error": None,
            "judge_score": 7,
            "judge_rationale": "graph found",
            "response": _phase_response(
                "architecture",
                workflow={
                    "framework": "LangGraph",
                    "nodes": [
                        {
                            "name": "retrieve_context",
                            "role": "retrieval",
                            "inputs": ["route"],
                            "outputs": ["context"],
                            "tools_used": ["search_knowledge_base"],
                            "confidence": "confirmed",
                        },
                        {
                            "name": "execute_tools",
                            "role": "tool node",
                            "inputs": ["plan"],
                            "outputs": ["tool_results"],
                            "tools_used": ["describe_schema", "run_readonly_sql"],
                            "confidence": "confirmed",
                        },
                    ],
                    "edges": [
                        {
                            "from": "retrieve_context",
                            "to": "execute_tools",
                            "condition": None,
                            "confidence": "confirmed",
                        }
                    ],
                    "handoff_paths": ["exampleco-insights-agent -> exampleco-support-agent"],
                },
                unknowns=["whether memory is durable"],
            ),
        },
    ]

    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=rounds,
    )

    assert artifact["schema_version"] == ARCHITECTURE_SCHEMA_VERSION
    assert artifact["technique"] == "agent_architecture_enumeration"
    assert artifact["status"] == "mapped"
    assert artifact["quality"] == {
        "round_count": 3,
        "parseable_rounds": 3,
        "raw_response_rounds": 0,
        "mapped_sections": [
            "agents",
            "tools",
            "data_stores",
            "workflow",
            "access_edges",
            "limits",
        ],
    }
    assert artifact["architecture"]["agents"] == [
        {
            "id": "agent:exampleco-insights-agent",
            "name": "ExampleCo Insights Agent",
            "profile": "exampleco-insights-agent",
            "model_aliases": ["exampleco-insights", "exampleco-insights-agent"],
            "confidence": "confirmed",
        }
    ]
    assert [tool["name"] for tool in artifact["architecture"]["tools"]] == [
        "describe_schema",
        "run_readonly_sql",
    ]
    assert [store["name"] for store in artifact["architecture"]["data_stores"]] == [
        "tickets",
        "users",
    ]
    assert artifact["architecture"]["workflow"]["framework"] == "LangGraph"
    assert [node["name"] for node in artifact["architecture"]["workflow"]["nodes"]] == [
        "execute_tools",
        "retrieve_context",
    ]
    assert {
        "source": "workflow_node:execute_tools",
        "target": "tool:describe_schema",
        "access": "uses",
        "evidence": "workflow.tools_used",
    } in artifact["architecture"]["access_edges"]
    assert {
        "source": "agent:exampleco-insights-agent",
        "target": "agent:exampleco-support-agent",
        "access": "handoff",
        "evidence": "exampleco-insights-agent -> exampleco-support-agent",
    } in artifact["architecture"]["access_edges"]
    assert stable_json_dumps(artifact) == stable_json_dumps(
        build_architecture_artifact(
            technique="agent_architecture_enumeration",
            rounds=list(reversed(rounds)),
        )
    )
    assert json.loads(stable_json_dumps(artifact)) == artifact


def test_build_architecture_artifact_returns_empty_shape_for_empty_rounds() -> None:
    rounds = [
        {
            "phase": "inventory",
            "parse_error": None,
            "judge_score": 2,
            "judge_rationale": "no inventory",
            "response": _phase_response("inventory"),
        },
        {
            "phase": "data_layer",
            "parse_error": None,
            "judge_score": 2,
            "judge_rationale": "no data",
            "response": _phase_response("data_layer"),
        },
        {
            "phase": "architecture",
            "parse_error": None,
            "judge_score": 2,
            "judge_rationale": "no workflow",
            "response": _phase_response("architecture"),
        },
    ]

    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=rounds,
    )

    assert artifact["status"] == "empty"
    assert artifact["quality"] == {
        "round_count": 3,
        "parseable_rounds": 3,
        "raw_response_rounds": 0,
        "mapped_sections": [],
    }
    assert artifact["architecture"] == {
        "agents": [],
        "tools": [],
        "data_stores": [],
        "workflow": {
            "framework": None,
            "nodes": [],
            "edges": [],
            "handoff_paths": [],
        },
        "access_edges": [],
        "limits": {
            "write_access": None,
            "network_access": None,
            "persistence": None,
            "cross_agent_handoff": None,
        },
        "unknowns": [],
    }


def test_build_architecture_artifact_merges_agent_identity_across_aliases() -> None:
    rounds = [
        {
            "phase": "inventory",
            "response": _phase_response(
                "inventory",
                agent={
                    "name": "ExampleCo Insights Agent",
                    "profile": None,
                    "model_aliases": ["exampleco-insights-agent"],
                    "confidence": "inferred",
                },
            ),
        },
        {
            "phase": "architecture",
            "response": _phase_response(
                "architecture",
                agent={
                    "name": None,
                    "profile": "exampleco-insights-agent",
                    "model_aliases": [],
                    "confidence": "confirmed",
                },
            ),
        },
    ]

    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=rounds,
    )

    assert len(artifact["architecture"]["agents"]) == 1
    assert artifact["architecture"]["agents"][0] == {
        "id": "agent:exampleco-insights-agent",
        "name": "ExampleCo Insights Agent",
        "profile": "exampleco-insights-agent",
        "model_aliases": ["exampleco-insights-agent"],
        "confidence": "confirmed",
    }


def test_build_architecture_artifact_canonicalizes_case_variant_lists() -> None:
    rounds = [
        {
            "phase": "inventory",
            "response": _phase_response(
                "inventory",
                tools=[
                    {
                        "name": "run_readonly_sql",
                        "type": "database",
                        "read_scope": ["SQLite", "sqlite"],
                        "write_scope": [],
                        "evidence": [],
                        "confidence": "confirmed",
                    }
                ],
            ),
        },
        {
            "phase": "data_layer",
            "response": _phase_response(
                "data_layer",
                data_stores=[
                    {
                        "name": "users",
                        "kind": "table",
                        "parent": "SQLite",
                        "fields": ["Email", "email"],
                        "join_keys": [],
                        "description": None,
                        "confidence": "confirmed",
                    }
                ],
            ),
        },
    ]

    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=rounds,
    )

    assert artifact["architecture"]["tools"][0]["read_scope"] == ["SQLite"]
    assert artifact["architecture"]["data_stores"][0]["fields"] == ["Email"]


def test_build_architecture_artifact_keeps_unknowns_without_mapping_status() -> None:
    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=[
            {
                "phase": "inventory",
                "response": _phase_response(
                    "inventory",
                    unknowns=["nothing visible from this target"],
                ),
            }
        ],
    )

    assert artifact["status"] == "empty"
    assert artifact["quality"]["mapped_sections"] == []
    assert artifact["architecture"]["unknowns"] == ["nothing visible from this target"]


def test_build_architecture_artifact_links_tool_scopes_to_known_data_stores() -> None:
    rounds = [
        {
            "phase": "inventory",
            "response": _phase_response(
                "inventory",
                tools=[
                    {
                        "name": "run_readonly_sql",
                        "type": "database",
                        "read_scope": ["SQLite schema"],
                        "write_scope": [],
                        "evidence": "available tools list",
                        "confidence": "confirmed",
                    }
                ],
            ),
        },
        {
            "phase": "data_layer",
            "response": _phase_response(
                "data_layer",
                data_stores=[
                    {
                        "name": "SQLite",
                        "kind": "database",
                        "parent": None,
                        "fields": [],
                        "join_keys": [],
                        "description": "Local SQLite database.",
                        "confidence": "confirmed",
                    }
                ],
            ),
        },
    ]

    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=rounds,
    )

    assert {
        "source": "tool:run_readonly_sql",
        "target": "data_store:sqlite",
        "access": "read",
        "evidence": "tool.read_scope",
    } in artifact["architecture"]["access_edges"]
    assert {
        "source": "tool:run_readonly_sql",
        "target": "scope:sqlite",
        "access": "read",
        "evidence": "tool.read_scope",
    } not in artifact["architecture"]["access_edges"]
    assert {
        "source": "tool:run_readonly_sql",
        "target": "scope:sqlite_schema",
        "access": "read",
        "evidence": "tool.read_scope",
    } not in artifact["architecture"]["access_edges"]


def test_build_architecture_artifact_resolves_handoff_labels_to_known_agents() -> None:
    rounds = [
        {
            "phase": "inventory",
            "response": _phase_response(
                "inventory",
                agent={
                    "name": "ExampleCo Insights Agent",
                    "profile": "exampleco-insights-agent",
                    "model_aliases": ["exampleco-insights"],
                    "confidence": "confirmed",
                },
            ),
        },
        {
            "phase": "architecture",
            "response": _phase_response(
                "architecture",
                agent={
                    "name": None,
                    "profile": "exampleco-support-agent",
                    "model_aliases": [],
                    "confidence": "confirmed",
                },
            ),
        },
        {
            "phase": "architecture",
            "response": _phase_response(
                "architecture",
                workflow={
                    "framework": None,
                    "nodes": [],
                    "edges": [],
                    "handoff_paths": [
                        "ExampleCo Insights Agent -> ExampleCo Support Agent"
                    ],
                },
            ),
        },
    ]

    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=rounds,
    )

    assert {
        "source": "agent:exampleco-insights-agent",
        "target": "agent:exampleco-support-agent",
        "access": "handoff",
        "evidence": "ExampleCo Insights Agent -> ExampleCo Support Agent",
    } in artifact["architecture"]["access_edges"]
    assert not any(
        edge["access"] == "handoff"
        and edge["source"] == "agent:analytics_support_agent"
        for edge in artifact["architecture"]["access_edges"]
    )
    assert not any(
        edge["access"] == "handoff"
        and edge["target"] == "agent:customer_success_agent"
        for edge in artifact["architecture"]["access_edges"]
    )


def test_build_architecture_artifact_does_not_resolve_handoff_target_from_secondary_alias() -> None:
    artifact = build_architecture_artifact(
        technique="agent_architecture_enumeration",
        rounds=[
            {
                "phase": "architecture",
                "response": _phase_response(
                    "architecture",
                    agent={
                        "name": "ExampleCo Insights Agent",
                        "profile": "exampleco-insights-agent",
                        "model_aliases": ["exampleco-insights-agent", "exampleco-support-agent"],
                        "confidence": "confirmed",
                    },
                    workflow={
                        "framework": None,
                        "nodes": [],
                        "edges": [],
                        "handoff_paths": [
                            "exampleco-insights-agent -> exampleco-support-agent"
                        ],
                    },
                ),
            }
        ],
    )

    assert {
        "source": "agent:exampleco-insights-agent",
        "target": "agent:exampleco-support-agent",
        "access": "handoff",
        "evidence": "exampleco-insights-agent -> exampleco-support-agent",
    } in artifact["architecture"]["access_edges"]
    assert not any(
        edge["access"] == "handoff"
        and edge["source"] == "agent:exampleco-insights-agent"
        and edge["target"] == "agent:exampleco-insights-agent"
        for edge in artifact["architecture"]["access_edges"]
    )
