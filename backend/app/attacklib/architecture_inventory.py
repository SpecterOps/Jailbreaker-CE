from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


ARCHITECTURE_SCHEMA_VERSION = "agent-architecture-map/v1"

_CONFIDENCE_RANK = {"unknown": 0, "inferred": 1, "confirmed": 2}
_LIMIT_KEYS = ("write_access", "network_access", "persistence", "cross_agent_handoff")
_PLACEHOLDER_VALUES = {"", "unknown", "not visible", "none", "null"}
_SCOPE_QUALIFIER_WORDS = {
    "database",
    "datastore",
    "db",
    "schema",
    "schemas",
    "store",
    "table",
    "tables",
}
_ROUND_PHASE_ORDER = {"inventory": 0, "data_layer": 1, "architecture": 2}
_SECTION_ORDER = (
    "agents",
    "tools",
    "data_stores",
    "workflow",
    "access_edges",
    "limits",
    "unknowns",
)


def stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def build_architecture_artifact(
    *, technique: str, rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    architecture = _empty_architecture()
    parseable_rounds = 0
    raw_response_rounds = 0

    for round_item in rounds:
        response = round_item.get("response")
        if isinstance(response, dict):
            parseable_rounds += 1
            _merge_phase_response(architecture, response)
        elif isinstance(round_item.get("raw_response"), str):
            raw_response_rounds += 1

    _sort_architecture(architecture)
    _rebuild_access_edges(architecture)
    mapped_sections = _mapped_sections(architecture)

    return {
        "schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "technique": technique,
        "status": _status_for(
            mapped_sections, parseable_rounds, raw_response_rounds, len(rounds)
        ),
        "quality": {
            "round_count": len(rounds),
            "parseable_rounds": parseable_rounds,
            "raw_response_rounds": raw_response_rounds,
            "mapped_sections": mapped_sections,
        },
        "architecture": architecture,
        "rounds": _canonical_rounds(rounds),
    }


def _empty_architecture() -> dict[str, Any]:
    return {
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


def _merge_phase_response(architecture: dict[str, Any], response: dict[str, Any]) -> None:
    _merge_agent(architecture, response.get("agent"))

    for tool in _as_list(response.get("tools")):
        _merge_tool(architecture, tool)

    for data_store in _as_list(response.get("data_stores")):
        _merge_data_store(architecture, data_store)

    _merge_workflow(architecture, response.get("workflow"))
    _merge_limits(architecture, response.get("limits"))
    architecture["unknowns"] = _merge_list(
        architecture["unknowns"], _clean_list(_as_list(response.get("unknowns")))
    )


def _merge_agent(architecture: dict[str, Any], agent: object) -> None:
    if not isinstance(agent, dict):
        return

    name = _clean_scalar(agent.get("name"))
    profile = _clean_scalar(agent.get("profile"))
    aliases = _clean_list(_as_list(agent.get("model_aliases")))
    if not name and not profile and not aliases:
        return

    incoming = {
        "id": _entity_id("agent", profile or name or aliases[0]),
        "name": name,
        "profile": profile,
        "model_aliases": aliases,
        "confidence": _confidence(agent.get("confidence")),
    }
    matches = _matching_agents(architecture["agents"], incoming)
    if not matches:
        architecture["agents"].append(incoming)
        return

    existing = _select_agent_merge_target(matches, incoming["id"])
    for match in matches:
        if match is existing:
            continue
        _merge_agent_entry(existing, match)
        architecture["agents"].remove(match)
    _merge_agent_entry(existing, incoming)
    if existing["profile"]:
        existing["id"] = _entity_id("agent", existing["profile"])


def _merge_tool(architecture: dict[str, Any], tool: object) -> None:
    if not isinstance(tool, dict):
        return

    name = _clean_scalar(tool.get("name"))
    if not name:
        return

    tool_id = _entity_id("tool", name)
    incoming = {
        "id": tool_id,
        "name": name,
        "type": _clean_scalar(tool.get("type")),
        "read_scope": _clean_list(_as_list(tool.get("read_scope"))),
        "write_scope": _clean_list(_as_list(tool.get("write_scope"))),
        "evidence": _clean_list(_as_list(tool.get("evidence"))),
        "confidence": _confidence(tool.get("confidence")),
    }
    existing = _entry_by_id(architecture["tools"], tool_id)
    if existing is None:
        architecture["tools"].append(incoming)
        return

    existing["type"] = _select_scalar(existing["type"], incoming["type"])
    existing["read_scope"] = _merge_list(existing["read_scope"], incoming["read_scope"])
    existing["write_scope"] = _merge_list(existing["write_scope"], incoming["write_scope"])
    existing["evidence"] = _merge_list(existing["evidence"], incoming["evidence"])
    existing["confidence"] = _best_confidence(
        existing["confidence"], incoming["confidence"]
    )


def _merge_data_store(architecture: dict[str, Any], data_store: object) -> None:
    if not isinstance(data_store, dict):
        return

    name = _clean_scalar(data_store.get("name"))
    if not name:
        return

    store_id = _entity_id("data_store", name)
    incoming = {
        "id": store_id,
        "name": name,
        "kind": _clean_scalar(data_store.get("kind")),
        "parent": _clean_scalar(data_store.get("parent")),
        "fields": _clean_list(_as_list(data_store.get("fields"))),
        "join_keys": _clean_list(_as_list(data_store.get("join_keys"))),
        "description": _clean_scalar(data_store.get("description")),
        "confidence": _confidence(data_store.get("confidence")),
    }
    existing = _entry_by_id(architecture["data_stores"], store_id)
    if existing is None:
        architecture["data_stores"].append(incoming)
        return

    existing["kind"] = _select_scalar(existing["kind"], incoming["kind"])
    existing["parent"] = _select_scalar(existing["parent"], incoming["parent"])
    existing["fields"] = _merge_list(existing["fields"], incoming["fields"])
    existing["join_keys"] = _merge_list(existing["join_keys"], incoming["join_keys"])
    existing["description"] = _select_scalar(
        existing["description"], incoming["description"]
    )
    existing["confidence"] = _best_confidence(
        existing["confidence"], incoming["confidence"]
    )


def _merge_workflow(architecture: dict[str, Any], workflow: object) -> None:
    if not isinstance(workflow, dict):
        return

    target = architecture["workflow"]
    target["framework"] = _select_scalar(
        target["framework"], _clean_scalar(workflow.get("framework"))
    )

    for node in _as_list(workflow.get("nodes")):
        _merge_workflow_node(target, node)

    for edge in _as_list(workflow.get("edges")):
        _merge_workflow_edge(target, edge)

    target["handoff_paths"] = _merge_list(
        target["handoff_paths"], _clean_list(_as_list(workflow.get("handoff_paths")))
    )


def _merge_workflow_node(workflow: dict[str, Any], node: object) -> None:
    if not isinstance(node, dict):
        return

    name = _clean_scalar(node.get("name"))
    if not name:
        return

    node_id = _entity_id("workflow_node", name)
    incoming = {
        "id": node_id,
        "name": name,
        "role": _clean_scalar(node.get("role")),
        "inputs": _clean_list(_as_list(node.get("inputs"))),
        "outputs": _clean_list(_as_list(node.get("outputs"))),
        "tools_used": _clean_list(_as_list(node.get("tools_used"))),
        "confidence": _confidence(node.get("confidence")),
    }
    existing = _entry_by_id(workflow["nodes"], node_id)
    if existing is None:
        workflow["nodes"].append(incoming)
        return

    existing["role"] = _select_scalar(existing["role"], incoming["role"])
    existing["inputs"] = _merge_list(existing["inputs"], incoming["inputs"])
    existing["outputs"] = _merge_list(existing["outputs"], incoming["outputs"])
    existing["tools_used"] = _merge_list(existing["tools_used"], incoming["tools_used"])
    existing["confidence"] = _best_confidence(
        existing["confidence"], incoming["confidence"]
    )


def _merge_workflow_edge(workflow: dict[str, Any], edge: object) -> None:
    if not isinstance(edge, dict):
        return

    source = _clean_scalar(edge.get("from"))
    target = _clean_scalar(edge.get("to"))
    if not source or not target:
        return

    condition = _clean_scalar(edge.get("condition"))
    edge_id = "|".join((source.lower(), target.lower(), (condition or "").lower()))
    existing = _entry_by_id(workflow["edges"], edge_id)
    if existing is None:
        workflow["edges"].append(
            {
                "id": edge_id,
                "from": source,
                "to": target,
                "condition": condition,
                "confidence": _confidence(edge.get("confidence")),
            }
        )
        return

    existing["confidence"] = _best_confidence(
        existing["confidence"], _confidence(edge.get("confidence"))
    )


def _merge_limits(architecture: dict[str, Any], limits: object) -> None:
    if not isinstance(limits, dict):
        return

    for key in _LIMIT_KEYS:
        value = _clean_scalar(limits.get(key))
        if value:
            architecture["limits"][key] = _join_unique_values(
                architecture["limits"][key], value
            )


def _rebuild_access_edges(architecture: dict[str, Any]) -> None:
    edges: list[dict[str, str]] = []

    for tool in architecture["tools"]:
        for scope in tool["read_scope"]:
            _add_access_edge(
                edges,
                tool["id"],
                _scope_target_id(architecture, scope),
                "read",
                "tool.read_scope",
            )
        for scope in tool["write_scope"]:
            _add_access_edge(
                edges,
                tool["id"],
                _scope_target_id(architecture, scope),
                "write",
                "tool.write_scope",
            )

    for node in architecture["workflow"]["nodes"]:
        for tool_name in node["tools_used"]:
            _add_access_edge(
                edges,
                node["id"],
                _entity_id("tool", tool_name),
                "uses",
                "workflow.tools_used",
            )

    for path in architecture["workflow"]["handoff_paths"]:
        parts = [_clean_scalar(part) for part in path.split("->")]
        agents = [part for part in parts if part]
        for source, target in zip(agents, agents[1:]):
            _add_access_edge(
                edges,
                _agent_target_id(architecture, source),
                _agent_target_id(architecture, target),
                "handoff",
                path,
            )

    architecture["access_edges"] = sorted(
        edges,
        key=lambda edge: (
            edge["source"],
            edge["target"],
            edge["access"],
            edge["evidence"],
        ),
    )


def _add_access_edge(
    edges: list[dict[str, str]], source: str, target: str, access: str, evidence: str
) -> None:
    edge = {"source": source, "target": target, "access": access, "evidence": evidence}
    if edge not in edges:
        edges.append(edge)


def _scope_target_id(architecture: dict[str, Any], scope: str) -> str:
    exact_match: str | None = None
    qualified_match: str | None = None
    for store in architecture["data_stores"]:
        store_name = _clean_scalar(store.get("name"))
        store_id = _clean_scalar(store.get("id"))
        labels = [store_name]
        if store_id and ":" in store_id:
            labels.append(store_id.split(":", 1)[1])

        for label in labels:
            if not label:
                continue
            if _identity_token(label) == _identity_token(scope):
                exact_match = store["id"]
                break
            if _scope_matches_store(scope, label):
                qualified_match = qualified_match or store["id"]
        if exact_match:
            return exact_match
    if qualified_match:
        return qualified_match
    return _entity_id("scope", scope)


def _agent_target_id(architecture: dict[str, Any], label: str) -> str:
    label_token = _identity_token(label)
    for agent in architecture["agents"]:
        if label_token in _agent_primary_identity_tokens(agent):
            return agent["id"]
    return _entity_id("agent", label)


def _sort_architecture(architecture: dict[str, Any]) -> None:
    architecture["agents"].sort(key=lambda item: item["id"])
    architecture["tools"].sort(key=lambda item: item["id"])
    architecture["data_stores"].sort(key=lambda item: item["id"])
    architecture["workflow"]["nodes"].sort(key=lambda item: item["id"])
    architecture["workflow"]["edges"].sort(key=lambda item: item["id"])
    architecture["workflow"]["handoff_paths"] = sorted(
        architecture["workflow"]["handoff_paths"], key=_casefold_sort_key
    )
    architecture["unknowns"] = sorted(architecture["unknowns"], key=_casefold_sort_key)


def _mapped_sections(architecture: dict[str, Any]) -> list[str]:
    mapped: list[str] = []
    for section in _SECTION_ORDER:
        if section == "unknowns":
            continue
        if section == "workflow":
            workflow = architecture["workflow"]
            if (
                workflow["framework"]
                or workflow["nodes"]
                or workflow["edges"]
                or workflow["handoff_paths"]
            ):
                mapped.append(section)
        elif section == "limits":
            if any(architecture["limits"].values()):
                mapped.append(section)
        elif architecture[section]:
            mapped.append(section)
    return mapped


def _status_for(
    mapped_sections: list[str],
    parseable_rounds: int,
    raw_response_rounds: int,
    round_count: int,
) -> str:
    if not mapped_sections:
        return "empty"
    if raw_response_rounds or parseable_rounds < round_count:
        return "partial"
    return "mapped"


def _canonical_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = deepcopy(rounds)
    return sorted(
        copied,
        key=lambda round_item: (
            _ROUND_PHASE_ORDER.get(str(round_item.get("phase")), len(_ROUND_PHASE_ORDER)),
            stable_json_dumps(round_item),
        ),
    )


def _matching_agents(
    agents: list[dict[str, Any]], incoming: dict[str, Any]
) -> list[dict[str, Any]]:
    incoming_tokens = _agent_identity_tokens(incoming)
    return [
        agent
        for agent in agents
        if incoming_tokens.intersection(_agent_identity_tokens(agent))
    ]


def _select_agent_merge_target(
    matches: list[dict[str, Any]], incoming_id: str
) -> dict[str, Any]:
    for match in matches:
        if match["id"] == incoming_id:
            return match
    return sorted(matches, key=lambda match: match["id"])[0]


def _merge_agent_entry(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    target["name"] = _select_scalar(target["name"], incoming["name"])
    target["profile"] = _select_scalar(target["profile"], incoming["profile"])
    target["model_aliases"] = _merge_list(
        target["model_aliases"], incoming["model_aliases"]
    )
    target["confidence"] = _best_confidence(
        target["confidence"], incoming["confidence"]
    )


def _agent_identity_tokens(agent: dict[str, Any]) -> set[str]:
    tokens = {
        _identity_token(value)
        for value in [
            _clean_scalar(agent.get("name")),
            _clean_scalar(agent.get("profile")),
            *_clean_list(_as_list(agent.get("model_aliases"))),
        ]
        if value
    }
    agent_id = _clean_scalar(agent.get("id"))
    if agent_id and ":" in agent_id:
        tokens.add(_identity_token(agent_id.split(":", 1)[1]))
    return tokens


def _agent_primary_identity_tokens(agent: dict[str, Any]) -> set[str]:
    tokens = {
        _identity_token(value)
        for value in [
            _clean_scalar(agent.get("name")),
            _clean_scalar(agent.get("profile")),
        ]
        if value
    }
    agent_id = _clean_scalar(agent.get("id"))
    if agent_id and ":" in agent_id:
        tokens.add(_identity_token(agent_id.split(":", 1)[1]))
    return tokens


def _entry_by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def _entity_id(kind: str, name: str) -> str:
    return f"{kind}:{_slug(name)}"


def _slug(value: str) -> str:
    return " ".join(value.strip().split()).replace(" ", "_").lower()


def _identity_token(value: str) -> str:
    return "_".join(_separator_parts(value))


def _separator_parts(value: str) -> list[str]:
    cleaned = _clean_scalar(value)
    if not cleaned:
        return []
    return [part for part in re.split(r"[\s_-]+", cleaned.casefold()) if part]


def _scope_matches_store(scope: str, store_label: str) -> bool:
    scope_parts = _separator_parts(scope)
    store_parts = _separator_parts(store_label)
    if not scope_parts or not store_parts:
        return False
    if scope_parts == store_parts:
        return True
    if scope_parts[: len(store_parts)] == store_parts:
        return all(
            part in _SCOPE_QUALIFIER_WORDS for part in scope_parts[len(store_parts) :]
        )
    if scope_parts[-len(store_parts) :] == store_parts:
        return all(
            part in _SCOPE_QUALIFIER_WORDS for part in scope_parts[: -len(store_parts)]
        )
    return False


def _clean_scalar(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if text.lower() in _PLACEHOLDER_VALUES:
        return None
    return text


def _clean_list(values: list[object]) -> list[str]:
    return _canonical_list(values)


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_list(existing: list[str], incoming: list[str]) -> list[str]:
    return _canonical_list([*existing, *incoming])


def _confidence(value: object) -> str:
    cleaned = _clean_scalar(value)
    if cleaned in _CONFIDENCE_RANK:
        return cleaned
    return "unknown"


def _best_confidence(left: str, right: str) -> str:
    if _CONFIDENCE_RANK[left] >= _CONFIDENCE_RANK[right]:
        return left
    return right


def _join_unique_values(current: str | None, candidate: str) -> str:
    values = [candidate]
    if current:
        values.extend(item.strip() for item in current.split(" | ") if item.strip())
    return " | ".join(_canonical_list(values))


def _select_scalar(left: str | None, right: str | None) -> str | None:
    if left and right:
        return min(left, right, key=_casefold_sort_key)
    return left or right


def _canonical_list(values: list[object]) -> list[str]:
    representatives: dict[str, str] = {}
    for value in values:
        cleaned = _clean_scalar(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        current = representatives.get(key)
        if current is None or _casefold_sort_key(cleaned) < _casefold_sort_key(current):
            representatives[key] = cleaned
    return [
        representative
        for _, representative in sorted(
            representatives.items(), key=lambda item: (item[0], item[1])
        )
    ]


def _casefold_sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)
