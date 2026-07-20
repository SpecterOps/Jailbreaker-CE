from __future__ import annotations

from app.attacklib.architecture_inventory import ARCHITECTURE_SCHEMA_VERSION, stable_json_dumps

__all__ = ["demo_architecture_enumeration_output", "extract_conversation"]


# ---------------------------------------------------------------------------
# Canonical conversation schema
# ---------------------------------------------------------------------------
#
# All techniques produce a list of conversation entries with one of three
# `kind` values. Each kind has a fixed, predictable shape so the frontend
# can render it with a base renderer + per-kind specialization.
#
#   kind="round"         - standard iterative format. Crescendo, PAIR, TAP,
#                          erosion. One entry per turn.
#       {
#         "kind": "round",
#         "round_number": "1",
#         "is_backtrack": False,        # crescendo-specific marker
#         "attacker": {"input": str, "output": str},
#         "target":   {"input": str, "prompt": str, "response": str},
#         "judge":    {"input": str, "output": str, "score": "5",
#                      "rationale": str}
#       }
#
#   kind="generation"    - population-based attacks (AutoDAN, mutation).
#                          One entry per generation, contains many candidates.
#       {
#         "kind": "generation",
#         "generation_number": "1",
#         "population_size": "5",
#         "best_score": "8",
#         "succeeded": False,
#         "candidates": [ { ... candidate fields ... } ]
#       }
#
#   kind="single_shot"   - non-iterative single prompt/response.
#       {
#         "kind": "single_shot",
#         "prompt": str,
#         "response": str,
#         "judge": {"input": str, "output": str, "score": "5",
#                   "rationale": str}      # may be empty
#       }
# ---------------------------------------------------------------------------

# Techniques that use population-based search (multiple candidates per round).

_POPULATION_TECHNIQUES = {"autodan", "mutation"}


def extract_conversation(result) -> list[dict]:
    """Extract a canonical conversation log from a TechniqueRunResult.

    Returns a list of entries with one of three `kind` values:
    "round", "generation", or "single_shot". See the schema docstring above.
    """
    iterative_rounds = [
        t for t in result.traces if t.event_type == "iterative_round"
    ]

    if not iterative_rounds:
        return _extract_single_shot(result)

    if result.technique_key in _POPULATION_TECHNIQUES:
        return _extract_generations(iterative_rounds)

    entries = _extract_rounds(iterative_rounds)
    if (
        result.technique_key == "agent_architecture_enumeration"
        and result.response.final_output
    ):
        entries.append(
            {
                "kind": "artifact",
                "label": "Canonical architecture artifact",
                "response": result.response.final_output,
            }
        )
    return entries


def demo_architecture_enumeration_output() -> str:
    return stable_json_dumps(
        {
            "schema_version": ARCHITECTURE_SCHEMA_VERSION,
            "technique": "agent_architecture_enumeration",
            "status": "mapped",
            "quality": {
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
            },
            "architecture": {
                "agents": [
                    {
                        "id": "agent:exampleco-insights-agent",
                        "name": "ExampleCo Insights Agent",
                        "profile": "exampleco-insights-agent",
                        "model_aliases": ["exampleco-insights-agent"],
                        "confidence": "confirmed",
                    }
                ],
                "tools": [
                    {
                        "id": "tool:describe_schema",
                        "name": "describe_schema",
                        "type": "tool",
                        "read_scope": ["SQLite schema"],
                        "write_scope": [],
                        "evidence": ["demo target"],
                        "confidence": "confirmed",
                    },
                    {
                        "id": "tool:run_readonly_sql",
                        "name": "run_readonly_sql",
                        "type": "database",
                        "read_scope": ["SQLite"],
                        "write_scope": [],
                        "evidence": ["demo target"],
                        "confidence": "confirmed",
                    },
                ],
                "data_stores": [
                    {
                        "id": "data_store:sqlite",
                        "name": "SQLite",
                        "kind": "database",
                        "parent": None,
                        "fields": [],
                        "join_keys": [],
                        "description": "Demo analytics database.",
                        "confidence": "confirmed",
                    }
                ],
                "workflow": {
                    "framework": "LangGraph",
                    "nodes": [
                        {
                            "id": "workflow_node:execute_tools",
                            "name": "execute_tools",
                            "role": "tool node",
                            "inputs": ["plan"],
                            "outputs": ["tool_results"],
                            "tools_used": ["describe_schema", "run_readonly_sql"],
                            "confidence": "confirmed",
                        }
                    ],
                    "edges": [],
                    "handoff_paths": ["exampleco-insights-agent -> exampleco-support-agent"],
                },
                "access_edges": [
                    {
                        "source": "workflow_node:execute_tools",
                        "target": "tool:describe_schema",
                        "access": "uses",
                        "evidence": "workflow.tools_used",
                    },
                    {
                        "source": "agent:exampleco-insights-agent",
                        "target": "agent:exampleco-support-agent",
                        "access": "handoff",
                        "evidence": "exampleco-insights-agent -> exampleco-support-agent",
                    },
                ],
                "limits": {
                    "write_access": "read-only SQL only",
                    "network_access": None,
                    "persistence": None,
                    "cross_agent_handoff": "exampleco-insights-agent can hand off to exampleco-support-agent",
                },
                "unknowns": [],
            },
            "rounds": [],
        }
    )


def _extract_rounds(iterative_rounds: list) -> list[dict]:
    """Build kind=round entries - one per iteration. Used by Crescendo,
    PAIR, TAP, erosion, and any other turn-by-turn technique."""
    entries: list[dict] = []
    seen_rounds: dict[str, int] = {}

    for trace in iterative_rounds:
        attrs = trace.attributes
        round_num = attrs.get("round", "?")
        rationale = attrs.get("judge_rationale", "")

        # Detect Crescendo backtracks via the rationale prefix the loop
        # writes when it pops a refused exchange.
        is_backtrack = rationale.startswith("[BACKTRACK")

        # If we've seen this round number before AND this isn't a backtrack
        # marker, the prior occurrence was likely a backtrack retry.
        seen_rounds[round_num] = seen_rounds.get(round_num, 0) + 1

        entries.append({
            "kind": "round",
            "round_number": round_num,
            "is_backtrack": is_backtrack,
            "attacker": {
                "input": attrs.get("attacker_input", ""),
                "output": attrs.get("attacker_output", ""),
            },
            "target": {
                "input": attrs.get("target_input", ""),
                "prompt": attrs.get("candidate_prompt", ""),
                "response": attrs.get("target_response", ""),
                # Crescendo-Liar: non-empty when the response was rewritten
                # by the fabricator before being stored in the conversation
                # history. The frontend uses this to render a POISONED badge.
                "poisoned": attrs.get("poisoned_response", ""),
            },
            "judge": {
                "input": attrs.get("judge_input", ""),
                "output": attrs.get("judge_output", ""),
                "score": attrs.get("judge_score", "?"),
                "rationale": rationale,
            },
        })
    return entries


def _extract_generations(iterative_rounds: list) -> list[dict]:
    """Build kind=generation entries - one per generation, each containing
    multiple candidates. Used by AutoDAN and other population-based techniques."""
    import re as _re
    from collections import OrderedDict

    generations: dict[str, list] = OrderedDict()
    for trace in iterative_rounds:
        rn = trace.attributes.get("round", "?")
        generations.setdefault(rn, []).append(trace.attributes)

    entries: list[dict] = []
    for gen_num, candidates in generations.items():
        # Sort candidates within the generation by score descending.
        candidates.sort(
            key=lambda c: int(c.get("judge_score", "0")),
            reverse=True,
        )
        best_score = int(candidates[0].get("judge_score", "0"))

        candidate_list = []
        for i, attrs in enumerate(candidates):
            # Parse evolutionary metadata embedded in the attacker_input
            # (e.g. "[gen=2 pos=1/3 origin=elite_from_gen1]").
            meta = attrs.get("attacker_input", "")
            origin = "unknown"
            position = ""
            m = _re.search(r"origin=([^\s\]]+)", meta)
            if m:
                origin = m.group(1)
            m = _re.search(r"pos=(\d+/\d+)", meta)
            if m:
                position = m.group(1)

            candidate_list.append({
                "index": i + 1,
                "is_best": i == 0,
                "origin": origin,
                "position": position,
                "prompt": attrs.get("candidate_prompt", ""),
                "response": attrs.get("target_response", ""),
                "target_input": attrs.get("target_input", ""),
                "judge_input": attrs.get("judge_input", ""),
                "judge_output": attrs.get("judge_output", ""),
                "judge_score": attrs.get("judge_score", "0"),
                "judge_rationale": attrs.get("judge_rationale", ""),
            })

        entries.append({
            "kind": "generation",
            "generation_number": gen_num,
            "population_size": str(len(candidates)),
            "best_score": str(best_score),
            "succeeded": any(
                int(c.get("judge_score", "0")) >= 7 for c in candidates
            ),
            "candidates": candidate_list,
        })
    return entries


def _extract_single_shot(result) -> list[dict]:
    """Build a single kind=single_shot entry for non-iterative techniques."""
    prompt_text = ""
    full_input = ""
    for trace in result.traces:
        if trace.event_type == "prompt_sent":
            prompt_text = trace.attributes.get("rendered_prompt", "")
        # The live adapter captures the full message list (system + user)
        # in the prompt_rendered trace - use it for the raw input view.
        if trace.event_type == "prompt_rendered":
            full_input = trace.attributes.get("full_input", "")

    if not (prompt_text or result.response.final_output):
        return []

    judge = {"input": "", "output": "", "score": "?", "rationale": "", "error": ""}
    for trace in result.traces:
        if trace.event_type == "judge_evaluation":
            attrs = trace.attributes
            judge = {
                "input": attrs.get("judge_input", ""),
                "output": attrs.get("judge_output", ""),
                "score": attrs.get("judge_score", "?"),
                "rationale": attrs.get("judge_rationale", ""),
                "error": attrs.get("judge_error", ""),
            }
            break

    entry = {
        "kind": "single_shot",
        "prompt": prompt_text,
        "response": result.response.final_output,
        "judge": judge,
    }
    # Include full_input (with system prompt) when available - the
    # frontend's "Show raw target input" toggle renders this.
    if full_input:
        entry["full_input"] = full_input
    return [entry]
