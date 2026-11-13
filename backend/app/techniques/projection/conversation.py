"""Generic conversation views selected by declared presentation format."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Mapping

from app.techniques.contracts.events import ExecutionEvent
from app.techniques.projection.outcome import deduplicate_events

if TYPE_CHECKING:
    from app.techniques.core.histories import HistorySnapshot


_SUPPORTED_FORMATS = {
    "single_shot",
    "rounds",
    "generation",
    "tree",
    "diff",
    "judge",
    "artifact",
    "table",
    "raw",
}


class ConversationProjector:
    def project(
        self,
        events: tuple[ExecutionEvent, ...],
        *,
        presentation: Mapping[str, object],
    ) -> dict[str, object]:
        normalized = deduplicate_events(events)
        requested_format = presentation.get(
            "conversation_format",
            presentation.get("format", "raw"),
        )
        format_name = (
            requested_format
            if isinstance(requested_format, str)
            and requested_format in _SUPPORTED_FORMATS
            else "raw"
        )
        if format_name == "rounds":
            sections = self._round_sections(normalized)
        elif format_name == "single_shot":
            sections = [self._event_section(event) for event in normalized]
        elif format_name == "generation":
            sections = self._generation_sections(normalized)
        elif format_name == "tree":
            sections = self._tree_sections(normalized)
        elif format_name == "diff":
            sections = self._filtered_sections(normalized, {"value.transformed"})
        elif format_name == "judge":
            sections = self._filtered_sections(
                normalized,
                {"judge.scheduled", "judge.running", "judge.completed", "judge.failed", "judge.skipped"},
            )
        elif format_name == "artifact":
            sections = self._filtered_sections(normalized, {"artifact.created"})
        elif format_name == "table":
            sections = [{"rows": [self._event_section(event) for event in normalized]}]
        else:
            sections = [self._event_section(event) for event in normalized]
        final_artifact = presentation.get("final_artifact")
        projection: dict[str, object] = {
            "schema_version": 1,
            "format": format_name,
            "requested_format": requested_format,
            "sections": sections,
            "projected_event_count": len(normalized),
        }
        if isinstance(final_artifact, Mapping):
            projection["final_artifact"] = dict(final_artifact)
        return projection

    def _round_sections(
        self,
        events: tuple[ExecutionEvent, ...],
    ) -> list[dict[str, object]]:
        round_numbers = sorted(
            {event.round for event in events if event.round is not None}
        )
        sections: list[dict[str, object]] = []
        for round_number in round_numbers:
            round_events = [event for event in events if event.round == round_number]
            completed_indexes = [
                index
                for index, event in enumerate(round_events)
                if event.event_type == "round.completed"
            ]
            if completed_indexes:
                attempt_starts = [0, *(index + 1 for index in completed_indexes[:-1])]
                attempt_ends = [
                    *(index + 1 for index in completed_indexes[:-1]),
                    len(round_events),
                ]
                attempts = [
                    round_events[start:end]
                    for start, end in zip(attempt_starts, attempt_ends, strict=True)
                ]
            else:
                attempts = [round_events]
            for attempt_events in attempts:
                responses = [
                    event.payload.get("response")
                    for event in attempt_events
                    if event.event_type == "target.completed"
                ]
                judges: dict[str, dict[str, object]] = {}
                for event in attempt_events:
                    if event.event_type not in {"judge.completed", "judge.failed"}:
                        continue
                    definition_id = event.payload.get("definition_id")
                    if not isinstance(definition_id, str):
                        continue
                    judges[definition_id] = {
                        "status": event.payload.get(
                            "status",
                            (
                                "failed"
                                if event.event_type == "judge.failed"
                                else "succeeded"
                            ),
                        ),
                        "score": (
                            event.payload.get("score")
                            if event.event_type == "judge.completed"
                            else None
                        ),
                        "result_id": event.payload.get("result_id"),
                    }
                sections.append(
                    {
                        "round_number": round_number,
                        "responses": responses,
                        "judges": judges,
                        "events": [
                            self._event_section(event) for event in attempt_events
                        ],
                    }
                )
        return sections

    def _generation_sections(
        self,
        events: tuple[ExecutionEvent, ...],
    ) -> list[dict[str, object]]:
        generation_numbers = sorted(
            {
                number
                for event in events
                if isinstance((number := event.payload.get("generation")), int)
                and not isinstance(number, bool)
            }
        )
        return [
            {
                "generation_number": number,
                "events": [
                    self._event_section(event)
                    for event in events
                    if event.payload.get("generation") == number
                ],
            }
            for number in generation_numbers
        ]

    def _tree_sections(
        self,
        events: tuple[ExecutionEvent, ...],
    ) -> list[dict[str, object]]:
        branch_ids = sorted(
            {event.branch_id for event in events if event.branch_id is not None}
        )
        return [
            {
                "branch_id": branch_id,
                "events": [
                    self._event_section(event)
                    for event in events
                    if event.branch_id == branch_id
                ],
            }
            for branch_id in branch_ids
        ]

    def _filtered_sections(
        self,
        events: tuple[ExecutionEvent, ...],
        event_types: set[str],
    ) -> list[dict[str, object]]:
        return [
            self._event_section(event)
            for event in events
            if event.event_type in event_types
        ]

    @staticmethod
    def _event_section(event: ExecutionEvent) -> dict[str, object]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "sequence": event.sequence,
            "round": event.round,
            "branch_id": event.branch_id,
            "payload": dict(event.payload),
        }


class ApplicationConversationProjector:
    """Translate one manifest-selected projection into the public schema.

    The normalized event journal remains the evidence source, while the
    captured projection decides the presentation shape.  This keeps package
    identity out of the application adapter and preserves event references in
    the raw fallback.
    """

    def project(
        self,
        projection: Mapping[str, object],
        *,
        events: tuple[ExecutionEvent, ...],
        history_snapshots: Mapping[str, HistorySnapshot],
        final_target_event: ExecutionEvent | None,
        official_judge_event: ExecutionEvent | None,
        attack_succeeded: bool,
    ) -> list[dict[str, object]]:
        normalized = deduplicate_events(events)
        event_by_id = {event.event_id: event for event in normalized}
        format_name = projection.get("format")
        sections = projection.get("sections")
        if not isinstance(format_name, str):
            format_name = "raw"
        if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes, bytearray)):
            sections = ()

        if format_name == "single_shot":
            entries = [
                self._single_shot(
                    normalized,
                    history_snapshots=history_snapshots,
                    target_event=final_target_event,
                    judge_event=official_judge_event,
                )
            ]
        elif format_name == "rounds":
            entries = self._rounds(
                sections,
                events=normalized,
                event_by_id=event_by_id,
                history_snapshots=history_snapshots,
                final_target_event=final_target_event,
                official_judge_event=official_judge_event,
            )
        elif format_name == "generation":
            entries = self._generations(
                sections,
                events=normalized,
                event_by_id=event_by_id,
                history_snapshots=history_snapshots,
                final_target_event=final_target_event,
                official_judge_event=official_judge_event,
                attack_succeeded=attack_succeeded,
            )
        elif format_name == "tree":
            entries = self._tree(
                sections,
                events=normalized,
                event_by_id=event_by_id,
                history_snapshots=history_snapshots,
                final_target_event=final_target_event,
                official_judge_event=official_judge_event,
            )
        else:
            entries = [
                {
                    "kind": "raw",
                    "format": format_name,
                    "sections": [_json_mapping(item) for item in sections],
                    "events": [
                        ConversationProjector._event_section(event)
                        for event in normalized
                    ],
                }
            ]
        return self._append_final_artifact(
            entries,
            config=_json_mapping(projection.get("final_artifact")),
            final_response_event=final_target_event,
            official_judge_event=official_judge_event,
        )

    @staticmethod
    def _append_final_artifact(
        entries: list[dict[str, object]],
        *,
        config: Mapping[str, object],
        final_response_event: ExecutionEvent | None,
        official_judge_event: ExecutionEvent | None,
    ) -> list[dict[str, object]]:
        if config.get("source") != "final_response" or final_response_event is None:
            return entries
        response = final_response_event.payload.get("response")
        if not isinstance(response, str):
            return entries
        media_type = config.get("media_type", "text/plain")
        if "media_type" in final_response_event.payload:
            response_media_type = final_response_event.payload["media_type"]
            if (
                not isinstance(response_media_type, str)
                or not response_media_type.strip()
                or response_media_type != media_type
            ):
                return entries

        artifact: dict[str, object] = {
            "kind": "artifact",
            "label": config.get("label", "Final artifact"),
            "media_type": media_type,
            "response": response,
        }
        if config.get("include_official_judge") is True:
            artifact["judge"] = _judge(official_judge_event)
        entries.append(artifact)
        return entries

    def _single_shot(
        self,
        events: tuple[ExecutionEvent, ...],
        *,
        history_snapshots: Mapping[str, HistorySnapshot],
        target_event: ExecutionEvent | None,
        judge_event: ExecutionEvent | None,
    ) -> dict[str, object]:
        return {
            "kind": "single_shot",
            "prompt": _prompt_for_target(
                target_event,
                events=events,
                history_snapshots=history_snapshots,
            ),
            "response": _target_response(target_event),
            "judge": _judge(judge_event),
        }

    def _rounds(
        self,
        sections: Sequence[object],
        *,
        events: tuple[ExecutionEvent, ...],
        event_by_id: Mapping[str, ExecutionEvent],
        history_snapshots: Mapping[str, HistorySnapshot],
        final_target_event: ExecutionEvent | None,
        official_judge_event: ExecutionEvent | None,
    ) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for section in sections:
            raw = _json_mapping(section)
            section_events = _section_events(raw, event_by_id)
            targets = [event for event in section_events if event.event_type == "target.completed"]
            target = targets[-1] if targets else None
            judges = [
                event
                for event in section_events
                if event.event_type in {"judge.completed", "judge.failed"}
                and (target is None or event.payload.get("target_response_event_id") in {None, target.event_id})
            ]
            judge = judges[-1] if judges else None
            if (
                judge is None
                and _judge_is_for_target(official_judge_event, target)
            ):
                judge = official_judge_event
            round_events = [
                event
                for event in section_events
                if event.event_type == "round.completed"
            ]
            round_event = round_events[-1] if round_events else None
            prompt = _prompt_for_target(
                target,
                events=section_events,
                history_snapshots=history_snapshots,
            )
            evidence = _round_message_evidence(
                round_event,
                target=target,
                events=events,
                history_snapshots=history_snapshots,
            )
            entries.append(
                {
                    "kind": "round",
                    "round_number": str(raw.get("round_number", "?")),
                    "is_backtrack": bool(
                        round_event is not None
                        and round_event.payload.get("backtracked") is True
                    ),
                    "attacker": {
                        "input": evidence["attacker_input"],
                        "output": evidence["attacker_output"],
                    },
                    "target": {
                        "input": "",
                        "prompt": evidence["target_prompt"] or prompt,
                        "response": _target_response(target),
                        "poisoned": evidence["transformed"],
                    },
                    "judge": _judge(judge),
                }
            )
        if entries:
            return entries
        return [
            {
                "kind": "round",
                "round_number": str(final_target_event.round or 1) if final_target_event is not None else "1",
                "is_backtrack": False,
                "attacker": {"input": "", "output": ""},
                "target": {
                    "input": "",
                    "prompt": _prompt_for_target(
                        final_target_event,
                        events=tuple(event_by_id.values()),
                        history_snapshots=history_snapshots,
                    ),
                    "response": _target_response(final_target_event),
                    "poisoned": "",
                },
                "judge": _judge(
                    official_judge_event
                    if _judge_is_for_target(
                        official_judge_event,
                        final_target_event,
                    )
                    else None
                ),
            }
        ]

    def _generations(
        self,
        sections: Sequence[object],
        *,
        events: tuple[ExecutionEvent, ...],
        event_by_id: Mapping[str, ExecutionEvent],
        history_snapshots: Mapping[str, HistorySnapshot],
        final_target_event: ExecutionEvent | None,
        official_judge_event: ExecutionEvent | None,
        attack_succeeded: bool,
    ) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        selected_ids = _selected_candidate_ids(events)
        for section in sections:
            raw = _json_mapping(section)
            section_events = _section_events(raw, event_by_id)
            candidates = self._candidates(
                section_events,
                evidence_events=events,
                history_snapshots=history_snapshots,
                selected_ids=selected_ids,
            )
            if not candidates:
                continue
            numeric_scores = [item["judge_score"] for item in candidates if _is_number(item.get("judge_score"))]
            best_score = max(numeric_scores) if numeric_scores else None
            for item in candidates:
                item["is_best"] = (
                    item.get("candidate_id") in selected_ids
                    if best_score is None
                    else item.get("judge_score") == best_score
                )
            entries.append(
                {
                    "kind": "generation",
                    "generation_number": str(raw.get("generation_number", "?")),
                    "population_size": str(len(candidates)),
                    "best_score": "" if best_score is None else str(best_score),
                    "succeeded": attack_succeeded,
                    "candidates": candidates,
                }
            )
        if entries:
            return entries

        # A package may intentionally present a one-shot workflow as a
        # generation.  Preserve that declaration instead of silently changing
        # the public kind.
        candidate_id = _candidate_id_for_target(final_target_event) or "candidate-1"
        score = _judge(official_judge_event).get("score")
        return [
            {
                "kind": "generation",
                "generation_number": "1",
                "population_size": "1",
                "best_score": "" if score is None else str(score),
                "succeeded": attack_succeeded,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "index": 1,
                        "position": "1",
                        "origin": "single",
                        "parent_ids": [],
                        "prompt": _prompt_for_target(
                            final_target_event,
                            events=events,
                            history_snapshots=history_snapshots,
                        ),
                        "response": _target_response(final_target_event),
                        "judge_status": _judge(official_judge_event)["status"],
                        "judge_score": score,
                        "judge_rationale": _judge(official_judge_event)["rationale"],
                        "is_best": True,
                        "is_selected": True,
                    }
                ],
            }
        ]

    def _tree(
        self,
        sections: Sequence[object],
        *,
        events: tuple[ExecutionEvent, ...],
        event_by_id: Mapping[str, ExecutionEvent],
        history_snapshots: Mapping[str, HistorySnapshot],
        final_target_event: ExecutionEvent | None,
        official_judge_event: ExecutionEvent | None,
    ) -> list[dict[str, object]]:
        del sections, event_by_id
        selected_ids = _selected_candidate_ids(events)
        completed = _candidate_evidence(events)
        if not completed and final_target_event is not None:
            candidate_id = _candidate_id_for_target(final_target_event) or "branch-1"
            return [
                {
                    "kind": "tree",
                    "depth": str(final_target_event.round or 1),
                    "branches": [
                        {
                            "branch_id": candidate_id,
                            "candidate_id": candidate_id,
                            "depth": final_target_event.round or 1,
                            "parent_id": None,
                            "origin": "single",
                            "prompt": _prompt_for_target(
                                final_target_event,
                                events=events,
                                history_snapshots=history_snapshots,
                            ),
                            "response": _target_response(final_target_event),
                            "judge": _judge(official_judge_event),
                            "status": "selected",
                            "is_retained": True,
                            "is_pruned": False,
                            "is_selected": True,
                        }
                    ],
                }
            ]
        statuses = _candidate_statuses(events)
        by_depth: dict[int, list[dict[str, object]]] = {}
        for index, evidence in enumerate(completed, start=1):
            payload = evidence.payload
            candidate_id = str(payload.get("candidate_id") or evidence.branch_id or evidence.event_id)
            depth = _integer(payload.get("depth"), evidence.round or 1)
            target = _target_for_candidate(evidence, events)
            judge_event = _judge_for_candidate(candidate_id, target, events)
            state = statuses.get(candidate_id, "completed")
            by_depth.setdefault(depth, []).append(
                {
                    "branch_id": str(evidence.branch_id or candidate_id),
                    "candidate_id": candidate_id,
                    "index": index,
                    "depth": depth,
                    "parent_id": payload.get("parent_id"),
                    "origin": str(payload.get("origin") or "unknown"),
                    "prompt": str(
                        payload.get("prompt")
                        or _prompt_for_target(
                            target,
                            events=events,
                            history_snapshots=history_snapshots,
                        )
                    ),
                    "response": _target_response(target),
                    "judge": _judge(judge_event),
                    "status": "selected" if candidate_id in selected_ids else state,
                    "is_retained": state == "retained" or candidate_id in selected_ids,
                    "is_pruned": state == "pruned",
                    "is_selected": candidate_id in selected_ids,
                }
            )
        return [{"kind": "tree", "depth": str(depth), "branches": by_depth[depth]} for depth in sorted(by_depth)]

    def _candidates(
        self,
        candidate_events: tuple[ExecutionEvent, ...],
        *,
        evidence_events: tuple[ExecutionEvent, ...],
        history_snapshots: Mapping[str, HistorySnapshot],
        selected_ids: set[str],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for index, evidence in enumerate(_candidate_evidence(candidate_events), start=1):
            payload = evidence.payload
            candidate_id = str(payload.get("candidate_id") or evidence.branch_id or evidence.event_id)
            target = _target_for_candidate(evidence, evidence_events)
            judge_event = _judge_for_candidate(candidate_id, target, evidence_events)
            judge = _judge(judge_event)
            parents = payload.get("parent_ids", ())
            if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes, bytearray)):
                parents = ()
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "index": index,
                    "position": str(payload.get("position") or index),
                    "origin": str(payload.get("origin") or "unknown"),
                    "parent_ids": [str(item) for item in parents],
                    "prompt": str(
                        payload.get("prompt")
                        or _prompt_for_target(
                            target,
                            events=evidence_events,
                            history_snapshots=history_snapshots,
                        )
                    ),
                    "response": _target_response(target),
                    "judge_status": judge["status"],
                    "judge_score": judge["score"],
                    "judge_rationale": judge["rationale"],
                    "is_best": False,
                    "is_selected": candidate_id in selected_ids,
                }
            )
        return candidates


def _section_events(
    section: Mapping[str, object],
    event_by_id: Mapping[str, ExecutionEvent],
) -> tuple[ExecutionEvent, ...]:
    raw_events = section.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes, bytearray)):
        return ()
    selected: list[ExecutionEvent] = []
    for raw in raw_events:
        item = _json_mapping(raw)
        event_id = item.get("event_id")
        if isinstance(event_id, str) and event_id in event_by_id:
            selected.append(event_by_id[event_id])
    return tuple(selected)


def _candidate_evidence(events: tuple[ExecutionEvent, ...]) -> tuple[ExecutionEvent, ...]:
    completed = tuple(event for event in events if event.event_type == "candidate.completed")
    if completed:
        return completed
    return tuple(event for event in events if event.event_type == "candidate.created")


def _selected_candidate_ids(events: tuple[ExecutionEvent, ...]) -> set[str]:
    return {
        str(candidate_id)
        for event in events
        if event.event_type == "candidate.selected" and (candidate_id := event.payload.get("candidate_id")) is not None
    }


def _candidate_statuses(events: tuple[ExecutionEvent, ...]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for event in events:
        if event.event_type not in {"candidate.retained", "candidate.pruned"}:
            continue
        candidate_id = event.payload.get("candidate_id")
        if candidate_id is not None:
            statuses[str(candidate_id)] = event.event_type.removeprefix("candidate.")
    return statuses


def _target_for_candidate(
    candidate: ExecutionEvent,
    events: tuple[ExecutionEvent, ...],
) -> ExecutionEvent | None:
    target_id = candidate.payload.get("target_response_event_id")
    if isinstance(target_id, str):
        return next((event for event in events if event.event_id == target_id), None)
    matches = [
        event
        for event in events
        if event.event_type == "target.completed"
        and event.branch_id == candidate.branch_id
        and event.round == candidate.round
    ]
    return matches[-1] if matches else None


def _judge_for_candidate(
    candidate_id: str,
    target: ExecutionEvent | None,
    events: tuple[ExecutionEvent, ...],
) -> ExecutionEvent | None:
    matches = [
        event
        for event in events
        if event.event_type in {"judge.completed", "judge.failed"}
        and (
            event.payload.get("candidate_id") == candidate_id
            or (target is not None and event.payload.get("target_response_event_id") == target.event_id)
        )
    ]
    return matches[-1] if matches else None


def _candidate_id_for_target(target: ExecutionEvent | None) -> str | None:
    if target is None:
        return None
    candidate_id = target.payload.get("candidate_id")
    if candidate_id is not None:
        return str(candidate_id)
    return target.branch_id


def _prompt_for_target(
    target: ExecutionEvent | None,
    *,
    events: tuple[ExecutionEvent, ...],
    history_snapshots: Mapping[str, HistorySnapshot],
) -> str:
    if target is None:
        return ""
    candidate = next(
        (
            event
            for event in reversed(events)
            if event.event_type in {"candidate.completed", "candidate.created"}
            and event.branch_id == target.branch_id
            and event.round == target.round
            and isinstance(event.payload.get("prompt"), str)
        ),
        None,
    )
    if candidate is not None:
        return str(candidate.payload["prompt"])
    snapshot_id = target.payload.get("history_snapshot_id")
    if isinstance(snapshot_id, str):
        snapshot = next(
            (item for item in history_snapshots.values() if item.snapshot_id == snapshot_id),
            None,
        )
        if snapshot is not None:
            prompts = [message for message in snapshot.messages if message.role == "user"]
            if prompts:
                return str(prompts[-1].content)
    prompts = [
        message
        for snapshot in history_snapshots.values()
        for message in snapshot.messages
        if message.role == "user" and (target.branch_id is None or message.branch_id == target.branch_id)
    ]
    if prompts:
        return str(prompts[-1].content)
    return ""


def _round_message_evidence(
    round_event: ExecutionEvent | None,
    *,
    target: ExecutionEvent | None,
    events: tuple[ExecutionEvent, ...],
    history_snapshots: Mapping[str, HistorySnapshot],
) -> dict[str, str]:
    messages = {
        message.message_id: message
        for snapshot in history_snapshots.values()
        for message in snapshot.messages
    }
    payload = round_event.payload if round_event is not None else {}

    def content(reference: object) -> str:
        if not isinstance(reference, str):
            return ""
        message = messages.get(reference)
        return "" if message is None else str(message.content)

    transformed = content(payload.get("transformed_message_id"))
    if not transformed and target is not None:
        transformations = [
            event
            for event in events
            if event.event_type == "value.transformed"
            and (
                event.caused_by == target.event_id
                or event.payload.get("original_response_event_id") == target.event_id
            )
        ]
        if transformations:
            transformation_ids = {event.event_id for event in transformations}
            transformed_message = next(
                (
                    message
                    for message in messages.values()
                    if message.source_event_id in transformation_ids
                ),
                None,
            )
            if transformed_message is not None:
                transformed = str(transformed_message.content)

    target_prompt = content(payload.get("target_prompt_message_id"))
    if not target_prompt:
        target_prompt = content(payload.get("audit_user_message_id"))

    return {
        "attacker_input": content(payload.get("attacker_input_message_id")),
        "attacker_output": content(payload.get("attacker_output_message_id")),
        "target_prompt": target_prompt,
        "transformed": transformed,
    }


def _target_response(event: ExecutionEvent | None) -> str:
    if event is None:
        return ""
    response = event.payload.get("response")
    return response if isinstance(response, str) else ""


def _judge(event: ExecutionEvent | None) -> dict[str, object]:
    if event is None:
        return {
            "status": "not_scheduled",
            "input": "",
            "output": "",
            "score": None,
            "rationale": "",
            "error": "",
        }
    succeeded = event.event_type == "judge.completed"
    return {
        "status": "succeeded" if succeeded else "failed",
        "input": str(event.payload.get("input") or ""),
        "output": str(event.payload.get("output") or ""),
        "score": event.payload.get("score") if succeeded else None,
        "rationale": str(event.payload.get("rationale") or ""),
        "error": "" if succeeded else str(event.payload.get("parse_status") or "judge failed"),
    }


def _judge_is_for_target(
    judge: ExecutionEvent | None,
    target: ExecutionEvent | None,
) -> bool:
    return (
        judge is not None
        and target is not None
        and judge.payload.get("target_response_event_id") == target.event_id
    )


def _json_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _integer(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
