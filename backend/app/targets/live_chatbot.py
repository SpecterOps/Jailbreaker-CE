"""Live chatbot adapter that sends rendered prompts to an OpenAI-compatible API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.attacklib.renderer import render_prompt, render_scripted_turns
from app.evaluation.models import (
    ExecutionRequest,
    TargetKind,
    TargetResponse,
    TraceEvent,
)
from app.llm_client import ChatClient

from .base import TargetAdapter, TargetDescriptor
from .live_common import messages_to_trace_input, parse_generation_overrides

log = logging.getLogger("targets.live_chatbot")


@dataclass(slots=True)
class LiveChatbotAdapter(TargetAdapter):
    """Sends the rendered attack-case prompt to an OpenAI-compatible chat completions API."""

    descriptor: TargetDescriptor
    client: ChatClient = field(default_factory=ChatClient)
    system_prompt: str = "You are a helpful assistant."

    def execute(self, request: ExecutionRequest) -> TargetResponse:
        generation_overrides = parse_generation_overrides(
            request.case.generation_overrides
        )

        if request.case.conversation_turns:
            return self._execute_scripted(request, generation_overrides)

        messages = render_prompt(request.case, system_prompt=self.system_prompt)
        log.info(
            "execute technique=%s model=%s messages=%d",
            request.case.technique_key, self.client.model, len(messages),
        )
        log.info("prompt: %s", messages[-1].get("content", "")[:300])

        # Capture the full message list (including system prompt) so the
        # conversation log can show exactly what was sent to the target.
        full_input_str = messages_to_trace_input(messages)

        traces: list[TraceEvent] = [
            TraceEvent(
                event_type="prompt_rendered",
                actor="live_chatbot_adapter",
                message=f"Rendered {len(messages)} message(s) for model={self.client.model}.",
                attributes={
                    "model": self.client.model,
                    "message_count": str(len(messages)),
                    "full_input": full_input_str,
                    "system_prompt": self.system_prompt,
                    **(
                        {"generation_overrides": str(generation_overrides)}
                        if generation_overrides
                        else {}
                    ),
                },
            ),
        ]

        try:
            response_text, usage = self.client.chat(
                messages,
                extra_body=generation_overrides or None,
            )
        except Exception as exc:
            log.error("API call failed technique=%s: %s", request.case.technique_key, exc)
            traces.append(
                TraceEvent(
                    event_type="api_error",
                    actor="live_chatbot_adapter",
                    message=f"API call failed: {exc}",
                )
            )
            return TargetResponse(
                target_id=self.descriptor.target_id,
                target_kind=TargetKind.CHATBOT,
                final_output=f"[ERROR] {exc}",
                traces=tuple(traces),
                metadata={
                    "adapter": "live-chatbot",
                    "technique_key": request.case.technique_key,
                    "error": str(exc),
                },
            )

        traces.append(
            TraceEvent(
                event_type="api_response",
                actor="live_chatbot_adapter",
                message=f"Received response ({usage.get('total_tokens', '?')} tokens).",
                attributes=usage,
            ),
        )

        return TargetResponse(
            target_id=self.descriptor.target_id,
            target_kind=TargetKind.CHATBOT,
            final_output=response_text,
            traces=tuple(traces),
            metadata={
                "adapter": "live-chatbot",
                "technique_key": request.case.technique_key,
                "model": self.client.model,
                **usage,
            },
        )

    def _execute_scripted(
        self,
        request: ExecutionRequest,
        generation_overrides: dict[str, object],
    ) -> TargetResponse:
        planned_messages = render_scripted_turns(
            request.case,
            system_prompt=self.system_prompt,
        )
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        traces: list[TraceEvent] = [
            TraceEvent(
                event_type="prompt_rendered",
                actor="live_chatbot_adapter",
                message=(
                    f"Rendered {len(planned_messages)} scripted message(s) "
                    f"for model={self.client.model}."
                ),
                attributes={
                    "model": self.client.model,
                    "message_count": str(len(planned_messages)),
                    "full_input": messages_to_trace_input(planned_messages),
                    "system_prompt": self.system_prompt,
                    "scripted": "true",
                    **(
                        {"generation_overrides": str(generation_overrides)}
                        if generation_overrides
                        else {}
                    ),
                },
            ),
        ]

        response_text = ""
        final_usage: dict[str, str] = {}
        phase = 0
        turns = request.case.conversation_turns

        for index, turn in enumerate(turns):
            role = turn["role"]
            content = turn["content"].replace(
                "{intent}",
                request.case.unsafe_intent_label,
            )
            messages.append({"role": role, "content": content})

            if role != "user":
                continue

            next_turn = turns[index + 1] if index + 1 < len(turns) else None
            if next_turn and next_turn.get("role") == "assistant":
                continue

            phase += 1
            later_user_turn_exists = any(
                later_turn.get("role") == "user"
                for later_turn in turns[index + 1 :]
            )
            traces.append(
                TraceEvent(
                    event_type="api_call",
                    actor="live_chatbot_adapter",
                    message=f"Calling model for scripted phase {phase}.",
                    attributes={
                        "phase": str(phase),
                        "turn_index": str(index),
                        "message_count": str(len(messages)),
                        "final": str(not later_user_turn_exists).lower(),
                        "full_input": messages_to_trace_input(messages),
                    },
                )
            )

            try:
                response_text, usage = self.client.chat(
                    messages,
                    extra_body=generation_overrides or None,
                )
            except Exception as exc:
                log.error(
                    "API call failed technique=%s phase=%d: %s",
                    request.case.technique_key,
                    phase,
                    exc,
                )
                traces.append(
                    TraceEvent(
                        event_type="api_error",
                        actor="live_chatbot_adapter",
                        message=f"Scripted phase {phase} API call failed: {exc}",
                        attributes={"phase": str(phase)},
                    )
                )
                return TargetResponse(
                    target_id=self.descriptor.target_id,
                    target_kind=TargetKind.CHATBOT,
                    final_output=f"[ERROR] {exc}",
                    traces=tuple(traces),
                    metadata={
                        "adapter": "live-chatbot",
                        "technique_key": request.case.technique_key,
                        "error": str(exc),
                        "phase": str(phase),
                    },
                )

            traces.append(
                TraceEvent(
                    event_type="api_response",
                    actor="live_chatbot_adapter",
                    message=(
                        f"Received scripted phase {phase} response "
                        f"({usage.get('total_tokens', '?')} tokens)."
                    ),
                    attributes={"phase": str(phase), **usage},
                )
            )
            final_usage = usage

            if later_user_turn_exists:
                messages.append({"role": "assistant", "content": response_text})

        return TargetResponse(
            target_id=self.descriptor.target_id,
            target_kind=TargetKind.CHATBOT,
            final_output=response_text,
            traces=tuple(traces),
            metadata={
                "adapter": "live-chatbot",
                "technique_key": request.case.technique_key,
                "model": self.client.model,
                "scripted_phases": str(phase),
                **final_usage,
            },
        )
