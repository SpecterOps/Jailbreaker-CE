"""Canonical request composition for every modular judge invocation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from app.attacklib.iterative.prompts import JUDGE_SCHEMA
from app.judge_policy import (
    CUSTOM_JUDGE_SCHEMA,
    JudgePolicyClient,
    build_judge_messages,
    judge_policy_from_config,
)
from app.techniques.contracts.definition import JudgeDefinition
from app.techniques.core.judging import JudgeModelRequest


class CanonicalJudgeRequestComposer:
    """Compile stored evidence into the established judge prompt contract."""

    def __init__(self, clients: Mapping[str, object]) -> None:
        self._clients = MappingProxyType(dict(clients))

    def compose(
        self,
        *,
        definition: JudgeDefinition,
        evidence: Mapping[str, object],
    ) -> JudgeModelRequest:
        role = definition.role or "judge"
        client = self._clients.get(role)
        if client is None:
            raise KeyError(f"judge model role '{role}' has no client")
        intent = evidence.get("intent")
        response = evidence.get("response")
        if isinstance(response, Mapping):
            response = response.get("text")
        policy = judge_policy_from_config({"judge_policy": definition.policy})
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("judge evidence requires non-empty intent text")
        if not isinstance(response, str) or (policy.is_standard and not response):
            raise ValueError("judge evidence requires non-empty response text")
        effective_client = (
            client if policy.is_standard else JudgePolicyClient(client, policy)
        )
        messages = build_judge_messages(
            effective_client,
            intent=intent,
            response=response,
        )
        return JudgeModelRequest(
            messages=tuple(messages),
            options={
                "temperature": 0.0,
                "response_format": (
                    JUDGE_SCHEMA if policy.is_standard else CUSTOM_JUDGE_SCHEMA
                ),
            },
        )


__all__ = ["CanonicalJudgeRequestComposer"]
