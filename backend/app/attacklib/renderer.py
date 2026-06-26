"""Render an AttackCase into sendable prompt messages.

Each technique stores a ``template`` key in its ``assets`` dict with
``{intent}`` placeholders.  The renderer hydrates the template and returns
a list of chat-completion-style message dicts ready for an LLM API.

Optionally applies a ``ConverterPipeline`` to the user content before
building the messages — this is how encoding-based evasion (Base64,
leetspeak, homoglyphs, etc.) composes with any attack technique.
"""

from __future__ import annotations

from app.attacklib.base import AttackCase
from app.attacklib.converters import ConverterPipeline


def render_prompt(
    case: AttackCase,
    *,
    system_prompt: str | None = None,
    converter_pipeline: ConverterPipeline | None = None,
) -> list[dict[str, str]]:
    """Turn an AttackCase into a list of ``{"role": ..., "content": ...}`` messages.

    Resolution order for the user message:
    1. ``case.assets["template"]`` with ``{intent}`` replaced by ``unsafe_intent_label``.
    2. First variant template (``template_ignore``, ``template_direct``, etc.).
    3. ``case.assets["seed_prompt"]`` if no template exists.
    4. ``case.instructions`` as a final fallback.

    If *converter_pipeline* is provided, the resolved user content is passed
    through the pipeline before being added to the messages.
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = _resolve_user_content(case)
    if converter_pipeline:
        user_content = converter_pipeline.convert(user_content)
    messages.append({"role": "user", "content": user_content})

    return messages


def render_multi_turn(
    case: AttackCase,
    *,
    system_prompt: str | None = None,
    prior_turns: list[dict[str, str]] | None = None,
    converter_pipeline: ConverterPipeline | None = None,
) -> list[dict[str, str]]:
    """Build a multi-turn conversation from the case plus prior assistant/user turns.

    ``prior_turns`` should be a list of ``{"role": "assistant"|"user", "content": ...}``
    dicts representing the conversation so far.  The rendered case prompt is appended
    as the final user message.

    If *converter_pipeline* is provided, only the **new** user content is
    converted — prior turns are left as-is (they've already been sent).
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if prior_turns:
        messages.extend(prior_turns)

    user_content = _resolve_user_content(case)
    if converter_pipeline:
        user_content = converter_pipeline.convert(user_content)
    messages.append({"role": "user", "content": user_content})

    return messages


def render_scripted_turns(
    case: AttackCase,
    *,
    system_prompt: str | None = None,
    converter_pipeline: ConverterPipeline | None = None,
) -> list[dict[str, str]]:
    """Render an explicit scripted conversation stored on the case."""
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if not case.conversation_turns:
        messages.extend(render_prompt(case, converter_pipeline=converter_pipeline))
        return messages

    for turn in case.conversation_turns:
        role = turn["role"]
        content = turn["content"].replace("{intent}", case.unsafe_intent_label)
        if converter_pipeline and role == "user":
            content = converter_pipeline.convert(content)
        messages.append({"role": role, "content": content})

    return messages


def render_untrusted_document_block(case: AttackCase) -> str:
    """Render untrusted document assets with explicit trust-boundary delimiters."""
    blocks: list[str] = []
    for document in case.untrusted_documents:
        name = document.get("name", "document")
        content = document.get("content", "").replace("{intent}", case.unsafe_intent_label)
        blocks.append(
            "\n".join(
                [
                    f"BEGIN UNTRUSTED DOCUMENT: {name}",
                    content,
                    "END UNTRUSTED DOCUMENT",
                ]
            )
        )
    return "\n\n".join(blocks)


def render_workspace_block(case: AttackCase) -> str:
    """Render workspace files with explicit file-boundary delimiters."""
    blocks: list[str] = []
    for workspace_file in case.workspace_files:
        path = workspace_file.get("path", "workspace-file")
        content = workspace_file.get("content", "").replace("{intent}", case.unsafe_intent_label)
        blocks.append(
            "\n".join(
                [
                    f"BEGIN WORKSPACE FILE: {path}",
                    content,
                    "END WORKSPACE FILE",
                ]
            )
        )
    return "\n\n".join(blocks)


def _resolve_user_content(case: AttackCase) -> str:
    user_content = _resolve_trusted_user_content(case)
    appended_blocks = [
        block
        for block in (
            render_untrusted_document_block(case),
            render_workspace_block(case),
        )
        if block
    ]
    if appended_blocks:
        return "\n\n".join([user_content, *appended_blocks])
    return user_content


def _resolve_trusted_user_content(case: AttackCase) -> str:
    intent = case.unsafe_intent_label

    # 1. Explicit authored template from the technique definition.
    template = case.assets.get("template", "")
    if template:
        return template.replace("{intent}", intent)

    # 2. First variant template (template_ignore, template_direct, etc.)
    for key, value in case.assets.items():
        if key.startswith("template_") and value:
            return value.replace("{intent}", intent)

    # 3. Seed prompt sourced from config / DB seed library.
    seed_prompt = case.assets.get("seed_prompt", "")
    if seed_prompt:
        return seed_prompt.replace("{intent}", intent)

    # 4. Fall back to the full technique instructions (always substantive)
    return case.instructions
