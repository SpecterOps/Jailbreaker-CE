"""Tests for the centralized conversation memory system."""

from app.memory.conversation import ConversationMemory
from app.memory.backends import InMemoryBackend


def _make_memory() -> ConversationMemory:
    return ConversationMemory(backend=InMemoryBackend())


def test_create_session_and_add_messages():
    mem = _make_memory()
    session = mem.create_session(labels={"technique": "test"})

    session.add_system("You are helpful.")
    session.add_user("Hello")
    session.add_assistant("Hi there!")

    assert len(session) == 3
    msgs = session.to_messages()
    assert msgs == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


def test_turn_count():
    mem = _make_memory()
    session = mem.create_session()

    session.add_system("sys")
    session.add_user("turn 1")
    session.add_assistant("resp 1")
    session.add_user("turn 2")
    session.add_assistant("resp 2")

    assert session.turn_count == 2


def test_rollback_removes_last_n():
    mem = _make_memory()
    session = mem.create_session()

    session.add_system("sys")
    session.add_user("user msg")
    session.add_assistant("refused!")

    removed = session.rollback(2)
    assert removed == 2
    assert len(session) == 1
    assert session.messages[0].role == "system"


def test_rollback_more_than_exists():
    mem = _make_memory()
    session = mem.create_session()

    session.add_user("only message")
    removed = session.rollback(5)
    assert removed == 1
    assert len(session) == 0


def test_format_history_skips_system():
    mem = _make_memory()
    session = mem.create_session()

    session.add_system("hidden")
    session.add_user("question")
    session.add_assistant("answer")

    history = session.format_history()
    assert "hidden" not in history
    assert "[USER]" in history
    assert "[TARGET]" in history
    assert "question" in history
    assert "answer" in history


def test_format_history_includes_system_when_requested():
    mem = _make_memory()
    session = mem.create_session()

    session.add_system("system prompt")
    session.add_user("hello")

    history = session.format_history(include_system=True)
    assert "[SYSTEM]" in history
    assert "system prompt" in history


def test_empty_session_format():
    mem = _make_memory()
    session = mem.create_session()

    assert session.format_history() == "(no messages yet)"


def test_list_conversations():
    mem = _make_memory()
    s1 = mem.create_session(conversation_id="conv-1")
    s2 = mem.create_session(conversation_id="conv-2")

    s1.add_user("msg")
    s2.add_user("msg")

    convs = mem.list_conversations()
    assert set(convs) == {"conv-1", "conv-2"}


def test_resume_session():
    mem = _make_memory()
    session = mem.create_session(conversation_id="resume-test")
    session.add_system("sys")
    session.add_user("first")

    # Resume the same conversation
    resumed = mem.get_session("resume-test")
    assert len(resumed) == 2
    resumed.add_assistant("second")
    assert len(resumed) == 3


def test_clear_session():
    mem = _make_memory()
    session = mem.create_session(conversation_id="clear-test")
    session.add_user("a")
    session.add_user("b")

    count = session.clear()
    assert count == 2
    assert len(session) == 0


def test_last_message():
    mem = _make_memory()
    session = mem.create_session()

    assert session.last_message is None
    session.add_user("first")
    session.add_assistant("second")
    assert session.last_message is not None
    assert session.last_message.content == "second"
    assert session.last_message.role == "assistant"


def test_messages_have_labels():
    mem = _make_memory()
    session = mem.create_session(labels={"technique": "crescendo"})
    session.add_user("hello", turn="1")

    msg = session.messages[0]
    assert msg.labels["technique"] == "crescendo"
    assert msg.labels["turn"] == "1"


def test_messages_have_timestamps():
    mem = _make_memory()
    session = mem.create_session()
    session.add_user("test")

    msg = session.messages[0]
    assert msg.timestamp is not None
    assert msg.timestamp.tzinfo is not None  # timezone-aware


def test_append_after_rollback_fills_gap():
    """After rollback, new messages should get correct sequence numbers."""
    mem = _make_memory()
    session = mem.create_session()

    session.add_system("sys")       # seq 0
    session.add_user("attempt 1")   # seq 1
    session.add_assistant("refused") # seq 2

    session.rollback(2)              # removes seq 1 and 2

    session.add_user("attempt 2")   # should get seq 1
    session.add_assistant("ok")      # should get seq 2

    msgs = session.to_messages()
    assert len(msgs) == 3
    assert msgs[1]["content"] == "attempt 2"
    assert msgs[2]["content"] == "ok"
