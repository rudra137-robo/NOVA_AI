
from app import memory


def test_messages_survive_new_database_connections(
    tmp_path, monkeypatch
):
    # Use a temporary database, not NOVA's real database.
    test_db = tmp_path / "test_nova.db"
    monkeypatch.setattr(memory, "DB_PATH", str(test_db))

    memory.init_db()

    conversation_id = "test-conversation-1"

    memory.save_messages(conversation_id, [
        {"role": "user", "content": "My project is NOVA."},
        {"role": "assistant", "content": "Got it."},
    ])

    # get_history opens a fresh database connection.
    history = memory.get_history(conversation_id)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "My project is NOVA."
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Got it."


def test_delete_conversation():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        from pathlib import Path

        test_db = Path(directory) / "delete_test.db"
        original_path = memory.DB_PATH

        try:
            memory.DB_PATH = str(test_db)
            memory.init_db()

            conversation_id = "delete-me"

            memory.save_messages(conversation_id, [
                {"role": "user", "content": "Temporary message"},
            ])

            memory.delete_conversation(conversation_id)

            assert memory.get_history(conversation_id) == []
        finally:
            memory.DB_PATH = original_path
