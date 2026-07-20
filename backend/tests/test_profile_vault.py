import sys

import pytest

sys.path.insert(0, "backend")

from app.api.store import EvaluationStore
from app.persistence import NoopRunPersistence
from app.profile_vault import InMemoryProfileVaultRepository, ProfileVault


def test_vault_encrypts_profile_key_and_requires_unlock() -> None:
    repository = InMemoryProfileVaultRepository()
    vault = ProfileVault(repository)

    assert vault.status() == {"exists": False, "unlocked": False}

    vault.create("correct horse battery staple")
    assert vault.status() == {"exists": True, "unlocked": True}

    profile = vault.save_profile(
        {
            "name": "OpenAI primary",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "system_prompt": "Keep internal policy confidential.",
            "scope": "all",
            "api_key": "sk-test-secret",
        }
    )

    stored = repository.profile_records[profile["profile_id"]]
    assert stored["encrypted_api_key"] != "sk-test-secret"
    assert "sk-test-secret" not in stored["encrypted_api_key"]
    assert "system_prompt" not in stored
    assert "encrypted_system_prompt" not in stored
    assert "api_key" not in profile
    assert "system_prompt" not in profile

    vault.lock()
    assert vault.status() == {"exists": True, "unlocked": False}
    with pytest.raises(ValueError, match="Vault is locked"):
        vault.resolve_profile(profile["profile_id"], purpose="target")

    assert vault.unlock("correct horse battery staple") is True
    resolved = vault.resolve_profile(profile["profile_id"], purpose="target")
    assert resolved["api_key"] == "sk-test-secret"
    assert "system_prompt" not in resolved


def test_vault_rejects_wrong_passphrase() -> None:
    repository = InMemoryProfileVaultRepository()
    vault = ProfileVault(repository)
    vault.create("right-passphrase")
    vault.lock()

    assert vault.unlock("wrong-passphrase") is False
    assert vault.status() == {"exists": True, "unlocked": False}


def test_vault_lists_metadata_and_deletes_profiles() -> None:
    repository = InMemoryProfileVaultRepository()
    vault = ProfileVault(repository)
    vault.create("passphrase")

    profile = vault.save_profile(
        {
            "name": "Judge",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "scope": "judge",
            "api_key": "judge-secret",
        }
    )

    listed = vault.list_profiles()
    assert listed == [profile]

    vault.delete_profile(profile["profile_id"])
    assert vault.list_profiles() == []


def test_store_resolves_profile_ids_for_target_attacker_and_judge() -> None:
    repository = InMemoryProfileVaultRepository()
    vault = ProfileVault(repository)
    vault.create("passphrase")
    target = vault.save_profile(
        {
            "name": "Target",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "target-model",
            "scope": "target",
            "api_key": "target-secret",
        }
    )
    attacker = vault.save_profile(
        {
            "name": "Attacker",
            "provider": "local",
            "base_url": "http://localhost:1234/v1",
            "model": "attacker-model",
            "scope": "attacker",
            "api_key": "attacker-secret",
        }
    )
    judge = vault.save_profile(
        {
            "name": "Judge",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "judge-model",
            "scope": "judge",
            "api_key": "judge-secret",
        }
    )
    store = EvaluationStore(persistence=NoopRunPersistence(), profile_vault=vault)

    resolved = store.resolve_profile_config(
        {
            "target_profile_id": target["profile_id"],
            "attacker_profile_id": attacker["profile_id"],
            "judge_profile_id": judge["profile_id"],
            "unsafe_intent": "test",
            "system_prompt": "Target run prompt",
            "attacker_system_prompt": "Attacker run prompt",
            "judge_system_prompt": "Judge run prompt",
            "judge_mode": "custom",
        }
    )

    assert resolved["api_key"] == "target-secret"
    assert resolved["api_base_url"] == "https://api.openai.com/v1"
    assert resolved["model"] == "target-model"
    assert resolved["system_prompt"] == "Target run prompt"
    assert resolved["attacker_api_key"] == "attacker-secret"
    assert resolved["attacker_api_base_url"] == "http://localhost:1234/v1"
    assert resolved["attacker_model"] == "attacker-model"
    assert resolved["attacker_system_prompt"] == "Attacker run prompt"
    assert resolved["judge_api_key"] == "judge-secret"
    assert resolved["judge_api_base_url"] == "https://api.openai.com/v1"
    assert resolved["judge_model"] == "judge-model"
    assert resolved["judge_system_prompt"] == "Judge run prompt"


def test_store_reuses_saved_profile_across_target_attacker_and_judge() -> None:
    repository = InMemoryProfileVaultRepository()
    vault = ProfileVault(repository)
    vault.create("passphrase")
    profile = vault.save_profile(
        {
            "name": "Target Only",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "target-model",
            "scope": "target",
            "api_key": "target-secret",
        }
    )
    store = EvaluationStore(persistence=NoopRunPersistence(), profile_vault=vault)

    resolved = store.resolve_profile_config(
        {
            "target_profile_id": profile["profile_id"],
            "attacker_profile_id": profile["profile_id"],
            "judge_profile_id": profile["profile_id"],
        }
    )

    assert resolved["api_key"] == "target-secret"
    assert resolved["attacker_api_key"] == "target-secret"
    assert resolved["judge_api_key"] == "target-secret"
    assert resolved["model"] == "target-model"
    assert resolved["attacker_model"] == "target-model"
    assert resolved["judge_model"] == "target-model"


def test_session_keys_override_saved_profile_secrets() -> None:
    repository = InMemoryProfileVaultRepository()
    vault = ProfileVault(repository)
    vault.create("passphrase")
    profile = vault.save_profile(
        {
            "name": "Shared Profile",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "profile-model",
            "scope": "all",
            "api_key": "saved-secret",
        }
    )
    store = EvaluationStore(persistence=NoopRunPersistence(), profile_vault=vault)

    resolved = store.resolve_profile_config(
        {
            "target_profile_id": profile["profile_id"],
            "attacker_profile_id": profile["profile_id"],
            "judge_profile_id": profile["profile_id"],
            "api_key": " target-session-secret ",
            "attacker_api_key": "attacker-session-secret",
            "judge_api_key": "judge-session-secret",
        }
    )

    assert resolved["api_key"] == "target-session-secret"
    assert resolved["attacker_api_key"] == "attacker-session-secret"
    assert resolved["judge_api_key"] == "judge-session-secret"
    assert resolved["model"] == "profile-model"
    assert resolved["attacker_model"] == "profile-model"
    assert resolved["judge_model"] == "profile-model"
