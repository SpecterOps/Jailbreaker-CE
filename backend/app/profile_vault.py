from __future__ import annotations

import base64
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

VAULT_ID = "default"
KDF_NAME = "pbkdf2-sha256"
KDF_ITERATIONS = 600_000
CANARY = b"jailbreaker-vault-v1"
PROFILE_SCOPES = {"target", "attacker", "judge", "all"}


@dataclass(frozen=True)
class VaultRecord:
    vault_id: str
    salt_b64: str
    kdf_name: str
    kdf_iterations: int
    encrypted_canary: str
    created_at: str = ""
    updated_at: str = ""


class ProfileVaultRepository(Protocol):
    def get_vault(self) -> VaultRecord | None:
        ...

    def save_vault(self, record: VaultRecord) -> None:
        ...

    def list_profiles(self) -> list[dict[str, Any]]:
        ...

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        ...

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        ...

    def delete_profile(self, profile_id: str) -> None:
        ...


class InMemoryProfileVaultRepository:
    def __init__(self) -> None:
        self.vault_record: VaultRecord | None = None
        self.profile_records: dict[str, dict[str, Any]] = {}

    def get_vault(self) -> VaultRecord | None:
        return self.vault_record

    def save_vault(self, record: VaultRecord) -> None:
        self.vault_record = record

    def list_profiles(self) -> list[dict[str, Any]]:
        return sorted(
            (dict(profile) for profile in self.profile_records.values()),
            key=lambda item: (str(item.get("name", "")).lower(), str(item.get("profile_id", ""))),
        )

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        profile = self.profile_records.get(profile_id)
        return dict(profile) if profile else None

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(profile.get("profile_id") or f"profile-{uuid.uuid4().hex[:12]}")
        now = datetime.now(UTC).isoformat()
        existing = self.profile_records.get(profile_id, {})
        item = {
            **existing,
            **profile,
            "profile_id": profile_id,
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        }
        self.profile_records[profile_id] = item
        return dict(item)

    def delete_profile(self, profile_id: str) -> None:
        self.profile_records.pop(profile_id, None)


class PostgreSQLProfileVaultRepository:
    def __init__(self, dsn: str | None = None) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "PostgreSQLProfileVaultRepository requires the 'psycopg' package. "
                "Install it with: uv sync --extra postgres"
            ) from None
        self._dsn = dsn or os.environ.get(
            "DATABASE_URL",
            "postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker",
        )

    def get_vault(self) -> VaultRecord | None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT vault_id, salt_b64, kdf_name, kdf_iterations,
                       encrypted_canary, created_at, updated_at
                FROM llm_vault
                WHERE vault_id = %s
                """,
                (VAULT_ID,),
            ).fetchone()
        if not row:
            return None
        return VaultRecord(
            vault_id=row[0],
            salt_b64=row[1],
            kdf_name=row[2],
            kdf_iterations=int(row[3]),
            encrypted_canary=row[4],
            created_at=row[5].isoformat() if row[5] else "",
            updated_at=row[6].isoformat() if row[6] else "",
        )

    def save_vault(self, record: VaultRecord) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO llm_vault (
                    vault_id, salt_b64, kdf_name, kdf_iterations, encrypted_canary
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (vault_id) DO UPDATE SET
                    salt_b64 = EXCLUDED.salt_b64,
                    kdf_name = EXCLUDED.kdf_name,
                    kdf_iterations = EXCLUDED.kdf_iterations,
                    encrypted_canary = EXCLUDED.encrypted_canary,
                    updated_at = now()
                """,
                (
                    record.vault_id,
                    record.salt_b64,
                    record.kdf_name,
                    record.kdf_iterations,
                    record.encrypted_canary,
                ),
            )

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT profile_id, name, provider, base_url, model, scope,
                       encrypted_api_key, created_at, updated_at
                FROM llm_target_profiles
                ORDER BY lower(name), profile_id
                """
            ).fetchall()
        return [self._row_to_profile(row) for row in rows]

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT profile_id, name, provider, base_url, model, scope,
                       encrypted_api_key, created_at, updated_at
                FROM llm_target_profiles
                WHERE profile_id = %s
                """,
                (profile_id,),
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(profile.get("profile_id") or f"profile-{uuid.uuid4().hex[:12]}")
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                INSERT INTO llm_target_profiles (
                    profile_id, name, provider, base_url, model, scope, encrypted_api_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    provider = EXCLUDED.provider,
                    base_url = EXCLUDED.base_url,
                    model = EXCLUDED.model,
                    scope = EXCLUDED.scope,
                    encrypted_api_key = EXCLUDED.encrypted_api_key,
                    updated_at = now()
                RETURNING profile_id, name, provider, base_url, model, scope,
                          encrypted_api_key, created_at, updated_at
                """,
                (
                    profile_id,
                    str(profile.get("name", "")),
                    str(profile.get("provider", "")),
                    str(profile.get("base_url", "")),
                    str(profile.get("model", "")),
                    str(profile.get("scope", "all")),
                    str(profile.get("encrypted_api_key", "")),
                ),
            ).fetchone()
        return self._row_to_profile(row)

    def delete_profile(self, profile_id: str) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute("DELETE FROM llm_target_profiles WHERE profile_id = %s", (profile_id,))

    def ensure_schema(self, conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_vault (
                vault_id TEXT PRIMARY KEY,
                salt_b64 TEXT NOT NULL,
                kdf_name TEXT NOT NULL,
                kdf_iterations INTEGER NOT NULL,
                encrypted_canary TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_target_profiles (
                profile_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT '',
                base_url TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                scope TEXT NOT NULL DEFAULT 'all',
                encrypted_api_key TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_target_profiles_scope ON llm_target_profiles (scope)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_target_profiles_name ON llm_target_profiles (lower(name))")

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def _row_to_profile(self, row) -> dict[str, Any]:
        return {
            "profile_id": row[0],
            "name": row[1],
            "provider": row[2],
            "base_url": row[3],
            "model": row[4],
            "scope": row[5],
            "encrypted_api_key": row[6],
            "created_at": row[7].isoformat() if row[7] else "",
            "updated_at": row[8].isoformat() if row[8] else "",
        }


class ProfileVault:
    def __init__(self, repository: ProfileVaultRepository) -> None:
        self._repository = repository
        self._fernet: Fernet | None = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, bool]:
        return {
            "exists": self._repository.get_vault() is not None,
            "unlocked": self._fernet is not None,
        }

    def create(self, passphrase: str) -> dict[str, bool]:
        self._require_passphrase(passphrase)
        with self._lock:
            if self._repository.get_vault() is not None:
                raise ValueError("Vault already exists.")
            salt = os.urandom(16)
            fernet = Fernet(_derive_fernet_key(passphrase, salt, KDF_ITERATIONS))
            record = VaultRecord(
                vault_id=VAULT_ID,
                salt_b64=base64.b64encode(salt).decode("ascii"),
                kdf_name=KDF_NAME,
                kdf_iterations=KDF_ITERATIONS,
                encrypted_canary=fernet.encrypt(CANARY).decode("ascii"),
            )
            self._repository.save_vault(record)
            self._fernet = fernet
        return self.status()

    def unlock(self, passphrase: str) -> bool:
        self._require_passphrase(passphrase)
        record = self._repository.get_vault()
        if record is None:
            raise ValueError("Vault has not been created.")
        if record.kdf_name != KDF_NAME:
            raise ValueError(f"Unsupported vault KDF: {record.kdf_name}")
        salt = base64.b64decode(record.salt_b64.encode("ascii"))
        fernet = Fernet(_derive_fernet_key(passphrase, salt, record.kdf_iterations))
        try:
            if fernet.decrypt(record.encrypted_canary.encode("ascii")) != CANARY:
                return False
        except InvalidToken:
            return False
        with self._lock:
            self._fernet = fernet
        return True

    def lock(self) -> None:
        with self._lock:
            self._fernet = None

    def list_profiles(self) -> list[dict[str, Any]]:
        return [_public_profile(profile) for profile in self._repository.list_profiles()]

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        fernet = self._require_unlocked()
        name = str(profile.get("name", "")).strip()
        base_url = str(profile.get("base_url", "")).strip()
        model = str(profile.get("model", "")).strip()
        if not name:
            raise ValueError("Profile `name` is required.")
        if not base_url:
            raise ValueError("Profile `base_url` is required.")
        if not model:
            raise ValueError("Profile `model` is required.")
        scope = str(profile.get("scope", "all")).strip() or "all"
        if scope not in PROFILE_SCOPES:
            raise ValueError("Profile `scope` must be target, attacker, judge, or all.")

        api_key = str(profile.get("api_key", ""))
        existing = None
        if profile.get("profile_id"):
            existing = self._repository.get_profile(str(profile["profile_id"]))
        if not api_key and existing:
            encrypted_api_key = str(existing.get("encrypted_api_key", ""))
        else:
            encrypted_api_key = fernet.encrypt(api_key.encode("utf-8")).decode("ascii")

        saved = self._repository.save_profile(
            {
                "profile_id": str(profile.get("profile_id", "")) or None,
                "name": name,
                "provider": str(profile.get("provider", "")).strip(),
                "base_url": base_url,
                "model": model,
                "scope": scope,
                "encrypted_api_key": encrypted_api_key,
            }
        )
        return _public_profile(saved)

    def resolve_profile(self, profile_id: str, *, purpose: str) -> dict[str, str]:
        fernet = self._require_unlocked()
        profile = self._repository.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"LLM profile not found: {profile_id}")
        scope = str(profile.get("scope", "all"))
        token = str(profile.get("encrypted_api_key", ""))
        try:
            api_key = fernet.decrypt(token.encode("ascii")).decode("utf-8") if token else ""
        except InvalidToken as exc:
            raise ValueError("Stored profile API key cannot be decrypted with the unlocked vault.") from exc
        return {
            "profile_id": str(profile.get("profile_id", "")),
            "name": str(profile.get("name", "")),
            "provider": str(profile.get("provider", "")),
            "base_url": str(profile.get("base_url", "")),
            "model": str(profile.get("model", "")),
            "scope": scope,
            "api_key": api_key,
        }

    def delete_profile(self, profile_id: str) -> None:
        self._repository.delete_profile(profile_id)

    def _require_unlocked(self) -> Fernet:
        if self._fernet is None:
            raise ValueError("Vault is locked.")
        return self._fernet

    def _require_passphrase(self, passphrase: str) -> None:
        if not str(passphrase or "").strip():
            raise ValueError("Vault passphrase is required.")


def build_profile_vault() -> ProfileVault:
    try:
        return ProfileVault(PostgreSQLProfileVaultRepository())
    except Exception:
        return ProfileVault(InMemoryProfileVaultRepository())


def _derive_fernet_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def _public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": str(profile.get("profile_id", "")),
        "name": str(profile.get("name", "")),
        "provider": str(profile.get("provider", "")),
        "base_url": str(profile.get("base_url", "")),
        "model": str(profile.get("model", "")),
        "scope": str(profile.get("scope", "all")),
        "created_at": str(profile.get("created_at", "")),
        "updated_at": str(profile.get("updated_at", "")),
    }
