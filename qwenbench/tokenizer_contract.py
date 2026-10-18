"""Tokenizer compatibility contract (§3.5) — pure comparison logic.

Speculative decoding compares token IDs: a single shifted ID silently turns
verification into noise, so §3.5 of the plan turns the check into a contract
with five hard requirements:

1. identical (token -> ID) pairs over the whole shared vocabulary;
2. identical IDs for ``bos``, ``eos``, ``pad``, ``unk`` and the chat tokens of
   the target template;
3. any extra target token sits strictly beyond the base vocabulary (no shared
   ID is shifted);
4. ``apply_chat_template(target template)`` yields identical IDs on both
   tokenizers;
5. a missing tokenizer file is a failure, never a warning.

This module implements the *logic* (plain dicts in, plain dicts out), so it is
testable offline: no network, no GPU, no transformers. ``check_tokenizer_compat.py``
at the repository root is the CLI that feeds it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: Files that must exist on both sides. Missing one is a failure (§3.5, req. 5).
REQUIRED_FILES: tuple[str, ...] = ("tokenizer_config.json", "tokenizer.json")

#: Files that define the base vocabulary byte for byte. If both sides ship them,
#: a difference means the base vocabulary itself is not the same.
BASE_VOCAB_FILES: tuple[str, ...] = ("vocab.json", "merges.txt")

#: Every file the report looks at, in display order.
WATCHED_FILES: tuple[str, ...] = REQUIRED_FILES + BASE_VOCAB_FILES + ("special_tokens_map.json",)

#: Differences explicitly tolerated (§3.5): they change neither the vocabulary
#: nor the encoding. `tokenizer.json` and `special_tokens_map.json` may carry the
#: target's extra audio/TTS tokens, hence a warning rather than a failure.
TOLERATED_CONFIG_FIELDS: tuple[str, ...] = (
    "chat_template",
    "clean_up_tokenization_spaces",
    "add_prefix_space",
)

#: Severity of a failed check: "fail" breaks the verdict, "warn" is reported only.
Severity = Literal["fail", "warn"]

#: Roles whose IDs must match exactly (requirement 2).
REQUIRED_ROLES: tuple[str, ...] = ("bos", "eos", "pad", "unk")

#: Special tokens referenced by the chat template, e.g. ``<|im_start|>``.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_.\-]+\|>")


@dataclass(slots=True)
class Check:
    """One requirement of the contract.

    ``ok=None`` means "not testable with these inputs", which is different from
    a failure (same convention as ``qwenbench.report``).
    """

    id: str
    ok: bool | None
    detail: str
    severity: Severity = "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ok": self.ok,
            "severity": self.severity,
            "detail": self.detail,
        }


def summarize(checks: Sequence[Check]) -> dict[str, Any]:
    """Aggregate the checks into the contract verdict."""
    failed = [c.id for c in checks if c.severity == "fail" and c.ok is False]
    warned = [c.id for c in checks if c.severity == "warn" and c.ok is False]
    not_applicable = [c.id for c in checks if c.ok is None]
    return {
        "ok": not failed,
        "failed": failed,
        "warned": warned,
        "not_applicable": not_applicable,
    }


def sha256_bytes(data: bytes) -> str:
    """SHA256 of a file's bytes, as recorded in the run manifest (§6.4)."""
    return hashlib.sha256(data).hexdigest()


def compare_files(
    draft: Mapping[str, str | None],
    target: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    """Byte-level comparison of the tokenizer files (requirements 1 and 5).

    A missing file is fatal, except when the file is absent on *both* sides:
    that is a format choice (e.g. a single-file ``tokenizer.json``), not a
    failure. Byte drift on ``vocab.json`` / ``merges.txt`` is fatal because it
    moves the base vocabulary; drift on ``tokenizer.json`` /
    ``special_tokens_map.json`` is tolerated (extra audio/TTS tokens).
    """
    rows: list[dict[str, Any]] = []
    for name in WATCHED_FILES:
        a, b = draft.get(name), target.get(name)
        row: dict[str, Any] = {
            "file": name,
            "draft_sha256": a,
            "target_sha256": b,
        }
        if a is None and b is None:
            row["status"] = "absent"
            row["severity"] = "warn"
        elif a is None or b is None:
            row["status"] = "missing"
            row["severity"] = "fail"
        elif a == b:
            row["status"] = "identical"
            row["severity"] = "warn"
        elif name in BASE_VOCAB_FILES:
            row["status"] = "different"
            row["severity"] = "fail"
        else:
            row["status"] = "different"
            row["severity"] = "warn"
        rows.append(row)
    return rows


def file_check(rows: Sequence[Mapping[str, Any]]) -> Check:
    """Requirement 5: a missing tokenizer file is a failure, never a warning."""
    broken = [r["file"] for r in rows if r.get("severity") == "fail"]
    tolerated = [r["file"] for r in rows if r.get("status") == "different"]
    if broken:
        return Check("files", False, "missing or diverging: " + ", ".join(broken))
    detail = "all required files present"
    if tolerated:
        detail += "; tolerated drift: " + ", ".join(tolerated)
    return Check("files", True, detail)


def compare_vocab(
    draft: Mapping[str, int],
    target: Mapping[str, int],
) -> dict[str, Any]:
    """Compare two (token -> ID) maps (requirements 1 and 3).

    Returns the shared-token count, the pairs whose IDs differ, the tokens
    present on one side only, and the base vocabulary size (one past the highest
    shared ID). Extras falling *inside* the base range are ID shifts, which is
    exactly what requirement 3 forbids.
    """
    shared = [t for t in draft if t in target]
    mismatched = {
        t: {"draft": draft[t], "target": target[t]}
        for t in shared
        if draft[t] != target[t]
    }
    draft_only = sorted((t for t in draft if t not in target), key=lambda t: draft[t])
    target_only = sorted((t for t in target if t not in draft), key=lambda t: target[t])
    base_size = max((draft[t] for t in shared), default=-1) + 1

    return {
        "draft_size": len(draft),
        "target_size": len(target),
        "shared": len(shared),
        "base_size": base_size,
        "id_mismatches": mismatched,
        "draft_only": draft_only,
        "target_only": target_only,
        "draft_only_violations": id_violations(draft, draft_only, base_size),
        "target_only_violations": id_violations(target, target_only, base_size),
    }


def id_violations(
    vocab: Mapping[str, int],
    extras: Sequence[str],
    base_size: int,
) -> dict[str, int]:
    """Extra tokens whose ID falls inside the shared base range (an ID shift)."""
    return {
        token: vocab[token]
        for token in extras
        if vocab.get(token) is not None and vocab[token] < base_size
    }


def vocab_check(result: Mapping[str, Any]) -> Check:
    """Requirement 1 (shared IDs) and requirement 3 (extras beyond the base)."""
    problems: list[str] = []
    mismatches = result["id_mismatches"]
    if mismatches:
        sample = ", ".join(sorted(mismatches)[:5])
        problems.append(f"{len(mismatches)} shifted ID(s) (e.g. {sample})")
    draft_only = result["draft_only"]
    if draft_only:
        sample = ", ".join(map(repr, draft_only[:5]))
        problems.append(f"{len(draft_only)} draft-only token(s) unknown to the target (e.g. {sample})")
    violations = {
        **result["draft_only_violations"],
        **result["target_only_violations"],
    }
    if violations:
        sample = ", ".join(f"{t!r}={i}" for t, i in list(violations.items())[:5])
        problems.append(f"{len(violations)} extra token(s) inside the base range (e.g. {sample})")

    if problems:
        return Check("vocabulary", False, "; ".join(problems))

    detail = f'{result["shared"]} shared tokens, base vocabulary {result["base_size"]}'
    if result["target_only"]:
        detail += f', {len(result["target_only"])} extra target token(s), all beyond the base'
    return Check("vocabulary", True, detail)


def chat_special_tokens(template: str | None) -> list[str]:
    """Special tokens referenced by a chat template, in first-appearance order."""
    if not template:
        return []
    found: list[str] = []
    for token in _SPECIAL_TOKEN_RE.findall(template):
        if token not in found:
            found.append(token)
    return found


def compare_special_tokens(
    names: Sequence[str],
    draft_ids: Mapping[str, int | None],
    target_ids: Mapping[str, int | None],
) -> dict[str, dict[str, int | None]]:
    """Requirement 2: role and chat tokens must have the same ID on both sides.

    Tokens missing from one vocabulary get ``None`` rather than the tokenizer's
    unknown-token ID, so a shared ``unk`` cannot fake a match.
    """
    rows: dict[str, dict[str, int | None]] = {}
    for name in names:
        draft_id, target_id = draft_ids.get(name), target_ids.get(name)
        rows[name] = {
            "draft": draft_id,
            "target": target_id,
            "ok": draft_id is not None and draft_id == target_id,
        }
    return rows


def special_check(rows: Mapping[str, Mapping[str, int | None]]) -> Check:
    """Requirement 2: identical IDs for the roles and the template's tokens."""
    broken = [name for name, row in rows.items() if not row["ok"]]
    if broken:
        details = ", ".join(
            f"{name!r} (draft={rows[name]['draft']}, target={rows[name]['target']})"
            for name in broken
        )
        return Check("special_tokens", False, details)
    return Check("special_tokens", True, f"{len(rows)} special tokens with identical IDs")
