"""Tokenizer compatibility contract (§3.5) — CLI.

Speculative decoding compares token IDs, so the draft (Qwen3.5-2B) and the
target (Qwen3.8-27B) must agree exactly: one shifted ID silently turns
verification into noise. §3.5 of the plan turns that into five hard
requirements, all enforced here:

1. identical (token -> ID) pairs over the whole shared vocabulary;
2. identical IDs for ``bos``/``eos``/``pad``/``unk`` and the chat tokens of the
   target template;
3. any extra target token sits strictly beyond the base vocabulary;
4. ``apply_chat_template(target template)`` yields identical IDs on both sides;
5. a missing tokenizer file is a failure, never a warning.

Revisions are pinned: either passed explicitly, or resolved from the local HF
cache (``refs/main``). The tokenizer files are then copied out of that pinned
snapshot into ``--cache-dir``, and *those* bytes are what every check runs on —
so the verdict, the hashes in the artifact and the files shipped to the H100
job are the same object. There is no silent fallback to an unpinned download.

Usage:
    python check_tokenizer_compat.py
    python check_tokenizer_compat.py --offline
    python check_tokenizer_compat.py --draft-revision <sha> --target-revision <sha>

Exit codes: 0 contract holds, 2 contract broken, 1 operational error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from qwenbench.tokenizer_contract import (
    REQUIRED_FILES,
    REQUIRED_ROLES,
    WATCHED_FILES,
    Check,
    chat_special_tokens,
    compare_files,
    compare_special_tokens,
    compare_vocab,
    file_check,
    sha256_bytes,
    special_check,
    summarize,
    vocab_check,
)

DRAFT_MODEL = "Qwen/Qwen3.5-2B"
TARGET_MODEL = "Qwen/Qwen3.8-27B"

#: Where the frozen tokenizer copies and the contract artifact are written.
DEFAULT_CACHE_DIR = "cache/tokenizers"
DEFAULT_ARTIFACT = "benchmarks/compat/tokenizer_contract.json"

#: Cap on the samples kept in the JSON artifact (the full report stays on stdout).
SAMPLE_LIMIT = 20

#: Inputs for the functional encoding test: French, quotes, emoji, multilingual,
#: raw chat tokens, and a long run. These are evidence, not the contract.
FUNCTIONAL_TEXTS = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Le speculative decoding accélère l'inférence des LLMs.",
    "Il dit : « je ne crois pas qu'une vitesse de 100 tokens/s soit atteignable » !",
    "a" * 200,
    "🚀 emoji + mixed 中文 العربية text",
    "<|im_start|>system\nYou are helpful.<|im_end|>",
    "def f(x: int) -> int:\n    return x * 2\n",
]

#: Chat templates rendered on both tokenizers (requirement 4). `optional` cases
#: are reported without breaking the contract.
CHAT_CASES: list[tuple[str, list[dict[str, str]], dict[str, Any], bool]] = [
    (
        "single_user",
        [{"role": "user", "content": "What is the capital of Australia?"}],
        {},
        False,
    ),
    (
        "multi_turn",
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Bonjour, qui es-tu ?"},
            {"role": "assistant", "content": "Je suis Qwen."},
            {"role": "user", "content": "Résume en une phrase."},
        ],
        {},
        False,
    ),
    (
        "reasoning_effort_low",
        [{"role": "user", "content": "Combien font 17 × 23 ?"}],
        {"reasoning_effort": "low"},
        False,
    ),
    (
        "tool_schema",
        [{"role": "user", "content": "Quel temps fait-il à Paris ?"}],
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        },
        True,
    ),
]


class OperationalError(RuntimeError):
    """The check could not run at all (missing revision, missing file, load)."""


# --- revisions and frozen files -------------------------------------------


def _repo_cache_dir(model_id: str) -> Path:
    return Path(hf_constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"


def _snapshot_dir(model_id: str, revision: str) -> Path:
    return _repo_cache_dir(model_id) / "snapshots" / revision


def _cached_revision(model_id: str) -> str | None:
    """Commit SHA recorded by the local HF cache, or None if absent/ambiguous."""
    repo_dir = _repo_cache_dir(model_id)
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        return ref.read_text().strip() or None
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        shas = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
        if len(shas) == 1:
            return shas[0]
    return None


def _resolve_revision(model_id: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    cached = _cached_revision(model_id)
    if cached:
        print(f"  [pin] {model_id}: using cached revision {cached[:12]}…")
        return cached
    raise OperationalError(
        f"no pinned revision for {model_id}: pass --draft-revision/--target-revision "
        f"<sha>, or download the model first so the HF cache records one"
    )


def _fetch_files(
    model_id: str,
    revision: str,
    dest: Path,
    offline: bool,
) -> tuple[dict[str, str | None], dict[str, str]]:
    """Freeze the watched tokenizer files into ``dest`` and hash them.

    The source of truth is the pinned snapshot of the local HF cache: copying
    its bytes (rather than trusting whatever ``main`` resolves to) is what makes
    the verdict reproducible and what the H100 job can reuse as-is. Online, a
    file absent from the cache is fetched once to fill it.
    """
    snapshot = _snapshot_dir(model_id, revision)
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str | None] = {}
    errors: dict[str, str] = {}

    for name in WATCHED_FILES:
        source = snapshot / name
        if not source.is_file():
            if offline:
                hashes[name] = None
                errors[name] = f"absent from the pinned snapshot {revision[:12]}… (offline)"
                continue
            try:
                source = Path(hf_hub_download(repo_id=model_id, filename=name, revision=revision))
            except Exception as exc:  # missing file, network, auth…
                hashes[name] = None
                errors[name] = f"{type(exc).__name__}: {exc}"
                continue
        data = source.read_bytes()
        (dest / name).write_bytes(data)
        hashes[name] = sha256_bytes(data)
    return hashes, errors


def _read_config(directory: Path) -> dict[str, Any]:
    path = directory / "tokenizer_config.json"
    if not path.is_file():
        raise OperationalError(f"tokenizer_config.json is unavailable in {directory}")
    return json.loads(path.read_text())


def _template_from_config(config: dict[str, Any]) -> str | None:
    """The target's chat template: a string, or a named list of templates."""
    template = config.get("chat_template")
    if isinstance(template, str):
        return template
    if isinstance(template, list):
        for entry in template:
            if isinstance(entry, dict) and entry.get("name") in (None, "default"):
                return entry.get("template")
        return template[0].get("template") if template else None
    return None


def _as_token(value: Any) -> str | None:
    """A special-token spec from a config, as a plain string (or None)."""
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) else None


def _load_tokenizer(directory: Path) -> tuple[Any, str]:
    """Load the frozen tokenizer files, without ever touching the network.

    ``AutoTokenizer`` needs the model config to pick a class, which a partial
    snapshot may lack; the fallback instantiates the fast tokenizer straight from
    ``tokenizer.json``, which is enough for every check here.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        return tokenizer, "AutoTokenizer"
    except Exception as exc:  # no config.json, unknown class…
        auto_error = f"{type(exc).__name__}: {exc}"

    tokenizer_file = directory / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise OperationalError(f"cannot load a tokenizer from {directory}: {auto_error}")

    config = _read_config(directory)
    extra = [_as_token(token) for token in config.get("additional_special_tokens") or []]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token=_as_token(config.get("bos_token")),
        eos_token=_as_token(config.get("eos_token")),
        unk_token=_as_token(config.get("unk_token")),
        pad_token=_as_token(config.get("pad_token")),
        additional_special_tokens=[token for token in extra if token],
    )
    return tokenizer, f"PreTrainedTokenizerFast(tokenizer.json) [{auto_error[:80]}]"


# --- checks ----------------------------------------------------------------


def _token_id(tokenizer: Any, token: str | None) -> int | None:
    """ID of a special token, or None if that vocabulary does not know it."""
    if not token:
        return None
    return tokenizer.get_vocab().get(token)


def _special_names(draft: Any, target: Any, template: str | None) -> list[str]:
    names: list[str] = []
    for role in REQUIRED_ROLES:
        for tokenizer in (target, draft):
            token = getattr(tokenizer, f"{role}_token", None)
            if token and token not in names:
                names.append(token)
    for token in chat_special_tokens(template):
        if token not in names:
            names.append(token)
    return names


def _render(tokenizer: Any, messages: list[dict[str, str]], extra: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **extra,
    )


def _chat_template_report(
    draft: Any,
    target: Any,
    template: str | None,
) -> tuple[list[dict[str, Any]], Check]:
    """Requirement 4: the target template must render and encode identically."""
    if not template:
        return [], Check("chat_template", None, "the target ships no chat template")
    draft.chat_template = template
    target.chat_template = template

    cases: list[dict[str, Any]] = []
    broken: list[str] = []
    for name, messages, extra, optional in CHAT_CASES:
        row: dict[str, Any] = {"name": name, "optional": optional}
        try:
            text_draft = _render(draft, messages, extra)
            text_target = _render(target, messages, extra)
            ids_draft = draft(text_draft)["input_ids"]
            ids_target = target(text_target)["input_ids"]
        except Exception as exc:  # template error, unknown variable, bad schema…
            row["ok"] = None if optional else False
            row["error"] = f"{type(exc).__name__}: {exc}"
            if not optional:
                broken.append(f"{name}: {row['error']}")
            cases.append(row)
            continue
        same_text = text_draft == text_target
        same_ids = ids_draft == ids_target
        row.update(
            {
                "ok": same_ids and same_text,
                "same_text": same_text,
                "same_ids": same_ids,
                "length": len(ids_target),
                "rendered_sha256": sha256_bytes(text_target.encode()),
            }
        )
        if not (same_ids and same_text):
            broken.append(f"{name}: text={same_text}, ids={same_ids}")
        cases.append(row)

    if broken:
        return cases, Check("chat_template", False, "; ".join(broken))
    skipped = [c["name"] for c in cases if c.get("ok") is None]
    detail = f"{len(cases)} template case(s) render and encode identically"
    if skipped:
        detail += f"; not applicable: {', '.join(skipped)}"
    return cases, Check("chat_template", True, detail)


def _functional_report(draft: Any, target: Any) -> tuple[list[dict[str, Any]], Check, Check]:
    """Free-form encoding plus detokenization, as evidence for the video."""
    rows: list[dict[str, Any]] = []
    mismatched: list[str] = []
    decode_mismatched: list[str] = []
    for text in FUNCTIONAL_TEXTS:
        ids_draft = draft(text)["input_ids"]
        ids_target = target(text)["input_ids"]
        same = ids_draft == ids_target
        rows.append(
            {
                "text": text[:60],
                "ok": same,
                "draft_length": len(ids_draft),
                "target_length": len(ids_target),
            }
        )
        if not same:
            mismatched.append(text[:40])
        if draft.decode(ids_target) != target.decode(ids_target):
            decode_mismatched.append(text[:40])

    if mismatched:
        encode_check = Check("functional_encoding", False, f"differ on: {mismatched}")
    else:
        encode_check = Check("functional_encoding", True, f"{len(rows)} texts encode identically")

    # `clean_up_tokenization_spaces` is explicitly tolerated (§3.5): decode drift
    # is reported, never blocking.
    if decode_mismatched:
        decode_check = Check(
            "detokenization",
            False,
            f"decode drift on: {decode_mismatched}",
            severity="warn",
        )
    else:
        decode_check = Check("detokenization", True, "identical detokenization")
    return rows, encode_check, decode_check


# --- reporting -------------------------------------------------------------


def _print_files(files: list[dict[str, Any]]) -> None:
    print("\n--- File comparison (SHA256) ---")
    for row in files:
        if row["status"] == "identical":
            mark = "✅"
        elif row["status"] in ("missing", "different") and row["severity"] == "fail":
            mark = "❌"
        elif row["status"] == "absent":
            mark = "·"
        else:
            mark = "⚠️"
        print(f"  {mark} {row['file']:28s}  [{row['status']}]")


def _print_report(checks: list[Check]) -> dict[str, Any]:
    print("\n--- Contract ---")
    marks = {True: "✅", False: "❌", None: "·"}
    for check in checks:
        print(f"  {marks[check.ok]} {check.id:20s} {check.detail}")

    verdict = summarize(checks)
    print("\n--- Verdict ---")
    if verdict["ok"]:
        print("  ✅ COMPATIBLE — the draft can be initialized from Qwen3.5-2B")
        print("     and used as a speculative-decoding proposer for the 27B.")
    else:
        print(f"  ❌ INCOMPATIBLE — failed: {', '.join(verdict['failed'])}")
        print("     Options: pick another starting checkpoint for the student,")
        print("     or extend the vocabulary (far more expensive).")
    if verdict["warned"]:
        print(f"  ⚠️  warnings: {', '.join(verdict['warned'])}")
    if verdict["not_applicable"]:
        print(f"  ·  not applicable: {', '.join(verdict['not_applicable'])}")
    return verdict


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenizer compatibility contract (§3.5).")
    parser.add_argument("--draft", default=DRAFT_MODEL)
    parser.add_argument("--target", default=TARGET_MODEL)
    parser.add_argument("--draft-revision", default=None, help="commit SHA, pinned")
    parser.add_argument("--target-revision", default=None, help="commit SHA, pinned")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--json-out", default=DEFAULT_ARTIFACT)
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    args = parser.parse_args(argv)

    started_at = datetime.now(timezone.utc).astimezone()
    cache_dir = Path(args.cache_dir)

    print("=" * 78)
    print("TOKENIZER COMPATIBILITY CONTRACT (§3.5)")
    print(f"  draft  : {args.draft}")
    print(f"  target : {args.target}")
    print("=" * 78)

    try:
        draft_rev = _resolve_revision(args.draft, args.draft_revision)
        target_rev = _resolve_revision(args.target, args.target_revision)
        print(f"  [pin] draft  revision: {draft_rev}")
        print(f"  [pin] target revision: {target_rev}")

        draft_hashes, draft_errors = _fetch_files(
            args.draft, draft_rev, cache_dir / "draft", args.offline
        )
        target_hashes, target_errors = _fetch_files(
            args.target, target_rev, cache_dir / "target", args.offline
        )
        files = compare_files(draft_hashes, target_hashes)
        _print_files(files)

        missing = [name for name in REQUIRED_FILES if draft_hashes.get(name) is None or target_hashes.get(name) is None]
        if missing:
            raise OperationalError(f"required tokenizer file(s) unavailable: {', '.join(missing)}")

        draft, draft_loader = _load_tokenizer(cache_dir / "draft")
        target, target_loader = _load_tokenizer(cache_dir / "target")
        print(f"\n  loader: draft {draft_loader}")
        print(f"  loader: target {target_loader}")

        template = _template_from_config(_read_config(cache_dir / "target"))
    except OperationalError as exc:
        print(f"\n❌ cannot run the check: {exc}", file=sys.stderr)
        return 1

    vocab = compare_vocab(draft.get_vocab(), target.get_vocab())
    names = _special_names(draft, target, template)
    specials = compare_special_tokens(
        names,
        {name: _token_id(draft, name) for name in names},
        {name: _token_id(target, name) for name in names},
    )
    chat_cases, chat_check = _chat_template_report(draft, target, template)
    functional, encode_check, decode_check = _functional_report(draft, target)

    checks = [
        file_check(files),
        vocab_check(vocab),
        special_check(specials),
        chat_check,
        encode_check,
        decode_check,
    ]
    verdict = _print_report(checks)

    print(f"\n  vocabulary: draft {vocab['draft_size']}, target {vocab['target_size']}, "
          f"shared {vocab['shared']}, base {vocab['base_size']}")
    extra = vocab["target_only"]
    if extra:
        shown = ", ".join(f"{token!r}={target.get_vocab()[token]}" for token in extra[:7])
        print(f"  extra target tokens ({len(extra)}): {shown}")
    print(f"  special tokens: {', '.join(names)}")
    print(f"  frozen copies : {cache_dir}/draft, {cache_dir}/target")

    exit_code = 0 if verdict["ok"] else 2

    if not args.no_artifact:
        artifact = {
            "script": "check_tokenizer_compat.py",
            "contract": "project-plan.md §3.5",
            "checked_at": started_at.isoformat(),
            "models": {
                "draft": {"id": args.draft, "revision": draft_rev},
                "target": {"id": args.target, "revision": target_rev},
            },
            "files": files,
            "download_errors": {**draft_errors, **target_errors},
            "loaders": {"draft": draft_loader, "target": target_loader},
            "vocabulary": {
                **{
                    key: value
                    for key, value in vocab.items()
                    if key not in ("id_mismatches", "draft_only", "target_only")
                },
                "id_mismatches": dict(list(vocab["id_mismatches"].items())[:SAMPLE_LIMIT]),
                "id_mismatch_count": len(vocab["id_mismatches"]),
                "draft_only": vocab["draft_only"][:SAMPLE_LIMIT],
                "target_only": vocab["target_only"],
            },
            "special_tokens": specials,
            "chat_template": {
                "sha256": sha256_bytes(template.encode()) if template else None,
                "cases": chat_cases,
            },
            "functional": functional,
            "checks": [check.to_dict() for check in checks],
            "verdict": verdict,
            "exit_code": exit_code,
        }
        out = Path(args.json_out)
        _write_artifact(out, artifact)
        print(f"\n  artifact: {out}")

    print(f"\n  required files  : {', '.join(REQUIRED_FILES)}")
    print(f"  watched files   : {', '.join(WATCHED_FILES)}")
    print("=" * 78)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
