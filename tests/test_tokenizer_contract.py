"""Offline tests for the tokenizer contract (§3.5).

No network, no GPU, no transformers: only the pure comparison logic. Each test
names the requirement it protects, so a regression points straight back at §3.5.
"""

from __future__ import annotations

from qwenbench import tokenizer_contract as tc

BASE_VOCAB = {"<|endoftext|>": 0, "hello": 1, " world": 2, "bonjour": 3}
AUDIO_TOKENS = {"<|audio_0|>": 4, "<|audio_1|>": 5}


# --- requirement 1: shared tokens keep their IDs ---------------------------


def test_identical_vocabularies_pass_requirement_1():
    result = tc.compare_vocab(BASE_VOCAB, dict(BASE_VOCAB))

    assert tc.vocab_check(result).ok is True
    assert result["shared"] == len(BASE_VOCAB)
    assert result["base_size"] == 4
    assert result["id_mismatches"] == {}


def test_one_shifted_id_breaks_the_contract_requirement_1():
    target = dict(BASE_VOCAB)
    target["bonjour"] = 99

    result = tc.compare_vocab(BASE_VOCAB, target)
    check = tc.vocab_check(result)

    assert check.ok is False
    assert result["id_mismatches"] == {"bonjour": {"draft": 3, "target": 99}}
    assert "1 shifted ID" in check.detail


def test_draft_only_token_breaks_the_contract_requirement_1():
    draft = {**BASE_VOCAB, "inventé": 4}

    check = tc.vocab_check(tc.compare_vocab(draft, dict(BASE_VOCAB)))

    assert check.ok is False
    assert "draft-only" in check.detail


# --- requirement 3: extras sit strictly beyond the base vocabulary ----------


def test_extra_tokens_beyond_the_base_are_accepted_requirement_3():
    target = {**BASE_VOCAB, **AUDIO_TOKENS}

    result = tc.compare_vocab(BASE_VOCAB, target)
    check = tc.vocab_check(result)

    assert check.ok is True
    assert result["base_size"] == 4
    assert result["target_only"] == list(AUDIO_TOKENS)
    assert result["target_only_violations"] == {}


def test_extra_token_inside_the_base_range_is_a_shift_requirement_3():
    # An audio token inserted at ID 2 would shift "world" and "bonjour".
    target = {"<|endoftext|>": 0, "hello": 1, "<|audio_0|>": 2, " world": 3, "bonjour": 4}

    result = tc.compare_vocab(BASE_VOCAB, target)
    check = tc.vocab_check(result)

    assert check.ok is False
    assert result["target_only_violations"] == {"<|audio_0|>": 2}
    assert "inside the base range" in check.detail


# --- requirement 5: a missing file is a failure ----------------------------


def _files(**overrides: str | None) -> dict[str, str | None]:
    """A complete, identical file-hash set, with per-file overrides.

    Overrides use underscores (``tokenizer_json``) because Python keywords cannot
    contain dots; they are mapped back to the real filenames.
    """
    hashes: dict[str, str | None] = {name: f"sha-{name}" for name in tc.WATCHED_FILES}
    for key, value in overrides.items():
        hashes[key.replace("_", ".")] = value
    return hashes


def test_identical_files_pass_requirement_5():
    rows = tc.compare_files(_files(), _files())

    assert tc.file_check(rows).ok is True
    assert all(row["status"] == "identical" for row in rows)


def test_missing_required_file_is_a_failure_requirement_5():
    rows = tc.compare_files(_files(tokenizer_json=None), _files())

    check = tc.file_check(rows)
    assert check.ok is False
    assert "tokenizer.json" in check.detail


def test_file_absent_on_both_sides_is_not_a_failure_requirement_5():
    rows = tc.compare_files(
        _files(merges_txt=None, vocab_json=None),
        _files(merges_txt=None, vocab_json=None),
    )

    assert tc.file_check(rows).ok is True
    absent = [row for row in rows if row["status"] == "absent"]
    assert len(absent) == 2


def test_base_vocabulary_drift_is_a_failure_requirement_5():
    rows = tc.compare_files(_files(vocab_json="sha-different"), _files())

    check = tc.file_check(rows)
    assert check.ok is False
    assert "vocab.json" in check.detail


def test_tokenizer_json_drift_is_tolerated():
    """The target's extra audio tokens live in tokenizer.json (§3.4)."""
    rows = tc.compare_files(_files(), _files(tokenizer_json="sha-different"))

    check = tc.file_check(rows)
    assert check.ok is True
    assert "tolerated drift" in check.detail


# --- requirement 2: roles and chat tokens have the same IDs ----------------


def test_chat_special_tokens_are_extracted_in_order():
    template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n<|im_end|>{% endif %}"
    )

    assert tc.chat_special_tokens(template) == ["<|im_start|>", "<|im_end|>"]
    assert tc.chat_special_tokens("no special token here") == []
    assert tc.chat_special_tokens(None) == []


def test_matching_special_ids_pass_requirement_2():
    rows = tc.compare_special_tokens(
        ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 0},
        {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 0},
    )

    assert tc.special_check(rows).ok is True


def test_special_token_unknown_to_one_side_fails_requirement_2():
    # `None` (absent), not the unknown-token ID: a shared <unk> must not pass.
    rows = tc.compare_special_tokens(
        ["<|im_start|>"],
        {"<|im_start|>": 151644},
        {"<|im_start|>": None},
    )

    check = tc.special_check(rows)
    assert check.ok is False
    assert "<|im_start|>" in check.detail


# --- verdict aggregation ---------------------------------------------------


def test_a_failure_dominates_warnings_and_not_applicable():
    checks = [
        tc.Check("a", True, "ok"),
        tc.Check("b", False, "warning only", severity="warn"),
        tc.Check("c", None, "not testable"),
    ]

    verdict = tc.summarize(checks)
    assert verdict["ok"] is True
    assert verdict["warned"] == ["b"]
    assert verdict["not_applicable"] == ["c"]

    verdict = tc.summarize([*checks, tc.Check("d", False, "broken")])
    assert verdict["ok"] is False
    assert verdict["failed"] == ["d"]


def test_sha256_bytes_is_stable():
    assert tc.sha256_bytes(b"abc") == tc.sha256_bytes(b"abc")
    assert len(tc.sha256_bytes(b"abc")) == 64
    assert tc.sha256_bytes(b"abc") != tc.sha256_bytes(b"abd")
