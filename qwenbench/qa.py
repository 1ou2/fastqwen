"""Deterministic, conservative QA for text-only teacher targets."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

POLICY = {
    "version": 1,
    "scope": "text-only S1/S2, no tools or reasoning",
    "character_run_minimum": 80,
    "repeated_line_minimum_chars": 30,
    "repeated_line_minimum_count": 10,
    "word_loop_maximum_words": 12,
    "word_loop_minimum_repeats": 20,
    "word_loop_tail_chars": 6000,
    "s1_code_fence_languages": [
        "python", "javascript", "typescript", "java", "bash", "sh", "sql",
        "cpp", "c++", "php", "ruby", "perl", "html", "css", "rust", "go",
    ],
    "template_markers": ["<think>", "</think>", "<|im_start|>", "<|im_end|>"],
}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()
_CHARACTER_RUN = re.compile(r"(\S)\1{79,}")
_WORD_LOOP = re.compile(r"(\b(?:\S+\s+){1,12})(?:\1){19,}")
_CODE_FENCE = re.compile(
    r"```(?:" + "|".join(re.escape(language) for language in POLICY["s1_code_fence_languages"]) + r")(?=\s|$)",
    re.IGNORECASE,
)


def assess(record: dict) -> tuple[str, list[str]]:
    reasons = []
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in ("id", "prompt")):
        return "rejected", ["invalid_id_or_prompt"]
    content = record.get("content")
    if not isinstance(content, str):
        return "rejected", ["invalid_content"]
    if record.get("finish_reason") == "length":
        return "rejected", ["capped_answer"]
    if record.get("finish_reason") not in ("stop", "tool_calls"):
        return "rejected", ["unsuccessful_finish"]
    if record.get("tools") or record.get("tool_calls") or record.get("finish_reason") == "tool_calls":
        return "review", ["unsupported_tool_record"]
    if not content.strip():
        return "rejected", ["empty_answer"]
    if record.get("slice") not in ("S1", "S2"):
        reasons.append("unsupported_slice")
    if (record.get("reasoning") or "").strip() or (record.get("spec") or {}).get("thinking"):
        reasons.append("reasoning_record")
    if any(marker in content for marker in POLICY["template_markers"]):
        reasons.append("template_or_reasoning_marker")
    if _CHARACTER_RUN.search(content):
        reasons.append("long_character_run")
    lines = Counter(line.strip() for line in content.splitlines() if len(line.strip()) >= 30)
    if max(lines.values(), default=0) >= 10:
        reasons.append("repeated_line")
    if _WORD_LOOP.search(content[-6000:]):
        reasons.append("repeated_word_sequence")
    if record.get("slice") == "S1" and _CODE_FENCE.search(content):
        reasons.append("s1_code_boundary")
    return ("review", reasons) if reasons else ("accepted", [])


def validation_member(prompt_sha256: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{prompt_sha256}".encode()).digest()
    return int.from_bytes(digest, "big") < int(fraction * (1 << 256))