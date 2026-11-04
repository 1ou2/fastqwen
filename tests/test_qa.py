import pytest

from qwenbench.eval_sets import prompt_sha256
from qwenbench.qa import assess, validation_member


def _record(**overrides):
    return {
        "id": "question-1", "slice": "S1", "prompt": "Explain seasons.",
        "content": "The Earth's axial tilt changes sunlight during the year.",
        "finish_reason": "stop", "reasoning": "", "tools": None,
        "tool_calls": [], "spec": {"thinking": False}, **overrides,
    }


def test_normal_answer_is_accepted():
    assert assess(_record()) == ("accepted", [])


@pytest.mark.parametrize("overrides, reason", [
    ({"finish_reason": "length"}, "capped_answer"),
    ({"content": " \n"}, "empty_answer"),
    ({"content": None}, "invalid_content"),
    ({"finish_reason": "content_filter"}, "unsuccessful_finish"),
    ({"prompt": ""}, "invalid_id_or_prompt"),
])
def test_unusable_records_are_rejected(overrides, reason):
    assert assess(_record(**overrides)) == ("rejected", [reason])


@pytest.mark.parametrize("overrides, reason", [
    ({"content": "name " * 100}, "repeated_word_sequence"),
    ({"content": "x" * 80}, "long_character_run"),
    ({"content": ("This exact long line keeps appearing here.\n" * 10)}, "repeated_line"),
    ({"reasoning": "private reasoning"}, "reasoning_record"),
    ({"content": "<think>reasoning</think>answer"}, "template_or_reasoning_marker"),
    ({"content": "Example:\n```python\nprint(1)\n```"}, "s1_code_boundary"),
])
def test_suspicious_stopped_answers_are_quarantined(overrides, reason):
    destination, reasons = assess(_record(**overrides))
    assert destination == "review"
    assert reason in reasons


def test_code_is_not_a_boundary_flag_in_s2():
    assert assess(_record(slice="S2", content="```python\nprint(1)\n```")) == ("accepted", [])


def test_tool_only_answer_is_reviewed_not_rejected_as_empty():
    assert assess(_record(content="", tool_calls=[{"function": {"name": "lookup"}}])) == (
        "review", ["unsupported_tool_record"],
    )


def test_repeated_list_labels_are_not_automatically_rejected():
    destination, reasons = assess(_record(content="Ingredients for this recipe section:\n" * 10))
    assert destination == "review"
    assert "repeated_line" in reasons


def test_validation_membership_is_normalized_and_reproducible():
    digest = prompt_sha256(" Explain   seasons ")
    assert digest == prompt_sha256("explain seasons")
    assert validation_member(digest, 42, 0) is False
    assert validation_member(digest, 42, 1) is True
    assert validation_member(digest, 42, .02) == validation_member(digest, 42, .02)
    members = [validation_member(prompt_sha256(f"prompt {index}"), 42, .02) for index in range(1000)]
    assert 5 < sum(members) < 40


def _export_fixture(tmp_path, rows):
    import hashlib
    import json

    teacher = {"model": "teacher", "revision": "revision", "image_digest": "image", "template_sha256": "template"}
    spec = {"thinking": False}
    shard = tmp_path / "shard.jsonl"
    records = [
        {**_record(), **row, "teacher": teacher, "template_sha256": "template", "spec": spec,
         "prompt_sha256": prompt_sha256(row.get("prompt", "Explain seasons.")),
         "usage": {"completion_tokens": 10}}
        for row in rows
    ]
    shard.write_text("".join(json.dumps(row) + "\n" for row in records))
    manifest = {
        "teacher": teacher, "spec": spec, "inputs": {"prompts": []},
        "failures": {"count": 0}, "totals": {"records": len(records), "completion_tokens": len(records) * 10},
        "output": {"path": str(shard), "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(), "bytes": shard.stat().st_size},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"exclusion": {"prompts": [{"sha256": prompt_sha256("frozen")} ]}}))
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"prompt": "reserved"}) + "\n")
    output = tmp_path / "export"
    args = ["--input", f"s1-en={shard}", "--output-dir", str(output),
            "--exclusion-manifest", str(exclusions), "--heldout-prompts", str(heldout)]
    return args, output, records


def test_export_cli_audits_all_dispositions_and_preserves_targets(tmp_path):
    import hashlib
    import json

    from scripts.export_training_data import main

    rows = [
        {"id": "good", "prompt": "normal", "content": "Exact target  \n"},
        {"id": "duplicate", "prompt": " NORMAL "},
        {"id": "eval", "prompt": "frozen"},
        {"id": "heldout", "prompt": "reserved"},
        {"id": "capped", "prompt": "long", "finish_reason": "length"},
        {"id": "repetitive", "prompt": "loop", "content": "name " * 100},
        {"id": "empty", "prompt": "empty", "content": ""},
        {"id": "tool", "prompt": "tool", "content": "", "tool_calls": [{"name": "lookup"}]},
    ]
    args, output, records = _export_fixture(tmp_path, rows)
    assert main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["counts"]["review"] == 2
    assert manifest["counts"]["rejected"] == 5
    accepted = [json.loads(line) for name in ("train", "validation") for line in (output / f"{name}.jsonl").read_text().splitlines()]
    assert len(accepted) == 1
    assert accepted[0]["content"] == records[0]["content"]
    assert accepted[0]["qa_group"] == "s1-en"
    assert len((output / "decisions.jsonl").read_text().splitlines()) == len(records)
    for filename, metadata in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == metadata["sha256"]
    with pytest.raises(SystemExit):
        main(args)


def test_export_rejects_modified_input_before_creating_output(tmp_path):
    from scripts.export_training_data import main

    args, output, _ = _export_fixture(tmp_path, [{"id": "good", "prompt": "normal"}])
    with (tmp_path / "shard.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        main(args)
    assert not output.exists()


def test_export_records_failed_status_for_record_provenance_mismatch(tmp_path):
    import hashlib
    import json

    from scripts.export_training_data import main

    args, output, records = _export_fixture(tmp_path, [{"id": "good", "prompt": "normal"}])
    records[0]["teacher"]["revision"] = "different"
    shard = tmp_path / "shard.jsonl"
    shard.write_text(json.dumps(records[0]) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output"].update(sha256=hashlib.sha256(shard.read_bytes()).hexdigest(), bytes=shard.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="teacher differs"):
        main(args)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_s1_cpp_fence_is_reviewed():
    assert assess(_record(content="```c++\nint main() {}\n```")) == ("review", ["s1_code_boundary"])


@pytest.mark.parametrize("problem", ["truncated", "malformed", "record_count", "token_count"])
def test_export_fails_closed_on_damaged_or_inconsistent_shards(tmp_path, problem):
    import hashlib
    import json

    from scripts.export_training_data import main

    args, output, _ = _export_fixture(tmp_path, [{"id": "good", "prompt": "normal"}])
    shard = tmp_path / "shard.jsonl"
    if problem == "truncated":
        shard.write_bytes(shard.read_bytes().rstrip(b"\n"))
    elif problem == "malformed":
        shard.write_text("{broken}\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output"].update(sha256=hashlib.sha256(shard.read_bytes()).hexdigest(), bytes=shard.stat().st_size)
    if problem == "record_count":
        manifest["totals"]["records"] += 1
    elif problem == "token_count":
        manifest["totals"]["completion_tokens"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        main(args)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_export_split_is_disjoint_and_independent_of_input_order(tmp_path):
    import json

    from scripts.export_training_data import main

    rows = [{"id": f"id-{index}", "prompt": f"Unique question {index}"} for index in range(300)]
    assignments = []
    for name, ordered_rows in (("forward", rows), ("reverse", list(reversed(rows)))):
        directory = tmp_path / name
        directory.mkdir()
        args, output, _ = _export_fixture(directory, ordered_rows)
        assert main(args) == 0
        splits = {split: {json.loads(line)["prompt_sha256"] for line in (output / f"{split}.jsonl").read_text().splitlines()}
                  for split in ("train", "validation")}
        assert splits["validation"]
        assert not splits["train"] & splits["validation"]
        assert len(splits["train"] | splits["validation"]) == len(rows)
        assignments.append(splits)
    assert assignments[0] == assignments[1]


def test_export_rejects_duplicate_ids_and_wrong_prompt_hashes(tmp_path):
    import hashlib
    import json

    from scripts.export_training_data import main

    args, output, records = _export_fixture(tmp_path, [
        {"id": "same-id", "prompt": "first"},
        {"id": "same-id", "prompt": "second"},
        {"id": "wrong-hash", "prompt": "third"},
    ])
    records[2]["prompt_sha256"] = prompt_sha256("different")
    shard = tmp_path / "shard.jsonl"
    shard.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output"].update(sha256=hashlib.sha256(shard.read_bytes()).hexdigest(), bytes=shard.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    assert main(args) == 0
    assert json.loads((output / "manifest.json").read_text())["reason_counts"] == {
        "duplicate_id_or_prompt": 1, "prompt_hash_mismatch": 1,
    }