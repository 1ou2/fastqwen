import pytest

from scripts.prepare_generation_prompts import SOURCES, _adapt, _select
from scripts.sample_corpora import _s1_filter_reason


def test_s1_filter_rejects_extreme_output_request():
    assert _s1_filter_reason("Write a 50,000-word romance novel.") == "s1_extreme_output"


def test_s1_filter_rejects_software_build_task():
    prompt = "Design and develop an online booking application for classes."

    assert _s1_filter_reason(prompt) == "s1_software_task"


def test_s1_filter_rejects_site_game_and_language_specific_code_tasks():
    prompts = [
        "Build a responsive social networking site with messaging.",
        "Develop a 2D mobile game using Swift and SpriteKit.",
        "Create a Go program that converts decimal to hexadecimal.",
    ]

    assert all(_s1_filter_reason(prompt) == "s1_software_task" for prompt in prompts)


def test_s1_filter_keeps_bounded_general_prompt():
    assert _s1_filter_reason("Explain why leaves change color in autumn.") is None


def test_s1_filter_keeps_non_software_program_prompt():
    prompt = "Develop an onboarding program for new employees."

    assert _s1_filter_reason(prompt) is None


def test_s1_filter_ignores_comma_before_the_word_words():
    assert _s1_filter_reason("Describe colors, words, and punctuation in a poem.") is None


def _cfp_row(prompt, dataset="OpenAssistant/oasst1"):
    return {"inputs": prompt, "targets": "DO NOT EXPORT", "dataset": dataset, "task": "open_qa", "subject": "science"}


def test_cfp_adapter_preserves_complete_prompt_and_provenance():
    row = _adapt("cfp", _cfp_row("Context; question\nwith a second line"), "revision", 7)

    assert row["prompt"] == "Context; question\nwith a second line"
    assert row["slice"] == "S1"
    assert row["language"] == "fr"
    assert row["source_row"] == 7
    assert "targets" not in row
    assert "DO NOT EXPORT" not in str(row)


def test_future_code_adapter_uses_english_instruction_only():
    source = {"id": "en-123", "input": "Write SQL", "output": "DO NOT EXPORT", "description_language": "en", "task_language": "sql"}
    row = _adapt("future-code", source, "revision", 2)

    assert row["prompt"] == "Write SQL"
    assert row["slice"] == "S2"
    assert row["source_id"] == "en-123"
    assert row["task_language"] == "sql"
    assert "output" not in row
    assert _adapt("future-code", {**source, "description_language": "ja"}, "revision", 2) is None
    source.pop("task_language")
    assert _adapt("future-code", source, "revision", 2)["task_language"] is None


def test_preparation_refuses_schema_drift():
    with pytest.raises(ValueError, match="missing columns"):
        _adapt("cfp", {"prompt": "wrong field"}, "revision", 0)


def test_preparation_filters_exclusions_duplicates_sources_and_lengths():
    from qwenbench.eval_sets import prompt_sha256

    rows = [
        _cfp_row("held out"), _cfp_row("one"), _cfp_row(" ONE "),
        _cfp_row("benchmark", "facebook/belebele"), _cfp_row("x" * 31), _cfp_row("two"),
    ]
    args = (rows, "cfp", SOURCES["cfp"]["revision"], 0, 1, 0, 30, 42, {prompt_sha256("held out")})
    prompts, heldout, stats = _select(*args)

    assert len(prompts) == len(heldout) == 1
    assert {row["prompt"] for row in prompts + heldout} == {"one", "two"}
    assert stats["excluded_prompt"] == stats["duplicate"] == stats["benchmark_source"] == stats["over_character_limit"] == 1
    assert _select(*args) == (prompts, heldout, stats)


def test_preparation_reservoir_and_scan_limit_are_deterministic():
    rows = [_cfp_row(f"prompt {index}") for index in range(40)]
    args = (rows, "cfp", "revision", 5, 2, 20, 100, 42, set())
    prompts, heldout, stats = _select(*args)

    assert len(prompts) == 5
    assert len(heldout) == 2
    assert stats["scanned"] == 20
    assert not {row["id"] for row in prompts} & {row["id"] for row in heldout}
    assert _select(*args) == (prompts, heldout, stats)


def test_preparation_fails_if_scan_budget_cannot_fill_request():
    with pytest.raises(ValueError, match="Only 1 eligible"):
        _select([_cfp_row("one")], "cfp", "revision", 2, 0, 0, 100, 42, set())


def test_preparation_cli_writes_audited_shards_and_refuses_overwrite(tmp_path, monkeypatch):
    import hashlib
    import json

    from scripts import prepare_generation_prompts as preparation
    from qwenbench.eval_sets import prompt_sha256

    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"exclusion": {"prompts": [{"sha256": prompt_sha256("frozen")}]}}))
    previous = tmp_path / "previous.jsonl"
    previous.write_text(json.dumps({"prompt": "previous"}) + "\n")
    card = tmp_path / "README.md"
    card.write_text("Source terms\n")
    source_rows = [_cfp_row(prompt) for prompt in ("frozen", "previous", "one", "two", "three")]
    monkeypatch.setattr(preparation, "_source_rows", lambda args, revision: (iter(source_rows), {"fixture": True}, card))
    output = tmp_path / "prepared"
    args = [
        "--source", "cfp", "--out-dir", str(output), "--holdout", "1",
        "--exclusion-manifest", str(exclusions), "--exclude-prompts", str(previous),
    ]

    assert preparation.main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["generation_records"] == 2
    assert manifest["heldout_records"] == 1
    assert manifest["stats"]["excluded_prompt"] == 2
    assert manifest["source"]["revision"] == SOURCES["cfp"]["revision"]
    for filename, metadata in manifest["files"].items():
        contents = (output / filename).read_bytes()
        assert metadata["sha256"] == hashlib.sha256(contents).hexdigest()
        assert metadata["bytes"] == len(contents)
    assert "DO NOT EXPORT" not in (output / "prompts.jsonl").read_text()
    with pytest.raises(SystemExit) as error:
        preparation.main(args)
    assert error.value.code == 2