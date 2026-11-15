import hashlib
import json

import pytest

from qwenbench.quantization import (
    calibration_conversation, check_weight_coverage, load_calibration,
    quantized_modules, verify_model_config, verify_nvfp4_layout, verify_preserved_weights, weight_map,
)


def test_recipe_only_selects_supported_text_projections():
    names = [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.visual.blocks.0.mlp.linear_fc1",
        "lm_head",
        "mtp.layers.0.mlp.gate_proj",
    ]
    assert quantized_modules(names, "mixed") == names[:2]
    assert quantized_modules(names, "mlp-only") == names[:1]
    with pytest.raises(ValueError, match="No supported"):
        quantized_modules(names[2:], "mixed")


def test_calibration_keeps_tools_and_reasoning_in_conversations():
    schema = {"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}
    record = {
        "id": "tools", "prompt": "Weather in Paris?", "content": "",
        "tools": [schema], "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}],
    }
    messages, prompts = calibration_conversation(record, "tools")
    assert prompts == [record["prompt"]]
    assert messages[-1]["tool_calls"] == record["tool_calls"]
    with pytest.raises(ValueError, match="real tool schemas"):
        calibration_conversation({**record, "tools": None}, "tools")
    reasoning = {**record, "content": "42", "reasoning": "Compute the result."}
    messages, _ = calibration_conversation(reasoning, "reasoning")
    assert messages[-1]["reasoning_content"] == reasoning["reasoning"]
    with pytest.raises(ValueError, match="reasoning_content"):
        calibration_conversation({**reasoning, "reasoning": ""}, "reasoning")


def _calibration(tmp_path, **overrides):
    target = {"id": "target", "revision": "a" * 40}
    contract = {"models": {"target": target}, "chat_template": {"sha256": "template"}}
    row = {"id": "a", "group": "fr", "input_ids": [1, 2, 3]}
    data = tmp_path / "calibration.jsonl"
    data.write_text(json.dumps(row) + "\n")
    manifest = {
        "status": "completed", "smoke": True, "target": target, "template_sha256": "template",
        "counts": {"fr": 1}, "max_length": 128, "tokens": 3,
        "files": {"calibration.jsonl": hashlib.sha256(data.read_bytes()).hexdigest()}, **overrides,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return contract, data, manifest


def test_calibration_checks_hash_counts_and_smoke_mode(tmp_path):
    contract, data, manifest = _calibration(tmp_path)
    assert len(load_calibration(tmp_path, contract, smoke=True)[1]) == 1
    with pytest.raises(ValueError, match="smoke/production"):
        load_calibration(tmp_path, contract, smoke=False)
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "counts": {"fr": 2}}))
    with pytest.raises(ValueError, match="counts"):
        load_calibration(tmp_path, contract, smoke=True)
    data.write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calibration(tmp_path, contract, smoke=True)


def test_calibration_rejects_invalid_token_ids_even_with_matching_hash(tmp_path):
    contract, data, manifest = _calibration(tmp_path)
    data.write_text(json.dumps({"group": "fr", "input_ids": [True, -1]}) + "\n")
    manifest["files"]["calibration.jsonl"] = hashlib.sha256(data.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid calibration"):
        load_calibration(tmp_path, contract, smoke=True)


def test_export_must_preserve_mtp_and_other_original_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    tensors = {"model.weight": torch.ones(2, dtype=torch.bfloat16),
               "mtp.fc.weight": torch.arange(4, dtype=torch.bfloat16)}
    save_file(tensors, str(source / "model.safetensors"))
    save_file({"model.weight": tensors["model.weight"]}, str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="lost original"):
        verify_preserved_weights(source, output)
    save_file({**tensors, "mtp.fc.weight": tensors["mtp.fc.weight"] + 1}, str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="changed protected"):
        verify_preserved_weights(source, output)
    save_file(tensors, str(output / "model.safetensors"))
    assert verify_preserved_weights(source, output)["mtp"] == ["mtp.fc.weight"]
    check_weight_coverage(weight_map(source), weight_map(output))


def test_normalized_export_config_allows_dtype_metadata_but_not_mtp_or_vision_changes():
    expected = {
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": False,
        "text_config": {"mtp_num_hidden_layers": 1, "hidden_size": 5120},
        "vision_config": {"model_type": "qwen3_5_vision", "hidden_size": 1152, "dtype": None},
    }
    actual = {**expected, "vision_config": {**expected["vision_config"], "dtype": "bfloat16"}}
    verify_model_config(expected, actual)
    with pytest.raises(ValueError, match="mtp_num_hidden_layers"):
        verify_model_config(expected, {**actual, "text_config": {**expected["text_config"], "mtp_num_hidden_layers": 0}})
    with pytest.raises(ValueError, match="vision_config.hidden_size"):
        verify_model_config(expected, {**actual, "vision_config": {**actual["vision_config"], "hidden_size": 1}})


def test_nvfp4_export_requires_packed_shape_and_all_scales(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    name = "model.language_model.layers.0.mlp.gate_proj"
    save_file({name + ".weight": torch.ones(16, 32, dtype=torch.bfloat16)}, str(source / "model.safetensors"))
    tensors = {
        name + ".weight": torch.zeros(16, 16, dtype=torch.uint8),
        name + ".weight_scale": torch.ones(16, 2).to(torch.float8_e4m3fn),
        name + ".weight_scale_2": torch.tensor(1.),
        name + ".input_scale": torch.tensor(1.),
    }
    save_file(tensors, str(output / "model.safetensors"))
    assert verify_nvfp4_layout(source, output, [name]) == 1
    save_file({key: value for key, value in tensors.items() if not key.endswith(".input_scale")},
              str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="Missing NVFP4"):
        verify_nvfp4_layout(source, output, [name])
    save_file({**tensors, name + ".weight": torch.zeros(16, 32, dtype=torch.uint8)},
              str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="dtype/shape"):
        verify_nvfp4_layout(source, output, [name])


def test_builder_is_deterministic_and_refuses_evaluation_contamination(tmp_path, monkeypatch):
    from scripts import build_calib_set

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "chat_template.jinja").write_text("template")
    contract = {"models": {"target": {"id": "target", "revision": "a" * 40}},
                "chat_template": {"sha256": "template"}}
    monkeypatch.setattr(build_calib_set, "pinned_target", lambda path: (contract, snapshot))
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}")
    exclusions = tmp_path / "excluded.jsonl"
    exclusions.write_text(json.dumps({"prompt": "reserved"}) + "\n")
    source = tmp_path / "fr.jsonl"
    records = [{"id": str(i), "text": f"French sample number {i} with enough words"} for i in range(8)]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    base = ["--input", f"fr={source}", "--samples", "fr=2", "--smoke",
            "--exclude-prompts", str(exclusions), "--contract", str(contract_path)]
    for name in ("first", "second"):
        assert build_calib_set.main(base + ["--output-dir", str(tmp_path / name)]) == 0
    assert (tmp_path / "first/calibration.jsonl").read_bytes() == (tmp_path / "second/calibration.jsonl").read_bytes()
    with pytest.raises(SystemExit):
        build_calib_set.main(base + ["--output-dir", str(tmp_path / "first")])
    with source.open("a") as handle:
        handle.write(json.dumps({"id": "eval", "text": "reserved"}) + "\n")
    with pytest.raises(ValueError, match="contamination"):
        build_calib_set.main(base + ["--output-dir", str(tmp_path / "contaminated")])
    assert not (tmp_path / "contaminated").exists()


def test_small_or_incomplete_production_mixtures_are_refused(tmp_path):
    from scripts.build_calib_set import main

    with pytest.raises(SystemExit):
        main(["--input", "fr=unused", "--samples", "fr=1", "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        main(["--input", "fr=unused", "--samples", "fr=1", "code=1", "tools=1", "reasoning=1",
              "--output-dir", str(tmp_path / "out")])
