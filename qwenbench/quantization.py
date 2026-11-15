"""Calibration selection and checkpoint integrity for offline NVFP4 export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

CALIBRATION_COUNTS = {"fr": 256, "code": 128, "tools": 64, "reasoning": 64}
QA_GROUPS = {"fr": "s1-fr", "code": "s2-code"}


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pinned_target(contract_path: Path) -> tuple[dict[str, Any], Path]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    target = contract["models"]["target"]
    if not isinstance(target["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", target["revision"]):
        raise ValueError("The target revision must be a pinned commit")
    snapshot = (Path.home() / ".cache/huggingface/hub"
                / ("models--" + target["id"].replace("/", "--")) / "snapshots" / target["revision"])
    for entry in contract["files"]:
        expected = entry.get("target_sha256")
        if expected and file_hash(snapshot / entry["file"]) != expected:
            raise ValueError(f"Pinned tokenizer file changed: {entry['file']}")
    template = snapshot / "chat_template.jinja"
    if file_hash(template) != contract["chat_template"]["sha256"]:
        raise ValueError("Pinned target chat template changed")
    return contract, snapshot


def calibration_conversation(record: dict[str, Any], group: str) -> tuple[list[dict[str, Any]], list[str]]:
    messages = record.get("messages")
    if messages is None:
        prompt, content = record.get("prompt"), record.get("content")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(content, str):
            raise ValueError(f"Invalid calibration conversation: {record.get('id')}")
        assistant = {"role": "assistant", "content": content}
        if record.get("reasoning"):
            assistant["reasoning_content"] = record["reasoning"]
        if record.get("tool_calls"):
            assistant["tool_calls"] = record["tool_calls"]
        messages = [{"role": "user", "content": prompt}, assistant]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Calibration messages must be a nonempty list")
    prompts = []
    assistants = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported calibration message")
        if not isinstance(message.get("content"), str):
            raise ValueError("Only text calibration messages are supported")
        if message["role"] == "user":
            if not message["content"].strip():
                raise ValueError("Empty calibration user message")
            prompts.append(message["content"])
        if message["role"] == "assistant":
            assistants.append(message)
    if not prompts or not assistants:
        raise ValueError("Calibration conversations require user and assistant messages")
    if group == "tools":
        tools = record.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ValueError("Tools calibration requires real tool schemas, including for no-tool cases")
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ValueError("Invalid calibration tool schema")
            function = tool["function"]
            if not isinstance(function.get("name"), str) or not isinstance(function.get("parameters"), dict):
                raise ValueError("Tool schemas require a name and parameters")
    if group == "reasoning" and not any(
        isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip()
        for message in assistants
    ):
        raise ValueError("Reasoning calibration requires an explicit nonempty reasoning_content")
    return messages, prompts


def weight_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        if not mapping or any(Path(name).name != name for name in mapping.values()):
            raise ValueError("Invalid safetensors index")
        return mapping
    from safetensors import safe_open

    path = directory / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: path.name for key in handle.keys()}


def check_weight_coverage(source: Mapping[str, str], exported: Mapping[str, str]) -> None:
    missing = sorted(source.keys() - exported.keys())
    if missing:
        raise ValueError(f"Export lost original checkpoint tensors: {missing}")


def verify_preserved_weights(source: Path, output: Path) -> dict[str, list[str]]:
    """Check exact MTP, vision, and Gated DeltaNet tensor preservation."""
    from safetensors import safe_open
    import torch

    original, exported = weight_map(source), weight_map(output)
    check_weight_coverage(original, exported)
    protected = {
        "mtp": sorted(key for key in original if key.startswith("mtp.")),
        "vision": sorted(key for key in original if key.startswith("model.visual.")),
        "delta_net": sorted(key for key in original if ".linear_attn." in key),
    }
    for key in (key for names in protected.values() for key in names):
        with safe_open(str(source / original[key]), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        with safe_open(str(output / exported[key]), framework="pt", device="cpu") as handle:
            other = handle.get_tensor(key)
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            raise ValueError(f"Export changed protected unquantized tensor: {key}")
    return protected


def verify_model_config(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Compare normalized architecture semantics, not serialized HF aliases."""
    for key in ("model_type", "architectures", "language_model_only"):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"Export changed target configuration: {key}")
    for section in ("text_config", "vision_config"):
        for key, value in expected[section].items():
            if key in ("dtype", "transformers_version", "_name_or_path"):
                continue
            if actual[section].get(key) != value:
                raise ValueError(f"Export changed target configuration: {section}.{key}")


def verify_nvfp4_layout(source: Path, output: Path, modules: Sequence[str]) -> int:
    from safetensors import safe_open

    original, exported = weight_map(source), weight_map(output)
    for name in modules:
        key = f"{name}.weight"
        with safe_open(str(source / original[key]), framework="pt", device="cpu") as handle:
            rows, columns = handle.get_slice(key).get_shape()
        if columns % 16:
            raise ValueError(f"NVFP4 weights require complete 16-element blocks: {key}")
        expected = {
            "weight": ("U8", [rows, columns // 2]),
            "weight_scale": ("F8_E4M3", [rows, columns // 16]),
            "weight_scale_2": ("F32", []),
            "input_scale": ("F32", []),
        }
        for suffix, (dtype, shape) in expected.items():
            key = f"{name}.{suffix}"
            if key not in exported:
                raise ValueError(f"Missing NVFP4 weight/scale tensor: {key}")
            with safe_open(str(output / exported[key]), framework="pt", device="cpu") as handle:
                tensor = handle.get_slice(key)
                if tensor.get_dtype() != dtype or tensor.get_shape() != shape:
                    raise ValueError(f"Invalid NVFP4 dtype/shape: {key}")
    return len(modules)


def quantized_modules(names: Sequence[str], recipe: str) -> list[str]:
    if recipe not in ("mixed", "mlp-only"):
        raise ValueError(f"Unknown NVFP4 recipe: {recipe}")
    selected = [
        name for name in names
        if name.startswith("model.language_model.layers.")
        and (".mlp." in name or (recipe == "mixed" and ".self_attn." in name))
    ]
    if not selected:
        raise ValueError("No supported text projections matched the NVFP4 recipe")
    return selected


def load_calibration(directory: Path, contract: Mapping[str, Any], *, smoke: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["smoke"] != smoke:
        raise ValueError("Calibration must be completed and match the requested smoke/production mode")
    if manifest["target"] != contract["models"]["target"] or manifest["template_sha256"] != contract["chat_template"]["sha256"]:
        raise ValueError("Calibration target/template does not match the pinned contract")
    if not smoke and set(manifest["counts"]) != set(CALIBRATION_COUNTS):
        raise ValueError("Production calibration requires all four source groups")
    if not smoke and (sum(manifest["counts"].values()) < 512 or manifest["max_length"] < 2048):
        raise ValueError("Production calibration requires at least 512 sequences and a 2048-token cap")
    path = directory / "calibration.jsonl"
    if file_hash(path) != manifest["files"]["calibration.jsonl"]:
        raise ValueError("Calibration token file hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = {group: 0 for group in manifest["counts"]}
    for row in rows:
        ids = row.get("input_ids")
        if (row.get("group") not in counts or not isinstance(ids, list)
                or not 2 <= len(ids) <= manifest["max_length"]
                or any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError("Invalid calibration token sequence/group")
        counts[row["group"]] += 1
    if counts != manifest["counts"] or not rows:
        raise ValueError("Calibration counts do not match the manifest")
    if sum(len(row["input_ids"]) for row in rows) != manifest["tokens"]:
        raise ValueError("Calibration token total does not match the manifest")
    return manifest, rows
