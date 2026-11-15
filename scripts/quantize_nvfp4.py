#!/usr/bin/env python3
"""Calibrate and export the pinned full Qwen target with a mixed NVFP4 recipe."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.quantization import (
    file_hash, load_calibration, pinned_target, quantized_modules, verify_model_config,
    verify_nvfp4_layout, verify_preserved_weights, weight_map, write_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", choices=("mixed", "mlp-only"), default="mixed")
    parser.add_argument("--contract", type=Path, default=ROOT / "benchmarks/compat/tokenizer_contract.json")
    parser.add_argument("--smoke", action="store_true", help="Require smoke calibration; export is not a quality-qualified checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Check pinned inputs without loading the model or writing outputs")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This initial exporter supports a single visible GPU, not distributed calibration")
    contract, snapshot = pinned_target(args.contract)
    output = args.output_dir.resolve()
    if output.is_relative_to(snapshot.resolve()) or output.is_relative_to(args.calibration_dir.resolve()):
        parser.error("Do not write exports into the pinned checkpoint or frozen calibration directory")
    calibration, rows = load_calibration(args.calibration_dir, contract, smoke=args.smoke)
    if calibration["contract_sha256"] != file_hash(args.contract):
        raise ValueError("Calibration tokenizer contract changed")
    original = weight_map(snapshot)
    source_config = json.loads((snapshot / "config.json").read_text())
    if source_config.get("model_type") != "qwen3_5" or source_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("This exporter requires the pinned full Qwen3.5-compatible target architecture")
    vocabulary = source_config["text_config"]["vocab_size"]
    if any(token >= vocabulary for row in rows for token in row["input_ids"]):
        raise ValueError("Calibration contains token IDs outside the target vocabulary")
    mtp_names = sorted(key for key in original if key.startswith("mtp."))
    if source_config["text_config"].get("mtp_num_hidden_layers", 0) and not mtp_names:
        raise ValueError("Target declares MTP layers but the checkpoint has no MTP weights")
    preflight = {
        "target": contract["models"]["target"], "source": str(snapshot),
        "recipe": args.recipe, "calibration_counts": calibration["counts"],
        "calibration_tokens": calibration["tokens"], "mtp_tensors": len(mtp_names),
        "smoke": args.smoke, "output": str(output),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return 0

    import torch
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoConfig, AutoModelForImageTextToText

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one free CUDA GPU with CUDA_VISIBLE_DEVICES")
    required = sum((snapshot / name).stat().st_size for name in set(original.values())) + 8 * 1024**3
    free, _ = torch.cuda.mem_get_info()
    if free < required:
        raise ValueError(f"GPU has {free} free bytes; full BF16 loading needs at least {required} including reserve")
    existing_parent = next(parent for parent in output.parents if parent.exists())
    if shutil.disk_usage(existing_parent).free < required:
        raise ValueError(f"Export needs at least {required} free disk bytes including temporary-file reserve")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    started = time.monotonic()
    manifest = {
        "script": "scripts/quantize_nvfp4.py", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), **preflight,
        "calibration_manifest_sha256": file_hash(args.calibration_dir / "manifest.json"),
        "calibration_sha256": calibration["files"]["calibration.jsonl"],
        "contract_sha256": file_hash(args.contract),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "nvidia-modelopt", "safetensors", "accelerate")},
        "command": sys.argv, "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "notes": [
            "Single-GPU offline PTQ; native Blackwell/vLLM inference remains a Spark gate.",
            "Vision, Gated DeltaNet projections, LM head, and native MTP remain unquantized.",
            "MTP is carried by the official ModelOpt exporter and checked tensor-for-tensor.",
            "Smoke exports are implementation checks, not production calibration or quality results.",
        ],
    }
    write_manifest(manifest_path, manifest)
    try:
        source_files = sorted(set(original.values()) | {"config.json", "model.safetensors.index.json", "chat_template.jinja"})
        manifest["source_files"] = {name: file_hash(snapshot / name) for name in source_files}
        sources = ["scripts/quantize_nvfp4.py", "scripts/build_calib_set.py", "qwenbench/quantization.py",
                   "requirements-quantization.txt"]
        manifest["implementation"] = {name: file_hash(ROOT / name) for name in sources}
        for name in sources:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        model, loading = AutoModelForImageTextToText.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}, output_loading_info=True,
        )
        manifest["loading_info"] = {
            key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()
        }
        if (any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs"))
                or set(loading.get("unexpected_keys", [])) - set(mtp_names)):
            raise ValueError(f"Target did not fully load apart from known checkpoint-only MTP: {loading}")
        if set(model.state_dict()) != set(original) - set(mtp_names):
            raise ValueError("Loaded target tensor names differ from the original non-MTP checkpoint")
        model.eval()
        selected = quantized_modules(
            [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)], args.recipe,
        )
        cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
        cfg["quant_cfg"].append({"quantizer_name": "*", "enable": False})
        for name in selected:
            for suffix in ("input_quantizer", "weight_quantizer"):
                cfg["quant_cfg"].append({"quantizer_name": f"{name}.{suffix}", "enable": True})
        manifest["quantized_modules"] = selected
        manifest["quantization_config"] = cfg
        write_manifest(manifest_path, manifest)

        def forward_loop(active_model):
            with torch.inference_mode():
                for index, row in enumerate(rows, 1):
                    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
                    result = active_model(
                        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1,
                    )
                    if not torch.isfinite(result.logits).all():
                        raise ValueError(f"Non-finite calibration logits: {row['id']}")
                    print(f"Calibration {index}/{len(rows)} group={row['group']} tokens={ids.shape[1]}", flush=True)

        model = mtq.quantize(model, cfg, forward_loop)
        for name in selected:
            module = model.get_submodule(name)
            for suffix in ("input_quantizer", "weight_quantizer"):
                quantizer = getattr(module, suffix)
                if not quantizer.is_enabled or quantizer.amax is None or not torch.isfinite(quantizer.amax).all():
                    raise ValueError(f"Uncalibrated NVFP4 quantizer: {name}.{suffix}")
        checkpoint = output / "checkpoint"
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=checkpoint, dtype=torch.bfloat16)
        for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                         "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
                         "video_preprocessor_config.json", "LICENSE"):
            source = snapshot / filename
            if source.exists():
                shutil.copyfile(source, checkpoint / filename)
        manifest["preserved_weights"] = verify_preserved_weights(snapshot, checkpoint)
        verify_model_config(
            AutoConfig.from_pretrained(snapshot, local_files_only=True).to_dict(),
            AutoConfig.from_pretrained(checkpoint, local_files_only=True).to_dict(),
        )
        quant_config = json.loads((checkpoint / "hf_quant_config.json").read_text())
        if quant_config["quantization"]["quant_algo"] != "NVFP4":
            raise ValueError("Exporter did not produce an NVFP4 checkpoint")
        manifest["verified_nvfp4_modules"] = verify_nvfp4_layout(snapshot, checkpoint, selected)
        for name, expected in manifest["source_files"].items():
            if file_hash(snapshot / name) != expected:
                raise ValueError(f"Original checkpoint changed during export: {name}")
        if file_hash(args.calibration_dir / "calibration.jsonl") != manifest["calibration_sha256"]:
            raise ValueError("Calibration changed during export")
        manifest["files"] = {str(path.relative_to(output)): file_hash(path)
                             for path in sorted(checkpoint.iterdir()) if path.is_file()}
        manifest["export_bytes"] = sum(path.stat().st_size for path in checkpoint.iterdir() if path.is_file())
        manifest["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["runtime_seconds"] = time.monotonic() - started
        write_manifest(manifest_path, manifest)
    print(f"Completed NVFP4 export: {output / 'checkpoint'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
