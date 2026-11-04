#!/usr/bin/env python3
"""Checkpointed, single-GPU text-only SFT from an audited QA export."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.training import encode_example, pad_examples, restore_model_weights  # noqa: E402


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_split(path, tokenizer, template, args, split, stats):
    import datasets

    groups = set(args.group)
    kept = 0

    def examples():
        nonlocal kept
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                group = record["qa_group"]
                if group not in groups:
                    continue
                counters = stats[group]
                if split == "train" and args.train_limit is not None and kept >= args.train_limit:
                    continue
                if split == "validation" and counters["records"] >= args.eval_per_group:
                    continue
                encoded = encode_example(tokenizer, template, record, args.max_length)
                if encoded is None:
                    counters["overlength"] += 1
                    continue
                counters["records"] += 1
                counters["input_tokens"] += len(encoded["input_ids"])
                counters["max_input_length"] = max(counters["max_input_length"], len(encoded["input_ids"]))
                counters["label_tokens"] += sum(value != -100 for value in encoded["labels"])
                kept += 1
                yield encoded

    features = datasets.Features({
        key: datasets.Sequence(datasets.Value("int32"))
        for key in ("input_ids", "attention_mask", "labels")
    })
    dataset = datasets.Dataset.from_generator(examples, features=features, cache_dir=str(args.output_dir / "cache"))
    if not len(dataset) or any(not stats[group]["records"] for group in groups):
        raise ValueError(f"No retained {split} examples for one or more requested groups")
    dataset.save_to_disk(str(args.output_dir / split))
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", action="append", choices=("s1-en", "s1-fr", "s2-code"), required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-per-group", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Only single-GPU training is validated by this initial harness")
    positive = (args.max_length, args.eval_per_group, args.batch_size, args.gradient_accumulation,
                args.learning_rate, args.save_steps, args.eval_steps, args.epochs)
    if any(value <= 0 for value in positive) or args.max_length < 2:
        parser.error("Lengths, counts, rates, and epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1 or (args.train_limit is not None and args.train_limit < 1):
        parser.error("Invalid maximum steps or training limit")
    if args.stop_after_steps is not None and (args.stop_after_steps < 1 or args.max_steps < args.stop_after_steps):
        parser.error("--stop-after-steps requires a positive --max-steps at least as large")
    args.qa_dir = args.qa_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.group = sorted(set(args.group))
    qa_path = args.qa_dir / "manifest.json"
    qa = json.loads(qa_path.read_text())
    if qa["status"] != "completed":
        parser.error("QA export must have completed status")
    data_files = {name: args.qa_dir / f"{name}.jsonl" for name in ("train", "validation")}
    for name, path in data_files.items():
        if file_hash(path) != qa["files"][f"{name}.jsonl"]["sha256"]:
            raise ValueError(f"QA data hash mismatch: {path}")
    contract_path = ROOT / "benchmarks/compat/tokenizer_contract.json"
    contract = json.loads(contract_path.read_text())
    teacher = qa["cohort"]["teacher"]
    if teacher["model"] != contract["models"]["target"]["id"] or teacher["revision"] != contract["models"]["target"]["revision"]:
        raise ValueError("QA teacher does not match the tokenizer contract")
    if contract["vocabulary"]["id_mismatch_count"] or not all(case["ok"] for case in contract["chat_template"]["cases"]):
        raise ValueError("Tokenizer contract has failed compatibility checks")
    hub = Path.home() / ".cache/huggingface/hub"
    locations = {
        role: hub / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
        for role, model in contract["models"].items()
    }
    for item in contract["files"]:
        for role in locations:
            expected = item.get(f"{role}_sha256")
            if expected and file_hash(locations[role] / item["file"]) != expected:
                raise ValueError(f"Pinned {role} tokenizer file changed: {item['file']}")
    template_path = locations["target"] / "chat_template.jinja"
    template = template_path.read_text()
    if hashlib.sha256(template.encode()).hexdigest() != teacher["template_sha256"] or teacher["template_sha256"] != contract["chat_template"]["sha256"]:
        raise ValueError("Teacher chat template hash mismatch")
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key not in ("prepare_only", "resume_from", "stop_after_steps")
    }
    configuration.update(qa_manifest_sha256=file_hash(qa_path), tokenizer_contract_sha256=file_hash(contract_path))
    source_files = {name: file_hash(ROOT / name) for name in (
        "scripts/train_draft.py", "qwenbench/training.py", "qwenbench/qa.py",
    )}
    configuration["source_files"] = source_files
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output_dir / "training-manifest.json"
    existing = args.output_dir.exists()
    if existing:
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration_sha256"] != identity:
            raise ValueError("Existing run configuration differs; use a fresh output directory")
        retry_preflight = manifest["status"] == "failed" and not manifest.get("last_launch_at")
        if args.prepare_only or (args.resume_from is None and manifest["status"] != "prepared" and not retry_preflight):
            parser.error("Existing training run requires an explicit checkpoint resume")
        for name, digest in manifest["tokenized_files"].items():
            if file_hash(args.output_dir / name) != digest:
                raise ValueError(f"Tokenized data changed: {name}")
    else:
        if args.resume_from is not None:
            parser.error("Resume requires the original output directory and manifest")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in source_files:
            destination = args.output_dir / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        manifest = {
            "status": "preparing", "started_at": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration, "configuration_sha256": identity,
            "student": contract["models"]["draft"], "teacher": teacher,
            "notes": ["Single GPU; full text-decoder SFT, no tools, vision inputs, packing, or target truncation.",
                      "Requested source groups keep natural row proportions; no balancing or resampling.",
                      "Evaluation retains up to eval_per_group examples per requested group in source order.",
                      "A training limit is for a single-group smoke, not a representative mixed-source sample."],
        }
        write_manifest(manifest_path, manifest)
    try:
        import datasets
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(locations["draft"], local_files_only=True)
        tokenizer.chat_template = template
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not existing:
            stats = {name: defaultdict(lambda: defaultdict(int)) for name in data_files}
            for name, path in data_files.items():
                prepare_split(path, tokenizer, template, args, name, stats[name])
            manifest["data_stats"] = {name: {group: dict(counts) for group, counts in values.items()} for name, values in stats.items()}
            manifest["tokenized_files"] = {
                str(path.relative_to(args.output_dir)): file_hash(path)
                for name in data_files for path in sorted((args.output_dir / name).rglob("*")) if path.is_file()
            }
            manifest["status"] = "prepared"
            write_manifest(manifest_path, manifest)
        if args.prepare_only:
            print(json.dumps({"status": "prepared", "data_stats": manifest["data_stats"]}, indent=2))
            return 0

        import torch
        import accelerate

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one available GPU")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 CUDA training is required")
        manifest["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                "accelerate": accelerate.__version__, "datasets": datasets.__version__, "cuda": torch.version.cuda}
        manifest["gpu"] = torch.cuda.get_device_name(0)
        transformers.set_seed(args.seed)
        train_data = datasets.load_from_disk(str(args.output_dir / "train"))
        eval_data = datasets.load_from_disk(str(args.output_dir / "validation"))
        planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(
            args.epochs * math.ceil(len(train_data) / (args.batch_size * args.gradient_accumulation))
        )
        training_args = transformers.TrainingArguments(
            output_dir=str(args.output_dir), max_steps=args.max_steps, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
            warmup_steps=math.ceil(planned_steps * .03), lr_scheduler_type="cosine", weight_decay=.01, max_grad_norm=1.0,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch", logging_steps=1, eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=3, save_only_model=False,
            prediction_loss_only=True, report_to="none", seed=args.seed, data_seed=args.seed,
            dataloader_num_workers=0, dataloader_pin_memory=True,
        )
        model, loading = transformers.AutoModelForCausalLM.from_pretrained(
            locations["draft"], local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Student text weights did not fully load: {loading}")
        manifest["loading_info"] = {
            key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()
        }
        model.config.use_cache = False
        model.to("cuda")

        def collator(examples):
            return {key: torch.tensor(values, dtype=torch.long) for key, values in pad_examples(examples, tokenizer.pad_token_id).items()}

        class StopAfter(transformers.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= args_stop:
                    control.should_training_stop = True
                    control.should_save = True
                return control

        class FiniteMetrics(transformers.TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ("loss", "grad_norm", "eval_loss"):
                    if logs and key in logs and not math.isfinite(float(logs[key])):
                        raise ValueError(f"Non-finite training metric at step {state.global_step}: {key}")
                return control

        args_stop = args.stop_after_steps
        callbacks = [FiniteMetrics()] + ([StopAfter()] if args_stop is not None else [])

        class TextTrainer(transformers.Trainer):
            def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
                restored, info = transformers.AutoModelForCausalLM.from_pretrained(
                    resume_from_checkpoint, local_files_only=True, dtype=torch.bfloat16,
                    attn_implementation="sdpa", output_loading_info=True,
                )
                restore_model_weights(model if model is not None else self.model, restored, info)
                manifest["resume_weights_verified"] = True
                print("Checkpoint weights restored with strict key matching", flush=True)

        trainer = TextTrainer(
            model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data,
            data_collator=collator, processing_class=tokenizer, callbacks=callbacks,
        )
        resume = None
        if args.resume_from is not None:
            resume_path = args.resume_from.resolve()
            if resume_path.parent != args.output_dir:
                raise ValueError("Resume checkpoint must belong to this output directory")
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
                if not (resume_path / name).is_file():
                    raise ValueError(f"Incomplete resume checkpoint: {name}")
            resume = str(resume_path)
        manifest["status"] = "training"
        manifest.pop("error", None)
        manifest["last_launch_at"] = datetime.now(timezone.utc).isoformat()
        manifest["resume_from"] = resume
        write_manifest(manifest_path, manifest)
        result = trainer.train(resume_from_checkpoint=resume)
        manifest["train_metrics"] = result.metrics
        manifest["global_step"] = trainer.state.global_step
        manifest["eval_metrics"] = trainer.evaluate()
        trainer.save_state()
        if args_stop is not None and trainer.state.global_step < args.max_steps:
            manifest["status"] = "stopped"
        else:
            trainer.save_model(str(args.output_dir / "final"))
            tokenizer.save_pretrained(args.output_dir / "final")
            manifest["status"] = "completed"
        print(json.dumps({"status": manifest["status"], "global_step": trainer.state.global_step,
                          "eval_metrics": manifest["eval_metrics"]}, indent=2))
        return 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    sys.exit(main())