#!/usr/bin/env python3
"""Freeze, evaluate, and summarize a paired offline greedy draft comparison."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.acceptance import aggregate_replays, paired_acceptance_interval, replay_greedy
from qwenbench.eval_sets import prompt_sha256
from qwenbench.training import encode_example

GROUPS = ("s1-en", "s1-fr", "s2-code")
SOURCES = ("scripts/compare_drafts.py", "qwenbench/acceptance.py", "qwenbench/training.py",
           "qwenbench/qa.py", "qwenbench/eval_sets.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, values):
    with Path(path).open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def check_files(base, files):
    for name, expected in files.items():
        if digest(Path(base) / name) != expected:
            raise ValueError(f"File hash mismatch: {name}")


def prepare(args):
    import datasets
    import httpx
    from transformers import AutoTokenizer

    run = args.run_dir.resolve()
    if run.exists():
        raise ValueError("Choose a fresh comparison directory")
    training_dir = args.training_dir.resolve()
    training = read_json(training_dir / "training-manifest.json")
    if training["status"] != "completed":
        raise ValueError("Trained pilot must have completed")
    qa_dir = Path(training["configuration"]["qa_dir"])
    if digest(qa_dir / "manifest.json") != training["configuration"]["qa_manifest_sha256"]:
        raise ValueError("QA manifest differs from training provenance")
    qa = read_json(qa_dir / "manifest.json")
    for name in ("train", "validation"):
        if digest(qa_dir / f"{name}.jsonl") != qa["files"][f"{name}.jsonl"]["sha256"]:
            raise ValueError(f"QA {name} file changed")
    contract_path = ROOT / "benchmarks/compat/tokenizer_contract.json"
    if digest(contract_path) != training["configuration"]["tokenizer_contract_sha256"]:
        raise ValueError("Tokenizer contract changed since training")
    contract = read_json(contract_path)
    if contract["vocabulary"]["id_mismatch_count"] or not all(case["ok"] for case in contract["chat_template"]["cases"]):
        raise ValueError("Tokenizer compatibility contract failed")
    hub = Path.home() / ".cache/huggingface/hub"
    snapshots = {role: hub / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
                 for role, model in contract["models"].items()}
    for item in contract["files"]:
        for role in snapshots:
            expected = item.get(f"{role}_sha256")
            if expected and digest(snapshots[role] / item["file"]) != expected:
                raise ValueError(f"Tokenizer file changed: {role}/{item['file']}")
    template = (snapshots["target"] / "chat_template.jinja").read_text()
    if hashlib.sha256(template.encode()).hexdigest() != training["teacher"]["template_sha256"]:
        raise ValueError("Teacher template changed")
    tokenizer = AutoTokenizer.from_pretrained(snapshots["draft"], local_files_only=True)
    trained_tokenizer = AutoTokenizer.from_pretrained(training_dir / "final", local_files_only=True)
    if tokenizer.get_vocab() != trained_tokenizer.get_vocab():
        raise ValueError("Raw and trained tokenizer token IDs differ")
    if digest(training_dir / "final/chat_template.jinja") != hashlib.sha256(template.encode()).hexdigest():
        raise ValueError("Trained template differs from teacher")
    check_files(training_dir, training["tokenized_files"])
    cached_validation = datasets.load_from_disk(str(training_dir / "validation"))
    validation_rows = rows(qa_dir / "validation.jsonl")
    monitored = []
    counts = Counter()
    for record in validation_rows:
        group = record["qa_group"]
        if counts[group] >= training["data_stats"]["validation"][group]["records"]:
            continue
        encoded = encode_example(tokenizer, template, record, training["configuration"]["max_length"])
        if encoded is None:
            continue
        expected = cached_validation[len(monitored)]
        if any(encoded[key] != expected[key] for key in encoded):
            raise ValueError("Reconstructed monitored validation does not match cached training validation")
        monitored.append({"id": record["id"], "group": group, "prompt_sha256": record["prompt_sha256"], **encoded})
        counts[group] += 1
    if len(monitored) != len(cached_validation):
        raise ValueError("Could not reconstruct all monitored validation rows")
    monitored_hashes = {record["prompt_sha256"] for record in monitored}
    training_hashes = set()
    with (qa_dir / "train.jsonl").open() as handle:
        for line in handle:
            training_hashes.add(json.loads(line)["prompt_sha256"])
    candidate_sources = [(ROOT / "benchmarks/validation/acceptance.jsonl", None)]
    for path, group in ((ROOT / "data/runs/cfp-prep-20261103/heldout.jsonl", "s1-fr"),
                        (ROOT / "data/runs/future-code-prep-20261103/heldout.jsonl", "s2-code")):
        expected = next(item["sha256"] for item in qa["heldout_inputs"] if Path(item["path"]).resolve() == path.resolve())
        if digest(path) != expected:
            raise ValueError(f"Source hold-out changed: {path}")
        candidate_sources.append((path, group))
    candidates = [(record, record["qa_group"], str(qa_dir / "validation.jsonl"))
                  for record in validation_rows if record["qa_group"] == "s1-en"]
    for path, group in candidate_sources:
        for record in rows(path):
            assigned = group
            if assigned is None:
                if record["slice"] == "S1" and record.get("language") in ("en", "fr"):
                    assigned = "s1-" + record["language"]
                elif record["slice"] == "S2":
                    assigned = "s2-code"
                else:
                    continue
            candidates.append((record, assigned, str(path)))
    eligible = defaultdict(list)
    seen = set()
    skips = Counter()
    for record, group, source in candidates:
        key = prompt_sha256(record["prompt"])
        if record.get("tools") or key in training_hashes or key in monitored_hashes or key in seen:
            skips["tools_training_monitored_or_duplicate"] += 1
            continue
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}], chat_template=template,
            tokenize=True, add_generation_prompt=True, enable_thinking=False, reasoning_effort="low",
        )["input_ids"]
        if len(prefix) + args.max_new_tokens > args.max_context:
            skips["over_context"] += 1
            continue
        seen.add(key)
        eligible[group].append({"id": record["id"], "group": group, "prompt": record["prompt"],
                                "prompt_sha256": key, "source_file": source, "input_ids": prefix})
    selected = []
    for group in GROUPS:
        ordered = sorted(eligible[group], key=lambda record: hashlib.sha256(f"{args.seed}:{record['prompt_sha256']}".encode()).hexdigest())
        if len(ordered) < args.per_group:
            raise ValueError(f"Insufficient eligible {group} prompts: {len(ordered)}")
        selected.extend(ordered[:args.per_group])
    if len({item["id"] for item in selected}) != len(selected):
        raise ValueError("Selected prompt IDs are not unique")
    command = subprocess.run(
        ["docker", "inspect", "--format", '{"image":{{json .Config.Image}},"args":{{json .Args}}}', args.teacher_container],
        check=True, capture_output=True, text=True,
    )
    deployment = json.loads(command.stdout)
    if training["teacher"]["image_digest"] not in deployment["image"] or training["teacher"]["revision"] not in " ".join(deployment["args"]):
        raise ValueError("Live teacher deployment does not match training provenance")
    run.mkdir(parents=True, exist_ok=False)
    manifest_path = run / "manifest.json"
    manifest = {
        "stage": "prepare", "status": "running", "started_at": now(),
        "configuration": {"per_group": args.per_group, "max_new_tokens": args.max_new_tokens,
                          "max_context": args.max_context, "gammas": args.gammas, "seed": args.seed,
                          "diagnostic_per_group": args.diagnostic_per_group},
        "teacher": training["teacher"], "teacher_url": args.teacher_url,
        "deployment": deployment, "training_manifest_sha256": digest(training_dir / "training-manifest.json"),
        "qa_manifest_sha256": digest(qa_dir / "manifest.json"), "contract_sha256": digest(contract_path),
        "candidate_files": {str(path): digest(path) for path, _ in candidate_sources},
        "eligible_counts": {group: len(eligible[group]) for group in GROUPS}, "skips": dict(skips),
        "source_files": {name: digest(ROOT / name) for name in SOURCES},
        "models": {"raw": str(snapshots["draft"]), "trained": str(training_dir / "final")},
        "notes": ["Offline greedy trace replay, not native speculative serving or a throughput benchmark.",
                  "Diagnostic validation was monitored during SFT; acceptance excludes all monitored prompts and QA training prompts.",
                  "Acceptance English supplements frozen prompts with unmonitored QA validation; FR/code use frozen plus preparation hold-outs.",
                  "Length-capped traces measure finite-horizon acceptance; stop/EOS and capped counts must be reported.",
                  "Independent reference regeneration is a smoke determinism check; replay identity is by construction, not a native-engine exactness gate."],
    }
    write_json(manifest_path, manifest)
    try:
        for name in SOURCES:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        manifest["model_files"] = {
            arm: {str(path): digest(path) for pattern in ("*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
                  for path in sorted(Path(location).glob(pattern))}
            for arm, location in manifest["models"].items()
        }
        diagnostic = [item for group in GROUPS for item in [record for record in monitored if record["group"] == group][:args.diagnostic_per_group]]
        write_rows(run / "diagnostic.jsonl", diagnostic)
        write_rows(run / "prompts.jsonl", selected)
        with httpx.Client(timeout=900) as client, (run / "references.jsonl").open("x", encoding="utf-8") as handle:
            models_response = client.get(args.teacher_url.rstrip("/") + "/v1/models")
            models_response.raise_for_status()
            manifest["teacher_models_response"] = models_response.json()
            if manifest["teacher"]["model"] not in [item["id"] for item in models_response.json()["data"]]:
                raise ValueError("Expected teacher is not served")
            fingerprints = set()
            for index, record in enumerate(selected):
                body = {"model": manifest["teacher"]["model"], "prompt": record["input_ids"],
                        "temperature": 0, "top_p": 1, "seed": 0, "max_tokens": args.max_new_tokens,
                        "return_token_ids": True, "skip_special_tokens": False,
                        "stop_token_ids": [tokenizer.eos_token_id]}
                response = client.post(args.teacher_url.rstrip("/") + "/v1/completions", json=body)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                tokens = choice.get("token_ids")
                if choice.get("prompt_token_ids") != record["input_ids"] or not tokens or any(type(token) is not int for token in tokens):
                    raise ValueError("Teacher did not return exact prompt and completion token IDs")
                if payload["usage"]["completion_tokens"] != len(tokens) or len(tokens) > args.max_new_tokens:
                    raise ValueError("Teacher completion token count mismatch")
                if choice["finish_reason"] not in ("stop", "length"):
                    raise ValueError("Unsuccessful teacher trace")
                if choice["finish_reason"] == "stop" and tokens[-1] != tokenizer.eos_token_id:
                    raise ValueError("Teacher stop is not the expected terminal EOS")
                if tokenizer.eos_token_id in tokens[:-1]:
                    raise ValueError("Teacher trace continues after EOS")
                if index == 0:
                    repeated = client.post(args.teacher_url.rstrip("/") + "/v1/completions", json=body)
                    repeated.raise_for_status()
                    if repeated.json()["choices"][0].get("token_ids") != tokens:
                        raise ValueError("Independent greedy reference regeneration differs")
                    manifest["reference_regeneration_verified"] = True
                fingerprints.add(payload.get("system_fingerprint"))
                if len(fingerprints) != 1:
                    raise ValueError("Teacher fingerprint changed during trace collection")
                handle.write(json.dumps({**record, "output_ids": tokens, "finish_reason": choice["finish_reason"],
                                         "teacher_response": payload}, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"Teacher trace {index + 1}/{len(selected)} {record['group']} tokens={len(tokens)} finish={choice['finish_reason']}", flush=True)
            manifest["teacher_system_fingerprint"] = next(iter(fingerprints))
        manifest["files"] = {name: digest(run / name) for name in ("diagnostic.jsonl", "prompts.jsonl", "references.jsonl")}
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = now()
        write_json(manifest_path, manifest)


def evaluate(args):
    import torch
    import torch.nn.functional as functional
    import transformers

    run = args.run_dir.resolve()
    parent = read_json(run / "manifest.json")
    if parent["stage"] != "prepare" or parent["status"] != "completed":
        raise ValueError("Frozen trace preparation must have completed")
    check_files(ROOT, parent["source_files"])
    check_files(run, parent["files"])
    check_files(Path("/"), parent["model_files"][args.arm])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one free CUDA GPU for each evaluation arm")
    output = run / args.arm
    output.mkdir(exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = {"status": "running", "arm": args.arm, "started_at": now(),
                "parent_sha256": digest(run / "manifest.json"), "gpu": torch.cuda.get_device_name(0),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}}
    write_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        transformers.set_seed(parent["configuration"]["seed"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(parent["models"]["raw"], local_files_only=True)
        model, loading = transformers.AutoModelForCausalLM.from_pretrained(
            parent["models"][args.arm], local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Model did not load completely: {loading}")
        model.eval().to("cuda")
        generation = transformers.GenerationConfig(
            do_sample=False, use_cache=True, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        )

        def score(ids, labels):
            inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0, :-1]
                targets = torch.tensor(labels[1:], device="cuda", dtype=torch.long)
                mask = targets != -100
                target_logits = logits[mask].float()
                targets = targets[mask]
                loss_sum = functional.cross_entropy(target_logits, targets, reduction="sum").item()
                matches = (target_logits.argmax(-1) == targets).sum().item()
            if not len(targets) or not math.isfinite(loss_sum):
                raise ValueError("Invalid target count or non-finite loss")
            return {"nll_sum": loss_sum, "label_tokens": len(targets), "target_top1_matches": matches,
                    "mean_nll": loss_sum / len(targets)}

        def propose(prefix, budget):
            inputs = torch.tensor([prefix], device="cuda", dtype=torch.long)
            with torch.inference_mode():
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           generation_config=generation, max_new_tokens=budget)
            return generated[0, len(prefix):].tolist()

        with (output / "diagnostic.jsonl").open("x", encoding="utf-8") as handle:
            for record in rows(run / "diagnostic.jsonl"):
                result = {key: record[key] for key in ("id", "group", "prompt_sha256")}
                result.update(score(record["input_ids"], record["labels"]))
                handle.write(json.dumps(result) + "\n")
                handle.flush()
        print(f"{args.arm}: monitored-validation loss evaluation complete", flush=True)
        with (output / "agreement.jsonl").open("x", encoding="utf-8") as agreement, (output / "acceptance.jsonl").open("x", encoding="utf-8") as acceptance:
            references = rows(run / "references.jsonl")
            for index, record in enumerate(references):
                metadata = {key: record[key] for key in ("id", "group", "prompt_sha256")}
                prefix, reference = record["input_ids"], record["output_ids"]
                agreement.write(json.dumps({**metadata, **score(prefix + reference, [-100] * len(prefix) + reference)}) + "\n")
                agreement.flush()
                for gamma in parent["configuration"]["gammas"]:
                    result = replay_greedy(prefix, reference, propose, gamma)
                    if result["output_ids"] != reference:
                        raise ValueError("Replayed output differs from teacher trace")
                    acceptance.write(json.dumps({**metadata, "gamma": gamma, **result}) + "\n")
                    acceptance.flush()
                print(f"{args.arm}: acceptance {index + 1}/{len(references)} {record['group']}", flush=True)
        manifest["files"] = {name: digest(output / name) for name in ("diagnostic.jsonl", "agreement.jsonl", "acceptance.jsonl")}
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["finished_at"] = now()
        write_json(manifest_path, manifest)


def score_summary(values):
    tokens = sum(row["label_tokens"] for row in values)
    return {"examples": len(values), "label_tokens": tokens,
            "token_weighted_nll": sum(row["nll_sum"] for row in values) / tokens,
            "example_weighted_nll": sum(row["mean_nll"] for row in values) / len(values),
            "target_top1_accuracy": sum(row["target_top1_matches"] for row in values) / tokens}


def summarize(args):
    run = args.run_dir.resolve()
    parent = read_json(run / "manifest.json")
    check_files(run, parent["files"])
    arms = {}
    for arm in ("raw", "trained"):
        manifest = read_json(run / arm / "manifest.json")
        if manifest["status"] != "completed" or manifest["parent_sha256"] != digest(run / "manifest.json"):
            raise ValueError(f"Incomplete or incompatible {arm} results")
        check_files(run / arm, manifest["files"])
        arms[arm] = {name: rows(run / arm / f"{name}.jsonl") for name in ("diagnostic", "agreement", "acceptance")}
    references = rows(run / "references.jsonl")
    reference_ids = {row["id"]: row for row in references}
    summary = {"status": "completed", "finished_at": now(), "parent_sha256": digest(run / "manifest.json"),
               "method": "offline greedy trace replay; not native engine exactness or throughput",
               "reference_finish_reasons": dict(Counter(row["finish_reason"] for row in references)),
               "diagnostic": {}, "teacher_trace_agreement": {}, "acceptance": {}}
    for category, destination in (("diagnostic", "diagnostic"), ("agreement", "teacher_trace_agreement")):
        for group in (*GROUPS, "all"):
            selected = {arm: [row for row in data[category] if group == "all" or row["group"] == group] for arm, data in arms.items()}
            raw_ids = {row["id"]: row["label_tokens"] for row in selected["raw"]}
            if raw_ids != {row["id"]: row["label_tokens"] for row in selected["trained"]}:
                raise ValueError("Scored examples are not paired")
            summary[destination][group] = {arm: score_summary(values) for arm, values in selected.items()}
    for gamma in parent["configuration"]["gammas"]:
        indexed = {}
        for arm, data in arms.items():
            selected = [row for row in data["acceptance"] if row["gamma"] == gamma]
            indexed[arm] = {row["id"]: row for row in selected}
            if len(indexed[arm]) != len(selected) or set(indexed[arm]) != set(reference_ids):
                raise ValueError("Missing or duplicate acceptance prompts")
            for row in selected:
                if row["output_ids"] != reference_ids[row["id"]]["output_ids"]:
                    raise ValueError("Output-token identity failure")
        groups = {}
        for group in (*GROUPS, "all"):
            pairs = [(indexed["raw"][key], indexed["trained"][key]) for key, record in reference_ids.items() if group == "all" or record["group"] == group]
            groups[group] = {"raw": aggregate_replays([raw for raw, _ in pairs]),
                             "trained": aggregate_replays([trained for _, trained in pairs]),
                             "paired_difference": paired_acceptance_interval(pairs, parent["configuration"]["seed"], args.bootstrap_samples)}
        groups["source_balanced_mean_acceptance"] = {arm: sum(groups[group][arm]["acceptance_rate"] for group in GROUPS) / len(GROUPS) for arm in ("raw", "trained")}
        summary["acceptance"][str(gamma)] = groups
    if (run / "summary.json").exists():
        raise ValueError("Summary already exists; refusing to overwrite")
    write_json(run / "summary.json", summary)
    print(json.dumps({"status": "completed", "acceptance": {gamma: values["all"] for gamma, values in summary["acceptance"].items()}}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    preparation = subparsers.add_parser("prepare")
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--training-dir", type=Path, default=ROOT / "data/runs/sft-pilot-20261104")
    preparation.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    preparation.add_argument("--teacher-container", default="fr-code-smoke-20261103-214906-teacher-0")
    preparation.add_argument("--per-group", type=int, default=100)
    preparation.add_argument("--diagnostic-per-group", type=int, default=64)
    preparation.add_argument("--max-new-tokens", type=int, default=128)
    preparation.add_argument("--max-context", type=int, default=2048)
    preparation.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4, 8])
    preparation.add_argument("--seed", type=int, default=20261105)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--arm", choices=("raw", "trained"), required=True)
    reporting = subparsers.add_parser("summarize")
    reporting.add_argument("--run-dir", type=Path, required=True)
    reporting.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        if min(args.per_group, args.diagnostic_per_group, args.max_new_tokens, *args.gammas) < 1 or args.diagnostic_per_group > 64:
            parser.error("Positive counts required; diagnostic-per-group cannot exceed 64")
        if args.max_context <= args.max_new_tokens or len(set(args.gammas)) != len(args.gammas):
            parser.error("Invalid context budget or duplicate gammas")
        prepare(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:
        if args.bootstrap_samples < 2:
            parser.error("At least two bootstrap replicates required")
        summarize(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())