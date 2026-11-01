#!/usr/bin/env python3
"""Serve LiquidAI/d1-2B's typed batch API on a dedicated local GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import re
import sys
import threading
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.decision import MODEL_ID, POLICY_SHA256, QUESTIONS


def create_app(model: Any, identity: dict[str, Any], max_batch_size: int = 32) -> Any:
    from fastapi import FastAPI, HTTPException
    import torch

    app = FastAPI(title="fastqwen decision model")
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {**identity, "max_batch_size": max_batch_size}

    @app.post("/classify")
    def classify(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("policy_sha256") != POLICY_SHA256:
            raise HTTPException(409, "Decision policy mismatch")
        prompts = body.get("prompts")
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= max_batch_size:
            raise HTTPException(422, f"Expected 1 to {max_batch_size} prompts")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise HTTPException(422, "Prompts must be nonempty strings")
        if any(len(prompt) > 32_000 for prompt in prompts) or sum(map(len, prompts)) > 128_000:
            raise HTTPException(413, "Prompt exceeds the service character budget; no truncation applied")
        with lock, torch.inference_mode():
            results = model.system_one_batch([(prompt, QUESTIONS) for prompt in prompts])
        return {"policy_sha256": POLICY_SHA256, "results": list(results)}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="reviewed 40-character Hugging Face commit SHA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit SHA, not main or a tag")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")

    import torch
    import uvicorn
    from transformers import AutoModel

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=args.revision, code_revision=args.revision,
        trust_remote_code=True, dtype=dtype,
    ).to(args.device).eval()
    identity = {
        "model": MODEL_ID, "revision": args.revision, "policy_sha256": POLICY_SHA256,
        "device": args.device, "dtype": str(dtype),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "fastapi", "uvicorn")
        },
    }
    uvicorn.run(create_app(model, identity, args.max_batch_size), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())