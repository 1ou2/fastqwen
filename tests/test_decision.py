"""Decision policy and HTTP contract tests; no model download or GPU required."""

import json

import httpx
import pytest

from qwenbench.decision import DecisionClient, POLICY_SHA256, QUESTIONS, route_decision


def decision(label="S1", probability=0.94, suitability="suitable"):
    def answer(labels, selected, confidence):
        return {
            "choice": selected, "confidence": confidence,
            "probabilities": {
                key: confidence if key == selected else (1 - confidence) / (len(labels) - 1)
                for key in labels
            },
        }
    return {"answers": {
        "route": answer(QUESTIONS["route"]["criteria"], label, probability),
        "suitability": answer(QUESTIONS["suitability"]["criteria"], suitability, 0.95),
    }, "usage": {"input_tokens": 100, "output_tokens": 0}}


@pytest.mark.parametrize("label", ["S1", "S2"])
def test_confident_suitable_export(label):
    result = route_decision(decision(label))
    assert result["label"] == label
    assert result["export"] is True


@pytest.mark.parametrize("label", ["OTHER", "UNCERTAIN"])
def test_other_routes_not_exported(label):
    assert route_decision(decision(label))["export"] is False


def test_low_confidence_goes_to_review():
    result = route_decision(decision("S2", 0.6))
    assert result["label"] == "UNCERTAIN"
    assert result["export"] is False


@pytest.mark.parametrize("suitability", ["oversized", "incomplete"])
def test_suitability_does_not_change_task_type(suitability):
    result = route_decision(decision("S1", suitability=suitability))
    assert result["label"] == "S1"
    assert result["export"] is False


def test_client_sends_typed_batch():
    def handler(request):
        assert request.url.path == "/classify"
        assert json.loads(request.content) == {
            "prompts": ["Explain clouds", "Implement a WordPress theme"],
            "policy_sha256": POLICY_SHA256,
        }
        return httpx.Response(200, json={
            "policy_sha256": POLICY_SHA256, "results": [decision(), decision("S2")],
        })
    with DecisionClient(transport=httpx.MockTransport(handler)) as client:
        assert len(client.classify(["Explain clouds", "Implement a WordPress theme"])) == 2


def test_wrong_batch_size_rejected():
    with DecisionClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"policy_sha256": POLICY_SHA256, "results": []},
    ))) as client:
        with pytest.raises(ValueError, match="number"):
            client.classify(["prompt"])


@pytest.mark.parametrize("value", [float("nan"), 1.5, True])
def test_malformed_probabilities_rejected(value):
    result = decision()
    result["answers"]["route"]["probabilities"]["S1"] = value
    with pytest.raises(ValueError, match="probability"):
        route_decision(result)


def test_identity_policy_mismatch_rejected():
    with DecisionClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"model": "LiquidAI/d1-2B", "revision": "sha", "policy_sha256": "wrong"},
    ))) as client:
        with pytest.raises(ValueError, match="policy"):
            client.identity()


def test_jsonl_runner_exports_and_audits(tmp_path, monkeypatch):
    from scripts import classify_prompts

    source = tmp_path / "input.jsonl"
    rows = [
        {"id": "general", "slice": "S1", "prompt": "Explain clouds"},
        {"id": "software", "slice": "S1", "prompt": "Write a Python function"},
        {"id": "fragment", "slice": "S1", "prompt": "A navigation menu"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"exclusion": {"prompts": []}}))

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "model": "LiquidAI/d1-2B", "revision": "a" * 40,
                "policy_sha256": POLICY_SHA256, "max_batch_size": 32,
            })
        return httpx.Response(200, json={
            "policy_sha256": POLICY_SHA256,
            "results": [decision(), decision("S2"), decision("UNCERTAIN")],
        })
    monkeypatch.setattr(classify_prompts, "DecisionClient", lambda *args: DecisionClient(
        transport=httpx.MockTransport(handler),
    ))
    output = tmp_path / "output"
    assert classify_prompts.main([
        "--input", str(source), "--output-dir", str(output),
        "--exclusion-manifest", str(exclusions),
    ]) == 0
    assert json.loads((output / "S2.jsonl").read_text())["slice"] == "S2"
    assert len((output / "decisions.jsonl").read_text().splitlines()) == 3
    assert json.loads((output / "manifest.json").read_text())["counts"] == {"S1": 1, "S2": 1, "review": 1}


def test_service_batch_adapter_and_limits(monkeypatch):
    from contextlib import nullcontext
    import sys
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from scripts.serve_decision_model import create_app

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    class Model:
        def system_one_batch(self, requests):
            assert requests == [("Explain clouds", QUESTIONS), ("Debug Python", QUESTIONS)]
            return [decision(), decision("S2")]

    with TestClient(create_app(Model(), {
        "model": "LiquidAI/d1-2B", "revision": "a" * 40, "policy_sha256": POLICY_SHA256,
    }, max_batch_size=2)) as client:
        assert client.get("/health").json()["max_batch_size"] == 2
        response = client.post("/classify", json={
            "prompts": ["Explain clouds", "Debug Python"], "policy_sha256": POLICY_SHA256,
        })
        assert response.status_code == 200
        assert response.json()["results"][1]["answers"]["route"]["choice"] == "S2"
        for prompts, status in [([], 422), ([""], 422), (["a", "b", "c"], 422), (["x" * 32_001], 413)]:
            assert client.post("/classify", json={
                "prompts": prompts, "policy_sha256": POLICY_SHA256,
            }).status_code == status
        assert client.post("/classify", json={
            "prompts": ["prompt"], "policy_sha256": "wrong",
        }).status_code == 409