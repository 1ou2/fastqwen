"""Offline greedy draft acceptance against a frozen teacher token trace."""

from collections.abc import Callable, Sequence
import random


def replay_greedy(
    prompt: Sequence[int],
    reference: Sequence[int],
    propose: Callable[[list[int], int], list[int]],
    gamma: int,
) -> dict:
    if gamma < 1 or not reference:
        raise ValueError("A positive gamma and nonempty teacher trace are required")
    emitted = []
    accepted = 0
    proposed = 0
    corrections = 0
    bonuses = 0
    prefixes = []
    while len(emitted) < len(reference):
        start = len(emitted)
        budget = min(gamma, len(reference) - start)
        block = propose(list(prompt) + emitted, budget)
        if not block or len(block) > budget or any(type(token) is not int for token in block):
            raise ValueError("Draft must return one to budget integer token IDs")
        proposed += len(block)
        matched = 0
        for token in block:
            if token != reference[len(emitted)]:
                break
            emitted.append(token)
            matched += 1
        accepted += matched
        prefixes.append(matched)
        if len(emitted) < len(reference):
            emitted.append(reference[len(emitted)])
            if matched < len(block):
                corrections += 1
            else:
                bonuses += 1
    if emitted != list(reference):
        raise AssertionError("Greedy replay must reconstruct the frozen teacher trace")
    return {
        "proposed_tokens": proposed, "accepted_tokens": accepted,
        "correction_tokens": corrections, "bonus_tokens": bonuses,
        "verification_cycles": len(prefixes), "accepted_prefix_lengths": prefixes,
        "output_tokens": len(emitted), "output_ids": emitted,
        "acceptance_rate": accepted / proposed,
        "mean_accepted_prefix": accepted / len(prefixes),
        "tokens_per_verification": len(emitted) / len(prefixes),
    }


def aggregate_replays(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize empty acceptance results")
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("proposed_tokens", "accepted_tokens", "correction_tokens", "bonus_tokens",
                    "verification_cycles", "output_tokens")
    }
    if totals["proposed_tokens"] < 1 or totals["verification_cycles"] < 1:
        raise ValueError("Invalid acceptance denominators")
    totals.update(
        prompts=len(rows), acceptance_rate=totals["accepted_tokens"] / totals["proposed_tokens"],
        mean_accepted_prefix=totals["accepted_tokens"] / totals["verification_cycles"],
        tokens_per_verification=totals["output_tokens"] / totals["verification_cycles"],
    )
    return totals


def paired_acceptance_interval(pairs: Sequence[tuple[dict, dict]], seed: int, samples: int = 2000) -> dict:
    if not pairs or samples < 2:
        raise ValueError("Paired rows and at least two bootstrap replicates are required")
    strata = {}
    for raw, trained in pairs:
        if (raw["id"], raw["group"], raw["gamma"]) != (trained["id"], trained["group"], trained["gamma"]):
            raise ValueError("Acceptance rows are not paired by prompt, group, and gamma")
        strata.setdefault(raw["group"], []).append((raw, trained))
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        sampled = [rng.choice(rows) for rows in strata.values() for _ in range(len(rows))]
        raw_rate = sum(raw["accepted_tokens"] for raw, _ in sampled) / sum(raw["proposed_tokens"] for raw, _ in sampled)
        trained_rate = sum(trained["accepted_tokens"] for _, trained in sampled) / sum(trained["proposed_tokens"] for _, trained in sampled)
        differences.append(trained_rate - raw_rate)
    differences.sort()
    raw_rate = aggregate_replays([raw for raw, _ in pairs])["acceptance_rate"]
    trained_rate = aggregate_replays([trained for _, trained in pairs])["acceptance_rate"]
    return {
        "trained_minus_raw": trained_rate - raw_rate,
        "percentile_95": [differences[int(.025 * (samples - 1))], differences[int(.975 * (samples - 1))]],
        "replicates": samples, "seed": seed,
        "method": "paired prompt bootstrap, resampled within source groups; pooled token acceptance ratio",
    }