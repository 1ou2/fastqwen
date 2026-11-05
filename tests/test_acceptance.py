import pytest

from qwenbench.acceptance import aggregate_replays, paired_acceptance_interval, replay_greedy


def test_full_matches_include_separate_teacher_bonus():
    def propose(prefix, budget):
        position = len(prefix) - 1
        return [10, 11, 12, 13, 14][position:position + budget]

    result = replay_greedy([99], [10, 11, 12, 13, 14], propose, 2)
    assert result["accepted_tokens"] == 4
    assert result["bonus_tokens"] == 1
    assert result["correction_tokens"] == 0
    assert result["verification_cycles"] == 2
    assert result["output_ids"] == [10, 11, 12, 13, 14]


def test_only_consecutive_matches_before_first_mismatch_are_accepted():
    result = replay_greedy([], [1, 2, 3], lambda prefix, budget: [1, 8, 3][:budget], 3)
    assert result["accepted_tokens"] == 1
    assert result["correction_tokens"] == 2
    assert result["output_ids"] == [1, 2, 3]
    assert result["accepted_prefix_lengths"] == [1, 0]


def test_first_mismatch_commits_teacher_correction_before_next_proposal():
    seen = []

    def propose(prefix, budget):
        seen.append(prefix)
        return [9] * budget

    result = replay_greedy([99], [1, 2, 3], propose, 2)
    assert seen == [[99], [99, 1], [99, 1, 2]]
    assert result["accepted_tokens"] == 0
    assert result["correction_tokens"] == 3


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_final_partial_block_accounting(gamma):
    result = replay_greedy([], [7], lambda prefix, budget: [7], gamma)
    assert result["acceptance_rate"] == 1
    assert result["bonus_tokens"] == 0
    assert result["output_tokens"] == result["accepted_tokens"] + result["correction_tokens"] + result["bonus_tokens"]


def test_eos_is_an_ordinary_final_trace_token():
    result = replay_greedy([], [1, 99], lambda prefix, budget: [1, 99][len(prefix):len(prefix) + budget], 8)
    assert result["output_ids"] == [1, 99]
    assert result["accepted_tokens"] == 2


@pytest.mark.parametrize("block", [[], [1, 2, 3], [True]])
def test_invalid_proposal_fails_closed(block):
    with pytest.raises(ValueError, match="Draft must return"):
        replay_greedy([], [1, 2], lambda prefix, budget: block, 2)


def test_invalid_gamma_or_empty_trace_fails_closed():
    with pytest.raises(ValueError):
        replay_greedy([], [1], lambda prefix, budget: [1], 0)
    with pytest.raises(ValueError):
        replay_greedy([], [], lambda prefix, budget: [1], 1)


def test_pooled_acceptance_is_not_mean_of_prompt_ratios():
    rows = [replay_greedy([], [1], lambda prefix, budget: [1], 1),
            replay_greedy([], [1, 2, 3], lambda prefix, budget: [9] * budget, 2)]
    result = aggregate_replays(rows)
    assert result["acceptance_rate"] == 1 / 6
    assert result["prompts"] == 2


def test_paired_bootstrap_is_reproducible_and_zero_for_identical_arms():
    row = {**replay_greedy([], [1], lambda prefix, budget: [1], 1), "id": "one", "group": "en", "gamma": 1}
    result = paired_acceptance_interval([(row, row)], 42, 100)
    assert result["trained_minus_raw"] == 0
    assert result["percentile_95"] == [0, 0]
    assert result == paired_acceptance_interval([(row, row)], 42, 100)


def test_paired_bootstrap_refuses_mismatched_prompts():
    with pytest.raises(ValueError, match="not paired"):
        paired_acceptance_interval([({"id": "a", "group": "en", "gamma": 1},
                                     {"id": "b", "group": "en", "gamma": 1})], 0)


def test_comparison_cli_parses():
    from scripts.compare_drafts import main

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


def test_invalid_comparison_counts_are_refused(tmp_path):
    from scripts.compare_drafts import main

    with pytest.raises(SystemExit) as result:
        main(["prepare", "--run-dir", str(tmp_path / "run"), "--per-group", "0"])
    assert result.value.code == 2
    assert not (tmp_path / "run").exists()