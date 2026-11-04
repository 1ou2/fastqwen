import pytest

from qwenbench.training import encode_example, pad_examples, restore_model_weights


class Tokenizer:
    def __init__(self, mapping=True, mismatch=False):
        self.mapping = mapping
        self.mismatch = mismatch

    def apply_chat_template(self, messages, *, add_generation_prompt, **settings):
        assert settings["enable_thinking"] is False
        assert settings["reasoning_effort"] == "low"
        values = [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5, 9]
        if self.mismatch and not add_generation_prompt:
            values[0] = 7
        return {"input_ids": values} if self.mapping else values


def record():
    return {
        "id": "example", "slice": "S1", "prompt": "Question", "content": "Answer",
        "finish_reason": "stop", "spec": {"thinking": False, "reasoning_effort": "low"},
    }


@pytest.mark.parametrize("mapping", [True, False])
def test_prompt_masking_preserves_answer_and_end_token(mapping):
    encoded = encode_example(Tokenizer(mapping), "template", record(), 6)
    assert encoded["input_ids"] == [1, 2, 3, 4, 5, 9]
    assert encoded["labels"] == [-100, -100, -100, 4, 5, 9]


def test_overlength_target_is_skipped_not_truncated():
    assert encode_example(Tokenizer(), "template", record(), 5) is None


def test_prefix_mismatch_fails_closed():
    with pytest.raises(ValueError, match="prefix mismatch"):
        encode_example(Tokenizer(mismatch=True), "template", record(), 20)


def test_quarantined_record_cannot_enter_training():
    example = record()
    example["tool_calls"] = [{"name": "lookup"}]
    with pytest.raises(ValueError, match="Non-accepted"):
        encode_example(Tokenizer(), "template", example, 20)


def test_padding_masks_padding_but_not_real_end_token():
    long = encode_example(Tokenizer(), "template", record(), 20)
    short = {"input_ids": [1, 9], "attention_mask": [1, 1], "labels": [-100, 9]}
    batch = pad_examples([long, short], 9)
    assert batch["input_ids"][1] == [1, 9, 9, 9, 9, 9]
    assert batch["attention_mask"][1] == [1, 1, 0, 0, 0, 0]
    assert batch["labels"][1] == [-100, 9, -100, -100, -100, -100]


def test_empty_batch_is_invalid():
    with pytest.raises(ValueError, match="empty batch"):
        pad_examples([], 0)


class Model:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict):
        assert strict is True
        self.state = dict(state)


def test_resume_restores_weights_not_only_step_counter():
    active = Model({"weight": 1})
    restore_model_weights(active, Model({"weight": 9}), {})
    assert active.state["weight"] == 9


@pytest.mark.parametrize("diagnostic", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_resume_refuses_incomplete_loader_results(diagnostic):
    active = Model({"weight": 1})
    with pytest.raises(ValueError, match="did not fully load"):
        restore_model_weights(active, Model({"weight": 9}), {diagnostic: ["problem"]})
    assert active.state["weight"] == 1


def test_resume_refuses_different_parameter_keys():
    with pytest.raises(ValueError, match="parameter keys differ"):
        restore_model_weights(Model({"weight": 1}), Model({"different": 9}), {})


def test_trainer_entry_point_parses():
    from scripts.train_draft import main

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


def test_unvalidated_distributed_launch_is_refused(monkeypatch, tmp_path):
    from scripts.train_draft import main

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit) as result:
        main(["--qa-dir", str(tmp_path), "--output-dir", str(tmp_path / "run"), "--group", "s1-en"])
    assert result.value.code == 2