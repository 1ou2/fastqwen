"""Text-only SFT serialization and padding without target truncation."""

from collections.abc import Mapping

from .qa import assess


def _token_ids(encoded) -> list[int]:
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
        raise ValueError("Expected one unbatched sequence of token IDs")
    return values


def encode_example(tokenizer, template: str, record: dict, max_length: int) -> dict | None:
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    destination, reasons = assess(record)
    if destination != "accepted":
        raise ValueError(f"Non-accepted training record: {record.get('id')}: {reasons}")
    messages = [{"role": "user", "content": record["prompt"]}]
    settings = {
        "chat_template": template, "tokenize": True, "enable_thinking": False,
        "reasoning_effort": record["spec"]["reasoning_effort"],
    }
    prefix = _token_ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **settings))
    complete = _token_ids(tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["content"]}],
        add_generation_prompt=False, **settings,
    ))
    if complete[:len(prefix)] != prefix:
        raise ValueError(f"Template prefix mismatch: {record['id']}")
    if len(complete) <= len(prefix):
        raise ValueError(f"No assistant targets: {record['id']}")
    if len(complete) > max_length:
        return None
    return {
        "input_ids": complete,
        "attention_mask": [1] * len(complete),
        "labels": [-100] * len(prefix) + complete[len(prefix):],
    }


def pad_examples(examples: list[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(example["input_ids"]) for example in examples)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for example in examples:
        length = len(example["input_ids"])
        if any(len(example[key]) != length for key in batch):
            raise ValueError("Input, attention, and label lengths differ")
        padding = width - length
        batch["input_ids"].append(example["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(example["attention_mask"] + [0] * padding)
        batch["labels"].append(example["labels"] + [-100] * padding)
    return batch


def restore_model_weights(model, restored, loading_info: dict) -> None:
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint did not fully load: {loading_info}")
    state = restored.state_dict()
    if state.keys() != model.state_dict().keys():
        raise ValueError("Checkpoint and active model parameter keys differ")
    model.load_state_dict(state, strict=True)