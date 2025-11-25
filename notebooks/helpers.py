from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing import Callable


def make_tokenize_fn(tokenizer: PreTrainedTokenizerBase, input_col: str = "transcript", target_col: str = "summary", max_input_length: int = 1024, max_target_length: int = 128) -> Callable:
    """Return a function suitable for `datasets.Dataset.map` that tokenizes examples.

    The returned function accepts a batch (dict of lists) and returns tokenized batch.
    """
    def fn(examples):
        inputs = examples[input_col]
        targets = examples.get(target_col)
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="longest")
        if targets is not None:
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="longest")
            model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return fn


def sample_dataset():
    data = {
        "dialogue": [
            "Alice: Hello Bob. How are you?\nBob: I'm fine, thanks. We need to discuss the meeting notes.",
            "Speaker1: This video shows examples of summarization tasks. Speaker2: We demonstrate modular code structure."
        ],
        "summary": [
            "Short meeting introduction and planning.",
            "Demonstration of modular summarization code."
        ]
    }
    return Dataset.from_dict(data)
