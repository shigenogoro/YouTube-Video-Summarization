# src/data/preprocess.py
from functools import partial

def preprocess_examples(example, tokenizer, config, is_train=True):
    """
    example: dict with keys like 'text' (document) and 'summary'
    returns tokenized inputs/labels
    """
    src = example[config["text_column"]]
    tgt = example.get(config.get("summary_column", "summary"), "")

    # tokenization params
    max_input = config.get("max_input_length", 1024)
    max_target = config.get("max_target_length", 256)
    truncation = True

    model_inputs = tokenizer(src,
                             max_length=max_input,
                             truncation=truncation,
                             padding=False)

    # tokenized labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt, max_length=max_target, truncation=truncation, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_dataset(dataset, tokenizer, preprocess_config):
    fn = partial(preprocess_examples, tokenizer=tokenizer, config=preprocess_config)
    tokenized = dataset.map(fn, batched=False)
    return tokenized
