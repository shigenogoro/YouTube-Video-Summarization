from datasets import load_dataset, DatasetDict
import os

def load_json_dataset(data_config):
    """
    data_config: dict with keys:
      - name: "meetingbank" or "local"
      - path: if local, path to folder or file pattern
      - hf_name: optional HuggingFace dataset identifier
    Returns: datasets.DatasetDict with train/validation/test
    """
    if data_config.get("hf_name"):
        ds = load_dataset(data_config["hf_name"])
        # Ensure train/validation/test exist or split
        return ds
    if data_config.get("name") == "meetingbank":
        # assume user provided local files under data/meetingbank/*.jsonl
        files = {}
        base = data_config.get("path", "data/meetingbank")
        # Expect train.jsonl, valid.jsonl, test.jsonl
        files["train"] = os.path.join(base, "train.jsonl")
        files["validation"] = os.path.join(base, "valid.jsonl")
        files["test"] = os.path.join(base, "test.jsonl")
        ds = load_dataset("json", data_files=files)
        return DatasetDict({
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"]
        })
    # generic local jsonl with split keys
    if data_config.get("path"):
        files = data_config["path"]
        ds = load_dataset("json", data_files=files)
        return ds
    raise ValueError("Unsupported data config")
