import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


DATASET_REPO = "lucas-ventura/chapter-llama"

# Small model for local experimentation
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# IMPORTANT: must be <= model.config.max_position_embeddings (2048 for TinyLlama)
MAX_INPUT_TOKENS = 1000   # total sequence length (prompt + targets)

# Keep transcripts modest so chapters fit too
MAX_TRANSCRIPT_CHARS = 1500

SEED = 42


def seconds_to_hhmmss(t: float) -> str:
    """Convert seconds to HH:MM:SS (floor)."""
    t = max(0, int(t))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_get(d: Dict, key: str, default=None):
    v = d.get(key, default)
    return v if v is not None else default


def load_json_from_hf(path_in_repo: str) -> Dict[str, Any]:
    """Download a JSON file from the HF dataset repo and load it."""
    local_path = hf_hub_download(repo_id=DATASET_REPO, filename=path_in_repo, repo_type="dataset")
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)



class VidChaptersAsrDataset(Dataset):
    """
    ASR-only dataset for Chapter-Llama style training.

    Assumptions (based on the released subset JSONs):
    - chapters_*.json:
        {
            "<video_id>": {
             "duration": float,
             "title": str,
             "description": str,
             "channel_id": str,
             "view_count": int,
             "chapters": {
                "<start_sec>": "<chapter title>"
                ...
                }
            },
          ...
        }

    - asrs_*.json: supports TWO shapes:

        (1) Dict-of-lists (as you observed):
            {
                "<video_id>": {
                "text":  [str, ...],
                "start": [float, ...],
                "end":   [float, ...]
            },
          ...
        }

        (2) List-of-dicts (original assumption):
            {
                "<video_id>": [
                    {"start": float, "end": float, "text": str},
                    ...
                    ],
                ...
            }

    Each __getitem__ returns a dict with tokenized
    input_ids, attention_mask, labels for a *single video*.
    """

    def __init__(
        self,
        tokenizer,
        chapters_json: Dict[str, Any],
        asrs_json: Dict[str, Any],
        max_input_tokens: int = 2048,
        max_transcript_chars: int = 3000,
        max_videos: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.chapters_json = chapters_json
        self.asrs_json = asrs_json
        self.max_input_tokens = max_input_tokens
        self.max_transcript_chars = max_transcript_chars

        all_vids = sorted(set(chapters_json.keys()) & set(asrs_json.keys()))
        if max_videos is not None:
            all_vids = all_vids[:max_videos]
        self.video_ids = all_vids

        # Pre-tokenize static instruction skeleton to avoid recomputing its ids
        self.base_instruction = (
            "You are an expert system that segments long YouTube videos into "
            "semantically meaningful chapters and names each chapter.\n\n"
            "Given the video title, description, and a subset of the ASR transcript "
            "with timestamps, output ALL chapters for the video as lines of the form:\n"
            "HH:MM:SS - Chapter title\n\n"
            "Be faithful to the content and avoid hallucinating chapters that are "
            "not supported by the transcript.\n\n"
        )

    def __len__(self):
        return len(self.video_ids)

    def _build_transcript_text(self, asr_segments) -> str:
        """
            Build a textual representation of ASR:

            Handles:
            - dict-of-lists: {"text": [...], "start": [...], "end": [...]}
            - list-of-dicts: [{"text": ..., "start": ..., "end": ...}, ...]

            Returns a single string like:
              [HH:MM:SS] utterance
              [HH:MM:SS] next utterance
              ...
            truncated by self.max_transcript_chars.
        """

        triplets = []

        if isinstance(asr_segments, dict) and all(
            k in asr_segments for k in ("text", "start", "end")
        ):
            texts = asr_segments["text"]
            starts = asr_segments["start"]
            ends = asr_segments["end"]
            for t, s, e in zip(texts, starts, ends):
                if not t:
                    continue
                try:
                    s_float = float(s)
                except Exception:
                    s_float = 0.0
                triplets.append((s_float, e, t))
        else:
            # Assume list[dict]
            for seg in asr_segments:
                text = seg.get("text", "")
                if not text:
                    continue
                try:
                    s_float = float(seg.get("start", 0.0))
                except Exception:
                    s_float = 0.0
                triplets.append((s_float, seg.get("end", 0.0), text))

        triplets.sort(key=lambda x: x[0])  # sort by start time

        chunks = []
        total_chars = 0

        for start, end, text in triplets:
            ts = seconds_to_hhmmss(start)
            line = f"[{ts}] {text.strip()}"
            new_len = total_chars + len(line) + 1
            if new_len > self.max_transcript_chars:
                break
            chunks.append(line)
            total_chars = new_len

        return "\n".join(chunks)

    def _build_chapter_target(self, chapters_dict: Dict[str, str]) -> str:
        """
        Turn chapters dict into canonical text:
        HH:MM:SS - Title
        one per line, sorted by time.
        """
        items = []
        for start_str, title in chapters_dict.items():
            try:
                t = float(start_str)
            except Exception:
                # Sometimes keys might already be numeric-ish strings; fallback
                try:
                    t = float(str(start_str).replace(",", ""))
                except Exception:
                    t = 0.0
            items.append((t, title))
        items.sort(key=lambda x: x[0])
        lines = [f"{seconds_to_hhmmss(t)} - {title}" for t, title in items]
        return "\n".join(lines)

    def __getitem__(self, idx):
        """
        Critical invariants:
        - Total sequence length <= self.max_input_tokens
        - There is at least ONE label token != -100
        """

        # Safety loop: try a few different videos if current one is unusable
        for _attempt in range(5):
            vid = self.video_ids[idx]
            chap_entry = self.chapters_json[vid]
            asr_segments = self.asrs_json[vid]

            video_title = safe_get(chap_entry, "title", "")
            video_desc = safe_get(chap_entry, "description", "")
            chapters_dict = safe_get(chap_entry, "chapters", {})

            # Build transcript and target chapter text
            transcript_text = self._build_transcript_text(asr_segments)
            chapters_text = self._build_chapter_target(chapters_dict)

            # ---- Prompt & target strings ----
            prompt = (
                self.base_instruction
                + f"Video title: {video_title}\n"
                + f"Video description: {video_desc}\n\n"
                + "### Transcript (partial, chronologically ordered):\n"
                + transcript_text
                + "\n\n### Chapters\n"
            )
            target = chapters_text

            # ---- Separate tokenization for prompt/target ----
            max_total = self.max_input_tokens
            max_prompt = int(max_total * 0.75)   # allow prompt to take up to 75%
            if max_prompt < 1:
                max_prompt = max_total // 2

            # Prompt
            prompt_enc = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt,
            )
            prompt_ids = prompt_enc["input_ids"]

            # Remaining room for target
            max_target = max_total - len(prompt_ids)
            if max_target <= 0:
                # Prompt alone too long: keep only last chunk of prompt
                prompt_ids = prompt_ids[-(max_total // 2):]
                max_target = max_total - len(prompt_ids)

            # Target (chapter lines)
            target_enc = self.tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max_target,
            )
            target_ids = target_enc["input_ids"]

            # If no target tokens → this sample is useless; try another video
            if len(target_ids) == 0:
                idx = (idx + 1) % len(self.video_ids)
                continue

            input_ids = prompt_ids + target_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + target_ids

            # Safety: truncate everything to max_total
            if len(input_ids) > max_total:
                input_ids = input_ids[:max_total]
                attention_mask = attention_mask[:max_total]
                labels = labels[:max_total]

            # Final check: ensure at least one token has a real label
            if all(l == -100 for l in labels):
                idx = (idx + 1) % len(self.video_ids)
                continue

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "video_id": vid,
            }

        # If we somehow failed 5 times in a row:
        raise RuntimeError("Failed to construct a valid training example with at least one label token.")


# -----------------------------
# COLLATOR
# -----------------------------
@dataclass
class ChapteringCollator:
    tokenizer: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Variable-length → pad
        max_len = max(len(ex["input_ids"]) for ex in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for ex in batch:
            n = len(ex["input_ids"])
            pad_len = max_len - n

            input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# -----------------------------
# INFERENCE
# -----------------------------
def run_single_video_inference(
    model_dir: str,
    subset_tag: str = "s",
    max_new_tokens: int = 256,
):
    """
    Load a fine-tuned checkpoint and run chaptering on ONE held-out video
    from chapters_{subset_tag}_test.json + asrs_{subset_tag}_test.json.

    subset_tag="s" → use:
      docs/subset_data/chapters/chapters_s_test.json
      docs/subset_data/asrs/asrs_s_test.json
    """

    # --------------------------
    # Device & model
    # --------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFER] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # --------------------------
    # Load test JSONs
    # --------------------------
    chapters_test = load_json_from_hf(
        f"docs/subset_data/chapters/chapters_{subset_tag}_test.json"
    )
    asrs_test = load_json_from_hf(
        f"docs/subset_data/asrs/asrs_{subset_tag}_test.json"
    )

    # Tiny dataset wrapper to reuse formatting helpers
    test_dataset = VidChaptersAsrDataset(
        tokenizer=tokenizer,
        chapters_json=chapters_test,
        asrs_json=asrs_test,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_transcript_chars=MAX_TRANSCRIPT_CHARS,
        max_videos=None,
    )

    if len(test_dataset) == 0:
        raise ValueError("[INFER] No test videos found.")

    vid0 = test_dataset.video_ids[0]
    chap_entry = chapters_test[vid0]
    asr_segments = asrs_test[vid0]

    video_title = chap_entry.get("title", "")
    video_desc = chap_entry.get("description", "")
    chapters_dict = chap_entry.get("chapters", {})

    # Reuse dataset helpers for text format
    transcript_text = test_dataset._build_transcript_text(asr_segments)
    gt_chapters_text = test_dataset._build_chapter_target(chapters_dict)
    base_instruction = test_dataset.base_instruction

    # Build prompt exactly like during training (minus target)
    prompt = (
        base_instruction
        + f"Video title: {video_title}\n"
        + f"Video description: {video_desc}\n\n"
        + "### Transcript (partial, chronologically ordered):\n"
        + transcript_text
        + "\n\n### Chapters\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Cut off the prompt, keep only new tokens
    input_len = inputs["input_ids"].shape[1]
    gen_only = gen_ids[0][input_len:]
    pred_text = tokenizer.decode(gen_only, skip_special_tokens=True)

    # Pretty-print
    print("=" * 80)
    print(f"[INFER] VIDEO ID: {vid0}")
    print("=" * 80)

    print("\n[INFER] TRANSCRIPT (first ~15 lines):")
    t_lines = transcript_text.splitlines()
    for line in t_lines[:15]:
        print(line)
    if len(t_lines) > 15:
        print("... (transcript truncated) ...")

    print("\n[INFER] GROUND-TRUTH CHAPTERS:")
    print(gt_chapters_text)

    print("\n[INFER] PREDICTED CHAPTERS:")
    print(pred_text)
    print("=" * 80)


# -----------------------------
# MAIN
# -----------------------------
def main():
    set_seed(SEED)

    # Device (MPS on Apple Silicon if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_INPUT_TOKENS

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False  # important for training

    max_ctx = model.config.max_position_embeddings
    print("Model max position embeddings:", max_ctx)
    assert MAX_INPUT_TOKENS <= max_ctx

    model.to(device)

    # -------------------------
    # Load subset JSONs
    # -------------------------
    print("Loading subset JSONs from Hugging Face...")

    chapters_train = load_json_from_hf("docs/subset_data/chapters/chapters_sml1k_train.json")
    asrs_train = load_json_from_hf("docs/subset_data/asrs/asrs_sml1k_train.json")

    chapters_val = load_json_from_hf("docs/subset_data/chapters/chapters_sml300_val.json")
    asrs_val = load_json_from_hf("docs/subset_data/asrs/asrs_sml300_val.json")

    # Build datasets
    train_dataset = VidChaptersAsrDataset(
        tokenizer=tokenizer,
        chapters_json=chapters_train,
        asrs_json=asrs_train,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_transcript_chars=MAX_TRANSCRIPT_CHARS,
        max_videos=50,  # or e.g. 200 for very quick smoke test
    )

    val_dataset = VidChaptersAsrDataset(
        tokenizer=tokenizer,
        chapters_json=chapters_val,
        asrs_json=asrs_val,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_transcript_chars=MAX_TRANSCRIPT_CHARS,
        max_videos=50,
    )

    print(f"#train videos: {len(train_dataset)}")
    print(f"#val videos:   {len(val_dataset)}")

    collator = ChapteringCollator(tokenizer=tokenizer)

    # -------------------------
    # Training args
    # -------------------------
    output_dir = "outputs/chapter_llama_asr_sml1k_tiny"
    os.makedirs(output_dir, exist_ok=True)

    # Conservative settings for stability on M4 + TinyLlama
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1.0,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,  # keep off for now for stability
        remove_unused_columns=False,
        max_grad_norm=1.0,             # gradient clipping
    )

    # Simple metric: validation loss (Trainer logs eval_loss) + we can add perplexity later
    def compute_metrics(eval_pred):
        # logits, labels = eval_pred
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Done.")

    # Save final model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Quick inference run on one held-out test video
    # ------------------------------------------------------------------
    print("\n=== Running quick inference on one held-out test video ===")
    run_single_video_inference(
        model_dir=output_dir,
        subset_tag="s",         # uses chapters_s_test / asrs_s_test
        max_new_tokens=256,
    )


if __name__ == "__main__":
    main()
