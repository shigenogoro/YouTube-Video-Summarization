import os
import re
import json

import random
import argparse

from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import ttest_rel

import numpy as np
import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

import evaluate
import matplotlib.pyplot as plt


COLOR_MAIN = "tab:blue"
COLOR_AUX = "lightgray"
COLOR_ERR = "black"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_json_from_hf(repo_id: str, filename: str) -> dict:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def seconds_to_hhmmss(t: float) -> str:
    t = max(0, int(t))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def normalize_transcript(asr_data: Any) -> List[Dict[str, Any]]:
    segments = []
    if isinstance(asr_data, dict):
        texts = asr_data.get("text", [])
        starts = asr_data.get("start", [])
        ends = asr_data.get("end", [])
        for t, s, e in zip(texts, starts, ends):
            segments.append({"start": float(s), "end": float(e), "text": str(t)})
    elif isinstance(asr_data, list):
        for seg in asr_data:
            segments.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": str(seg.get("text", "")),
                }
            )
    segments.sort(key=lambda x: x["start"])
    return segments


def normalize_chapters(chapters_dict: Dict[str, str], duration: float) -> List[Dict[str, Any]]:
    sorted_chaps = []
    for start_str, title in chapters_dict.items():
        try:
            start = float(start_str)
        except Exception:
            continue
        sorted_chaps.append((start, str(title)))
    sorted_chaps.sort(key=lambda x: x[0])

    final = []
    for i, (start, title) in enumerate(sorted_chaps):
        if i < len(sorted_chaps) - 1:
            end = float(sorted_chaps[i + 1][0])
        else:
            end = float(duration) if duration and duration > 0 else float(start + 600.0)
        final.append(
            {
                "start": float(start),
                "end": float(end),
                "title": title,
                "start_hhmmss": seconds_to_hhmmss(start),
                "end_hhmmss": seconds_to_hhmmss(end),
            }
        )
    return final


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def jaccard_overlap(a: str, b: str) -> float:
    sa = set(_tokenize_simple(a))
    sb = set(_tokenize_simple(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def resolve_hf_model_dir(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isdir(path):
        cfg = os.path.join(path, "config.json")
        w1 = os.path.join(path, "pytorch_model.bin")
        w2 = os.path.join(path, "model.safetensors")
        if os.path.exists(cfg) and (os.path.exists(w1) or os.path.exists(w2)):
            return path
        subdirs = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isdir(full) and name.startswith("checkpoint-"):
                subdirs.append(full)
        def _step(p: str) -> int:
            m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        subdirs.sort(key=_step, reverse=True)
        for sd in subdirs:
            cfg = os.path.join(sd, "config.json")
            w1 = os.path.join(sd, "pytorch_model.bin")
            w2 = os.path.join(sd, "model.safetensors")
            if os.path.exists(cfg) and (os.path.exists(w1) or os.path.exists(w2)):
                return sd
    return None


def extract_k_features(chapters: dict, asrs: dict, video_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for vid in video_ids:
        chap_entry = chapters.get(vid, {})
        asr_entry = asrs.get(vid, [])
        duration = float(chap_entry.get("duration", 0.0) or 0.0)
        if duration <= 0:
            continue
        transcript = normalize_transcript(asr_entry)
        if not transcript:
            continue
        transcript_text = " ".join([seg["text"] for seg in transcript])
        num_words = len(transcript_text.split())
        num_asr_segments = len(transcript)
        wps = (num_words / duration) if duration > 0 else 0.0
        num_chapters = len((chap_entry.get("chapters", {}) or {}))
        if num_chapters > 0:
            X.append([duration, num_words, num_asr_segments, wps])
            y.append(int(num_chapters))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_k_predictor_once(X: np.ndarray, y: np.ndarray, out_path: str, seed: int) -> Tuple[Any, Dict[str, Any]]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        model = joblib.load(out_path)
        return model, {"loaded": True}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    preds_round = np.round(preds).astype(int)
    acc = float(accuracy_score(y_test, preds_round))
    rec = float(recall_score(y_test, preds_round, average="weighted", zero_division=0))

    joblib.dump(model, out_path)
    return model, {"loaded": False, "mae": mae, "exact_match_acc": acc, "weighted_recall": rec}


def get_text_for_range(transcript: List[Dict[str, Any]], start: float, end: float) -> str:
    out = []
    for seg in transcript:
        c = (float(seg["start"]) + float(seg["end"])) / 2.0
        if float(start) <= c <= float(end):
            out.append(seg["text"])
    return " ".join(out).strip()


def build_ground_truth_chapter_dataset(chapters: dict, asrs: dict, video_ids: List[str]) -> Dataset:
    rows = []
    for vid in video_ids:
        chap_entry = chapters.get(vid, {})
        asr_entry = asrs.get(vid, [])
        duration = float(chap_entry.get("duration", 0.0) or 0.0)
        chap_dict = chap_entry.get("chapters", {}) or {}
        chaps = normalize_chapters(chap_dict, duration)
        transcript = normalize_transcript(asr_entry)
        if not transcript or not chaps:
            continue
        for c in chaps:
            text = get_text_for_range(transcript, c["start"], c["end"])
            if not text:
                continue
            rows.append(
                {
                    "video_id": vid,
                    "title": c["title"],
                    "text": text,
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "duration": float(duration) if duration > 0 else float(transcript[-1]["end"]),
                }
            )
    return Dataset.from_list(rows)


def fine_tune_bart_model(
    gt_dataset: Dataset,
    output_dir: str,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
    base_ckpt: str = "facebook/bart-base",
    max_train_samples: int = 0,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    existing = resolve_hf_model_dir(output_dir)
    if existing is not None:
        return existing

    set_seed(seed)
    device = get_device()

    ds = gt_dataset
    if max_train_samples and max_train_samples > 0 and len(ds) > max_train_samples:
        ds = ds.shuffle(seed=seed).select(range(max_train_samples))

    split = ds.train_test_split(test_size=0.1, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    tok = AutoTokenizer.from_pretrained(base_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_ckpt).to(device)

    def _tok(batch):
        x = tok(
            ["summarize: " + t for t in batch["text"]],
            truncation=True,
            max_length=max_input_length,
            padding="max_length",
        )
        y = tok(
            batch["title"],
            truncation=True,
            max_length=max_target_length,
            padding="max_length",
        )
        x["labels"] = y["input_ids"]
        return x

    train_t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    eval_t = eval_ds.map(_tok, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="no",
        save_strategy="no",
        predict_with_generate=False,
        load_best_model_at_end=False,
        fp16=(device == "cuda"),
        report_to="none",
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_t,
        eval_dataset=eval_t,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    resolved = resolve_hf_model_dir(output_dir)
    if resolved is None:
        raise RuntimeError(f"No usable BART checkpoint produced under: {output_dir}")
    return resolved


def create_alignment_pairs(gt_dataset: Dataset, num_negatives: int, seed: int, max_pairs: int = 0) -> Dataset:
    df = gt_dataset.to_pandas()
    grouped = df.groupby("video_id")
    rng = random.Random(seed)

    rows = []
    for vid, g in grouped:
        texts = g["text"].tolist()
        for _, row in g.iterrows():
            title = row["title"]
            true_text = row["text"]

            rows.append({"title": title, "text_segment": true_text, "label": 1})

            neg_candidates = [t for t in texts if t != true_text]
            if neg_candidates:
                take = min(num_negatives, len(neg_candidates))
                negs = rng.sample(neg_candidates, take)
                for nt in negs:
                    rows.append({"title": title, "text_segment": nt, "label": 0})

            if max_pairs and max_pairs > 0 and len(rows) >= max_pairs:
                break
        if max_pairs and max_pairs > 0 and len(rows) >= max_pairs:
            break

    rng.shuffle(rows)
    return Dataset.from_list(rows)


def fine_tune_alignment_model(
    gt_dataset: Dataset,
    output_dir: str,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    num_negatives: int,
    max_train_pairs: int = 0,
    base_ckpt: str = "bert-base-uncased",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    existing = resolve_hf_model_dir(output_dir)
    if existing is not None:
        return existing

    set_seed(seed)
    device = get_device()

    pairs = create_alignment_pairs(gt_dataset, num_negatives=num_negatives, seed=seed, max_pairs=max_train_pairs)
    split = pairs.train_test_split(test_size=0.1, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    tok = AutoTokenizer.from_pretrained(base_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(base_ckpt, num_labels=2).to(device)

    def _tok(batch):
        enc = tok(
            batch["title"],
            batch["text_segment"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        enc["labels"] = batch["label"]
        return enc

    train_t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    eval_t = eval_ds.map(_tok, batched=True, remove_columns=eval_ds.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=(device == "cuda"),
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_t,
        eval_dataset=eval_t,
        tokenizer=tok,
        compute_metrics=None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    resolved = resolve_hf_model_dir(output_dir)
    if resolved is None:
        raise RuntimeError(f"No usable ALIGN checkpoint produced under: {output_dir}")
    return resolved


def predict_num_segments(duration: float, transcript_segments: List[Dict[str, Any]], model: Any) -> int:
    transcript_text = " ".join([s["text"] for s in transcript_segments])
    num_words = len(transcript_text.split())
    num_asr_segments = len(transcript_segments)
    wps = (num_words / duration) if duration > 0 else 0.0
    feats = np.array([[duration, num_words, num_asr_segments, wps]], dtype=np.float32)
    pred = float(model.predict(feats)[0])
    k = int(max(1, round(pred)))
    return k


def segment_transcript_fixed_k(asr_segments: List[Dict[str, Any]], k: int, duration: Optional[float] = None) -> List[Dict[str, Any]]:
    if not asr_segments:
        return []
    asr_segments = sorted(asr_segments, key=lambda x: x["start"])
    if duration is None or duration <= 0:
        duration = float(asr_segments[-1]["end"])
    k = int(max(1, k))
    if k == 1:
        return [
            {
                "start": float(asr_segments[0]["start"]),
                "end": float(asr_segments[-1]["end"]),
                "text": " ".join([s["text"] for s in asr_segments]).strip(),
            }
        ]
    bounds = [i * (duration / k) for i in range(k + 1)]
    windows = []
    for i in range(k):
        b0, b1 = float(bounds[i]), float(bounds[i + 1])
        segs = []
        for s in asr_segments:
            c = (float(s["start"]) + float(s["end"])) / 2.0
            if (i < k - 1 and b0 <= c < b1) or (i == k - 1 and b0 <= c <= b1 + 1e-6):
                segs.append(s)
        if not segs:
            continue
        windows.append(
            {
                "start": float(segs[0]["start"]),
                "end": float(segs[-1]["end"]),
                "text": " ".join([x["text"] for x in segs]).strip(),
            }
        )
    if not windows:
        windows = [
            {
                "start": float(asr_segments[0]["start"]),
                "end": float(asr_segments[-1]["end"]),
                "text": " ".join([s["text"] for s in asr_segments]).strip(),
            }
        ]
    return windows


def segment_transcript_semantic_k(asr_segments: List[Dict[str, Any]], k: int, sbert_model: Any) -> List[Dict[str, Any]]:
    if not asr_segments:
        return []
    asr_segments = sorted(asr_segments, key=lambda x: x["start"])
    k = int(max(1, k))
    if k == 1 or len(asr_segments) == 1:
        return [
            {
                "start": float(asr_segments[0]["start"]),
                "end": float(asr_segments[-1]["end"]),
                "text": " ".join([s["text"] for s in asr_segments]).strip(),
            }
        ]
    k = min(k, len(asr_segments))
    texts = [seg["text"] for seg in asr_segments]
    emb = sbert_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    sims = []
    for i in range(len(emb) - 1):
        sim = float(cosine_similarity([emb[i]], [emb[i + 1]])[0][0])
        sims.append((sim, i))
    sims.sort(key=lambda x: x[0])
    split_indices = sorted([idx for _, idx in sims[: max(0, k - 1)]])
    windows = []
    cur = 0
    for si in split_indices:
        segs = asr_segments[cur : si + 1]
        if segs:
            windows.append(
                {
                    "start": float(segs[0]["start"]),
                    "end": float(segs[-1]["end"]),
                    "text": " ".join([s["text"] for s in segs]).strip(),
                }
            )
        cur = si + 1
    tail = asr_segments[cur:]
    if tail:
        windows.append(
            {
                "start": float(tail[0]["start"]),
                "end": float(tail[-1]["end"]),
                "text": " ".join([s["text"] for s in tail]).strip(),
            }
        )
    return windows


def build_segmented_dataset_for_eval(
    chapters_data: dict,
    asrs_data: dict,
    video_ids: List[str],
    method: str,
    fixed_k: int,
    k_model: Any,
    sbert_model: Any,
) -> Dataset:
    rows = []
    for vid in video_ids:
        chap_entry = chapters_data.get(vid, {})
        asr_entry = asrs_data.get(vid, [])
        duration = float(chap_entry.get("duration", 0.0) or 0.0)
        chapters_norm = normalize_chapters(chap_entry.get("chapters", {}) or {}, duration)
        transcript_norm = normalize_transcript(asr_entry)
        if not transcript_norm:
            continue

        if method == "fixed":
            k = int(fixed_k)
            segs = segment_transcript_fixed_k(transcript_norm, k=k, duration=duration if duration > 0 else None)
        elif method == "kpred":
            dur = duration if duration > 0 else float(transcript_norm[-1]["end"])
            k = predict_num_segments(dur, transcript_norm, k_model)
            segs = segment_transcript_semantic_k(transcript_norm, k=k, sbert_model=sbert_model)
        else:
            continue

        for seg_idx, seg in enumerate(segs):
            best_overlap = 0.0
            best_title = ""
            for chap in chapters_norm:
                s0 = max(float(seg["start"]), float(chap["start"]))
                s1 = min(float(seg["end"]), float(chap["end"]))
                overlap = max(0.0, s1 - s0)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_title = chap["title"]
            rows.append(
                {
                    "video_id": vid,
                    "seg_idx": int(seg_idx),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": str(seg["text"]),
                    "title": str(best_title) if best_title else "",
                }
            )
    return Dataset.from_list(rows)


def generate_titles(
    ds: Dataset,
    model: Any,
    tokenizer: Any,
    device: str,
    max_input_length: int,
    gen_max_len: int,
    num_beams: int,
    batch_size: int,
) -> List[str]:
    model.eval()
    texts = [x["text"] for x in ds]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            ["summarize: " + t for t in batch],
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_length=gen_max_len,
                num_beams=num_beams,
                early_stopping=True,
            )
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        out.extend([d.strip() for d in dec])
    return out


def compute_summarization_metrics_with_ci(
    preds: List[str],
    refs: List[str],
    seed: int,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    keep = [i for i, r in enumerate(refs) if isinstance(r, str) and r.strip()]
    if not keep:
        return {"n": 0}

    p = [preds[i] for i in keep]
    r = [refs[i] for i in keep]

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=p, references=r, use_aggregator=False)

    bert = evaluate.load("bertscore")
    bert_scores = bert.compute(predictions=p, references=r, lang="en")

    def _bootstrap_ci(values: np.ndarray) -> Tuple[float, float, float]:
        rng = np.random.RandomState(seed)
        values = np.asarray(values, dtype=np.float64)
        m = float(values.mean()) if len(values) else 0.0
        if len(values) < 2:
            return m, m, m
        means = []
        n = len(values)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            means.append(float(values[idx].mean()))
        lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
        return m, lo, hi

    out = {"n": int(len(p))}
    for k in ["rouge1", "rouge2", "rougeL"]:
        arr = np.array(rouge_scores[k], dtype=np.float64)
        m, lo, hi = _bootstrap_ci(arr)
        out[k] = {"mean": m, "ci95": [lo, hi]}
    bf1 = np.array(bert_scores["f1"], dtype=np.float64)
    m, lo, hi = _bootstrap_ci(bf1)
    out["bertscore_f1"] = {"mean": m, "ci95": [lo, hi]}
    return out


def score_alignment_pairs(
    titles: List[str],
    candidate_texts: List[str],
    align_model: Any,
    align_tokenizer: Any,
    device: str,
    batch_size: int = 32) -> np.ndarray:

    scores = []
    n = len(candidate_texts)

    for i in range(0, n, batch_size):
        bt = candidate_texts[i: i + batch_size]

        if len(titles) == n:
            tt = titles[i: i + len(bt)]
        else:
            tt = [titles[0]] * len(bt)

        inputs = align_tokenizer(
            tt,
            bt,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = align_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]

        scores.append(probs.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def build_self_retrieval_alignment_log(
    ds: Dataset,
    preds: List[str],
    method_name: str,
    align_model: Any,
    align_tokenizer: Any,
    device: str,
    tol_list: List[int],
) -> pd.DataFrame:
    df = ds.to_pandas()
    df["gen_title"] = preds
    rows = []
    for vid, g in df.groupby("video_id"):
        g = g.sort_values("seg_idx")
        candidate_texts = g["text"].tolist()
        candidate_starts = g["start"].astype(float).tolist()
        candidate_ends = g["end"].astype(float).tolist()

        for _, row in g.iterrows():
            q = str(row["gen_title"])
            true_start = float(row["start"])
            title_tokens = _tokenize_simple(q)
            cand_scores = score_alignment_pairs(
                titles=[q] * len(candidate_texts),
                candidate_texts=candidate_texts,
                align_model=align_model,
                align_tokenizer=align_tokenizer,
                device=device,
                batch_size=32,
            )
            best_idx = int(np.argmax(cand_scores))
            pred_start = float(candidate_starts[best_idx])
            pred_end = float(candidate_ends[best_idx])
            td = abs(pred_start - true_start)

            seg_text = str(row["text"])
            seg_words = len(seg_text.split())
            jacc = jaccard_overlap(q, seg_text)
            has_inaud = int(bool(re.search(r"\b(inaudible|unintelligible)\b", seg_text.lower())))

            rec = {
                "method": method_name,
                "video_id": vid,
                "seg_idx": int(row["seg_idx"]),
                "seg_start": float(row["start"]),
                "seg_end": float(row["end"]),
                "seg_words": int(seg_words),
                "gen_title": q,
                "title_words": int(len(title_tokens)),
                "pred_match_idx": best_idx,
                "pred_match_start": pred_start,
                "pred_match_end": pred_end,
                "time_diff": float(td),
                "best_prob": float(cand_scores[best_idx]),
                "jacc_title_seg": float(jacc),
                "has_inaudible": int(has_inaud),
                "seg_text_snip": seg_text[:300],
                "pred_text_snip": str(candidate_texts[best_idx])[:300],
            }
            for tol in tol_list:
                rec[f"ok_{tol}s"] = int(td <= float(tol))
            rows.append(rec)
    return pd.DataFrame(rows)


def alignment_ci_from_log(df_log: pd.DataFrame, tol_s: int, seed: int, n_boot: int = 2000) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    out = {}
    for method, g in df_log.groupby("method"):
        vals = g[f"ok_{tol_s}s"].astype(int).to_numpy()
        m = float(vals.mean()) if len(vals) else 0.0
        if len(vals) < 2:
            out[method] = {"mean": m, "ci95": [m, m], "n": int(len(vals))}
            continue
        means = []
        n = len(vals)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            means.append(float(vals[idx].mean()))
        lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
        out[method] = {"mean": m, "ci95": [lo, hi], "n": int(len(vals))}
    return out


def tolerance_sensitivity(df_log: pd.DataFrame, tols: List[int]) -> pd.DataFrame:
    rows = []
    for method, g in df_log.groupby("method"):
        for tol in tols:
            v = g[f"ok_{tol}s"].astype(int).to_numpy()
            rows.append({"method": method, "tol_s": int(tol), "score": float(v.mean()) if len(v) else 0.0, "n": int(len(v))})
    return pd.DataFrame(rows)


def sample_failed_cases(df_log: pd.DataFrame, tol_fail: int, n_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fail = df_log[df_log[f"ok_{tol_fail}s"] == 0].copy()
    if fail.empty:
        return fail
    methods = sorted(fail["method"].unique().tolist())
    per = max(1, n_total // max(1, len(methods)))
    samples = []
    for m in methods:
        fm = fail[fail["method"] == m]
        if fm.empty:
            continue
        take = min(per, len(fm))
        idx = rng.choice(len(fm), size=take, replace=False)
        samples.append(fm.iloc[idx])
    out = pd.concat(samples, ignore_index=True)
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=seed).reset_index(drop=True)
    if len(out) < n_total and len(fail) > len(out):
        remaining = fail.drop(out.index, errors="ignore")
        take = min(n_total - len(out), len(remaining))
        if take > 0:
            out = pd.concat([out, remaining.sample(n=take, random_state=seed)], ignore_index=True)
    out = out.reset_index(drop=True)

    out["Short Segment"] = (out["seg_words"] <= 30).astype(int)
    out["Long Segment"] = (out["seg_words"] >= 250).astype(int)
    out["Short Title"] = (out["title_words"] <= 3).astype(int)
    out["Low Title-Segment Overlap"] = (out["jacc_title_seg"] <= 0.02).astype(int)
    out["manual_tag_primary"] = ""
    out["manual_note"] = ""
    return out


def summarize_error_tags(df_sheet: pd.DataFrame) -> pd.DataFrame:
    TAG_COLS = [
        "Short Segment",
        "Long Segment",
        "Short Title",
        "Low Title-Segment Overlap",
    ]
    tag_cols = [c for c in TAG_COLS if c in df_sheet.columns]

    rows = []
    for method, g in df_sheet.groupby("method"):
        for tc in tag_cols:
            v = pd.to_numeric(g[tc], errors="coerce").fillna(0).astype(int).to_numpy()
            rows.append(
                {
                    "method": method,
                    "tag": tc,
                    "rate": float(v.mean()) if len(v) else 0.0,
                    "n": int(len(v)),
                }
            )

    return pd.DataFrame(rows, columns=["method", "tag", "rate", "n"])



def plot_alignment_bars(ci_15: Dict[str, Any], ci_30: Dict[str, Any], out_png: str) -> None:
    methods = sorted(set(list(ci_15.keys()) + list(ci_30.keys())))
    x = np.arange(len(methods))
    w = 0.35


    m15 = [ci_15[m]["mean"] for m in methods]
    e15 = [(ci_15[m]["mean"] - ci_15[m]["ci95"][0], ci_15[m]["ci95"][1] - ci_15[m]["mean"]) for m in methods]
    lo15 = [a for a, b in e15]
    hi15 = [b for a, b in e15]

    m30 = [ci_30[m]["mean"] for m in methods]
    e30 = [(ci_30[m]["mean"] - ci_30[m]["ci95"][0], ci_30[m]["ci95"][1] - ci_30[m]["mean"]) for m in methods]
    lo30 = [a for a, b in e30]
    hi30 = [b for a, b in e30]


    plt.bar(
        x - w / 2, m15, width=w,
        yerr=[lo15, hi15], capsize=3,
        color=COLOR_MAIN, edgecolor="black",
        error_kw={"ecolor": COLOR_ERR, "elinewidth": 1, "capthick": 1},
        label=r"$\pm 15$s",
    )
    plt.bar(
        x + w / 2, m30, width=w,
        yerr=[lo30, hi30], capsize=3,
        color=COLOR_AUX, edgecolor="black",
        error_kw={"ecolor": COLOR_ERR, "elinewidth": 1, "capthick": 1},
        label=r"$\pm 30$s",
    )
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.ylabel("Self-retrieval score")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_sensitivity(df_sens: pd.DataFrame, out_png: str) -> None:

    plt.figure(figsize=(7.0, 3.2))
    methods = sorted(df_sens["method"].unique().tolist())

    for i, method in enumerate(methods):
        g = df_sens[df_sens["method"] == method].sort_values("tol_s")
        c = COLOR_MAIN if i == 0 else COLOR_AUX
        plt.plot(g["tol_s"].tolist(), g["score"].tolist(), marker="o", color=c, label=method)

    plt.xlabel("Tolerance (s)")
    plt.ylabel("Self-retrieval score")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_error_tag_rates(df_tag: pd.DataFrame, out_png: str) -> None:

    tags = sorted(df_tag["tag"].unique().tolist())
    methods = sorted(df_tag["method"].unique().tolist())
    x = np.arange(len(tags))
    w = 0.35

    plt.figure(figsize=(7.0, 3.6))
    for i, m in enumerate(methods):
        gm = df_tag[df_tag["method"] == m].set_index("tag").reindex(tags)
        vals = gm["rate"].fillna(0.0).to_numpy()
        c = COLOR_MAIN if i == 0 else COLOR_AUX
        plt.bar(
            x + (i - (len(methods) - 1) / 2) * w,
            vals,
            width=w,
            color=c,
            edgecolor="black",
            label=m,
        )

    plt.xticks(x, tags, rotation=20, ha="right")
    plt.ylabel("Rate among failures")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_summarization_metrics(metrics_by_method: Dict[str, Any], out_png: str) -> None:

    methods = list(metrics_by_method.keys())
    cols = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]
    labels = ["R-1", "R-2", "R-L", "BERT-F1"]
    x = np.arange(len(cols))
    w = 0.35

    plt.figure(figsize=(7.0, 3.2))
    for i, m in enumerate(methods):
        md = metrics_by_method[m]
        means, lo, hi = [], [], []
        for c in cols:
            mean = float(md.get(c, {}).get("mean", 0.0))
            ci = md.get(c, {}).get("ci95", [mean, mean])
            means.append(mean)
            lo.append(mean - float(ci[0]))
            hi.append(float(ci[1]) - mean)

        color = COLOR_MAIN if i == 0 else COLOR_AUX
        plt.bar(
            x + (i - (len(methods) - 1) / 2) * w,
            means,
            width=w,
            yerr=[lo, hi],
            capsize=3,
            color=color,
            edgecolor="black",
            error_kw={"ecolor": COLOR_ERR, "elinewidth": 1, "capthick": 1},
            label=m,
        )

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def paired_ttest_alignment_from_log(df_log: pd.DataFrame, method_a: str, method_b: str, ok_col: str):

    df = df_log.copy()

    if ok_col not in df.columns:
        raise ValueError(f"{ok_col} not found. Available: {list(df.columns)}")

    df[ok_col] = df[ok_col].astype(int)

    a = (df[df["method"] == method_a]
         .groupby("video_id")[ok_col].mean()
         .rename("a"))
    b = (df[df["method"] == method_b]
         .groupby("video_id")[ok_col].mean()
         .rename("b"))

    joined = pd.concat([a, b], axis=1).dropna()
    x = joined["a"].to_numpy()
    y = joined["b"].to_numpy()

    if len(x) < 2:
        raise ValueError(f"Not enough paired videos (n={len(x)}).")

    t, p = ttest_rel(x, y)
    diff = x - y
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)

    return {
        "n_videos": int(len(joined)),
        "ok_col": ok_col,
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff_a_minus_b": float(diff.mean()),
        "t": float(t),
        "p": float(p),
        "cohen_d_paired": float(d),
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="lucas-ventura/chapter-llama")
    parser.add_argument("--chapters_json", type=str, default="docs/subset_data/chapters/chapters_sml1k_train.json")
    parser.add_argument("--asrs_json", type=str, default="docs/subset_data/asrs/asrs_sml1k_train.json")

    parser.add_argument("--output_dir", type=str, default="outputs_error_analysis")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--eval_videos", type=int, default=50)

    parser.add_argument("--fixed_k", type=int, default=6)
    parser.add_argument("--k_model_path", type=str, default="models/segment_num_predictor.joblib")
    parser.add_argument("--train_k_predict", action="store_true")

    parser.add_argument("--sbert_ckpt", type=str, default="all-MiniLM-L6-v2")

    parser.add_argument("--bart_out", type=str, default="models/vidchapter_bart_best_local")
    parser.add_argument("--train_bart", action="store_true")
    parser.add_argument("--bart_epochs", type=int, default=1)
    parser.add_argument("--bart_lr", type=float, default=3e-5)
    parser.add_argument("--bart_bs", type=int, default=2)
    parser.add_argument("--bart_max_in", type=int, default=512)
    parser.add_argument("--bart_max_out", type=int, default=64)
    parser.add_argument("--bart_beams", type=int, default=4)
    parser.add_argument("--bart_max_train_samples", type=int, default=0)

    parser.add_argument("--align_out", type=str, default="models/vidchapter_alignment_local")
    parser.add_argument("--train_align", action="store_true")
    parser.add_argument("--align_epochs", type=int, default=1)
    parser.add_argument("--align_lr", type=float, default=2e-5)
    parser.add_argument("--align_bs", type=int, default=16)
    parser.add_argument("--align_neg", type=int, default=3)
    parser.add_argument("--align_max_train_pairs", type=int, default=0)

    parser.add_argument("--gen_bs", type=int, default=4)

    parser.add_argument("--error_tol_fail", type=int, default=15)
    parser.add_argument("--error_n", type=int, default=100)

    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()

    if args.smoke_test:
        args.max_videos = 220
        args.eval_videos = 30
        args.bart_epochs = 1
        args.align_epochs = 1
        args.bart_max_in = min(args.bart_max_in, 256)
        args.bart_bs = min(args.bart_bs, 2)
        args.gen_bs = min(args.gen_bs, 2)
        args.align_bs = min(args.align_bs, 8)
        args.bart_max_train_samples = 800
        args.align_max_train_pairs = 8000

    chapters = load_json_from_hf(args.repo_id, args.chapters_json)
    asrs = load_json_from_hf(args.repo_id, args.asrs_json)

    video_ids = sorted(list(set(chapters.keys()) & set(asrs.keys())))
    if args.max_videos and args.max_videos > 0:
        video_ids = video_ids[: args.max_videos]

    rng = random.Random(args.seed)
    rng.shuffle(video_ids)

    split_idx = int(len(video_ids) * float(args.train_ratio))
    train_vids = video_ids[:split_idx]
    test_vids = video_ids[split_idx:]
    if args.eval_videos and args.eval_videos > 0:
        test_vids = test_vids[: args.eval_videos]

    X, y = extract_k_features(chapters, asrs, train_vids)
    if args.train_k_predict or (not os.path.exists(args.k_model_path)):
        k_model, k_metrics = train_k_predictor_once(X, y, out_path=args.k_model_path, seed=args.seed)
        with open(os.path.join(args.output_dir, "k_predict_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(k_metrics, f, indent=2)
    else:
        k_model = joblib.load(args.k_model_path)
        with open(os.path.join(args.output_dir, "k_predict_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"loaded": True}, f, indent=2)

    sbert_device = device if device in {"cuda", "mps"} else "cpu"
    sbert = SentenceTransformer(args.sbert_ckpt, device=sbert_device)

    gt_train = build_ground_truth_chapter_dataset(chapters, asrs, train_vids)

    bart_dir = resolve_hf_model_dir(args.bart_out)
    if bart_dir is None:
        if args.train_bart or True:
            bart_dir = fine_tune_bart_model(
                gt_dataset=gt_train,
                output_dir=args.bart_out,
                seed=args.seed,
                epochs=args.bart_epochs,
                lr=args.bart_lr,
                batch_size=args.bart_bs,
                max_input_length=args.bart_max_in,
                max_target_length=args.bart_max_out,
                max_train_samples=args.bart_max_train_samples,
            )
        else:
            bart_dir = "facebook/bart-base"

    align_dir = resolve_hf_model_dir(args.align_out)
    if align_dir is None:
        if args.train_align or True:
            align_dir = fine_tune_alignment_model(
                gt_dataset=gt_train,
                output_dir=args.align_out,
                seed=args.seed,
                epochs=args.align_epochs,
                lr=args.align_lr,
                batch_size=args.align_bs,
                num_negatives=args.align_neg,
                max_train_pairs=args.align_max_train_pairs,
            )
        else:
            align_dir = "bert-base-uncased"

    bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_dir).to(device)
    bart_tokenizer = AutoTokenizer.from_pretrained(bart_dir)

    align_model = AutoModelForSequenceClassification.from_pretrained(align_dir, num_labels=2).to(device)
    align_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds_fixed = build_segmented_dataset_for_eval(
        chapters_data=chapters,
        asrs_data=asrs,
        video_ids=test_vids,
        method="fixed",
        fixed_k=args.fixed_k,
        k_model=k_model,
        sbert_model=sbert,
    )
    ds_kpred = build_segmented_dataset_for_eval(
        chapters_data=chapters,
        asrs_data=asrs,
        video_ids=test_vids,
        method="kpred",
        fixed_k=args.fixed_k,
        k_model=k_model,
        sbert_model=sbert,
    )

    preds_fixed = generate_titles(
        ds=ds_fixed,
        model=bart_model,
        tokenizer=bart_tokenizer,
        device=device,
        max_input_length=args.bart_max_in,
        gen_max_len=args.bart_max_out,
        num_beams=args.bart_beams,
        batch_size=args.gen_bs,
    )
    preds_kpred = generate_titles(
        ds=ds_kpred,
        model=bart_model,
        tokenizer=bart_tokenizer,
        device=device,
        max_input_length=args.bart_max_in,
        gen_max_len=args.bart_max_out,
        num_beams=args.bart_beams,
        batch_size=args.gen_bs,
    )

    refs_fixed = [x["title"] for x in ds_fixed]
    refs_kpred = [x["title"] for x in ds_kpred]

    summ_fixed = compute_summarization_metrics_with_ci(preds_fixed, refs_fixed, seed=args.seed)
    summ_kpred = compute_summarization_metrics_with_ci(preds_kpred, refs_kpred, seed=args.seed + 1)

    summ_all = {"Fixed-Window": summ_fixed, "K-Prediction": summ_kpred}
    with open(os.path.join(args.output_dir, "summarization_metrics_ci.json"), "w", encoding="utf-8") as f:
        json.dump(summ_all, f, indent=2)

    plot_summarization_metrics(summ_all, os.path.join(plots_dir, "summarization_metrics_ci.png"))

    tol_list = [5, 10, 15, 20, 30]

    log_fixed = build_self_retrieval_alignment_log(
        ds=ds_fixed,
        preds=preds_fixed,
        method_name="Fixed-Window",
        align_model=align_model,
        align_tokenizer=align_tokenizer,
        device=device,
        tol_list=tol_list,
    )
    log_kpred = build_self_retrieval_alignment_log(
        ds=ds_kpred,
        preds=preds_kpred,
        method_name="K-Prediction",
        align_model=align_model,
        align_tokenizer=align_tokenizer,
        device=device,
        tol_list=tol_list,
    )

    log_all = pd.concat([log_fixed, log_kpred], ignore_index=True)
    log_path = os.path.join(args.output_dir, "alignment_self_retrieval_log.csv")
    log_all.to_csv(log_path, index=False)

    method_a = "Fixed-Window"
    method_b = "K-Prediction"

    print("Rows:", len(log_all))
    print("Methods:", log_all["method"].astype(str).str.strip().value_counts().head(20))
    print("Num unique videos:", log_all["video_id"].nunique())

    tt_15 = paired_ttest_alignment_from_log(log_all, method_a, method_b, ok_col="ok_15s")
    tt_30 = paired_ttest_alignment_from_log(log_all, method_a, method_b, ok_col="ok_30s")

    with open(os.path.join(args.output_dir, "paired_ttest_alignment.json"), "w", encoding="utf-8") as f:
        json.dump({"ttest_15s": tt_15, "ttest_30s": tt_30}, f, indent=2)

    print("[Paired t-test] saved to:", os.path.join(args.output_dir, "paired_ttest_alignment.json"))
    print("[Paired t-test] 15s:", tt_15)
    print("[Paired t-test] 30s:", tt_30)

    ci_15 = alignment_ci_from_log(log_all, tol_s=15, seed=args.seed)
    ci_30 = alignment_ci_from_log(log_all, tol_s=30, seed=args.seed + 7)

    with open(os.path.join(args.output_dir, "alignment_ci_15s.json"), "w", encoding="utf-8") as f:
        json.dump(ci_15, f, indent=2)
    with open(os.path.join(args.output_dir, "alignment_ci_30s.json"), "w", encoding="utf-8") as f:
        json.dump(ci_30, f, indent=2)

    plot_alignment_bars(ci_15, ci_30, os.path.join(plots_dir, "alignment_f1_ci.png"))

    df_sens = tolerance_sensitivity(log_all, tols=tol_list)
    df_sens.to_csv(os.path.join(args.output_dir, "alignment_sensitivity.csv"), index=False)
    plot_sensitivity(df_sens, os.path.join(plots_dir, "alignment_sensitivity.png"))

    sheet = sample_failed_cases(log_all, tol_fail=args.error_tol_fail, n_total=args.error_n, seed=args.seed)
    sheet_path = os.path.join(args.output_dir, f"error_analysis_failed_{args.error_n}_tol{args.error_tol_fail}.csv")
    sheet.to_csv(sheet_path, index=False)

    if not sheet.empty:
        df_tag = summarize_error_tags(sheet)
        tag_csv = os.path.join(args.output_dir, "error_tag_summary.csv")
        df_tag.to_csv(tag_csv, index=False)
        plot_error_tag_rates(df_tag, os.path.join(plots_dir, "error_tag_rates.png"))

    print("Done.")
    print("Saved:")
    print(" -", os.path.join(args.output_dir, "summarization_metrics_ci.json"))
    print(" -", os.path.join(args.output_dir, "alignment_ci_15s.json"))
    print(" -", os.path.join(args.output_dir, "alignment_ci_30s.json"))
    print(" -", log_path)
    print(" -", sheet_path)
    print("Plots:")
    print(" -", os.path.join(plots_dir, "summarization_metrics_ci.png"))
    print(" -", os.path.join(plots_dir, "alignment_f1_ci.png"))
    print(" -", os.path.join(plots_dir, "alignment_sensitivity.png"))
    if not sheet.empty:
        print(" -", os.path.join(plots_dir, "error_tag_rates.png"))


if __name__ == "__main__":
    main()
