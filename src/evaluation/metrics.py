# src/evaluation/metrics.py
import evaluate
import numpy as np
from transformers import PreTrainedTokenizerBase

rouge = evaluate.load("rouge")

def postprocess_text(preds, refs):
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in refs]
    return preds, refs

def compute_metrics(eval_pred):
    """
    eval_pred: (preds, labels) from trainer.predict
    preds: numpy array of token ids or decoded strings depending on trainer
    """
    preds, labels = eval_pred
    # preds may be token ids -> decode if tokenizer present via closure (HF passes tokenizer)
    if isinstance(preds, np.ndarray):
        # trainer passes decoded strings usually; if not, user can decode before calling
        pass
    # Here we assume preds and labels are decoded strings lists
    decoded_preds = preds
    decoded_labels = labels

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # extract rouge-l fmeasure for metric_for_best_model expectation
    result = {k: round(v * 100, 4) for k, v in result.items()}
    # Add mean length
    result["gen_len"] = np.mean([len(p.split()) for p in decoded_preds])
    return result
