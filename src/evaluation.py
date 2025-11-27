from rouge_score import rouge_scorer
from bert_score import score

def compute_rouge(predictions, references):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L for a list of predictions vs references.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    
    results = []
    for pred, ref in zip(predictions, references):
        results.append(scorer.score(ref, pred))
    return results

def compute_bertscore(predictions, references, model="bert-base-uncased"):
    """
    Compute BERTScore Precision/Recall/F1.
    """
    P, R, F1 = score(predictions, references, lang="en", model_type=model)
    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist()
    }
