import nltk

def first_sentence_baseline(text):
    """
    Baseline 1: Return the first sentence as summary.
    """
    sentences = nltk.sent_tokenize(text)
    return sentences[0] if sentences else ""

def five_percent_baseline(text):
    """
    Baseline 2: Return the first 5% of words as summary.
    """
    words = text.split()
    num = max(1, int(len(words) * 0.05))
    return " ".join(words[:num])
