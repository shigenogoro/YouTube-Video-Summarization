# src/training/collator.py
from transformers import DataCollatorForSeq2Seq

def get_data_collator(tokenizer, model=None):
    return DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
