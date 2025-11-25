# src/models/build_model.py
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

def build_model_and_tokenizer(model_name, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=cfg)
    return model, tokenizer