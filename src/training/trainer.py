# src/training/trainer.py
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os

def build_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    train_args_cfg = config["training_args"]
    output_dir = train_args_cfg.get("output_dir", "saved_models")
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=train_args_cfg.get("evaluation_strategy", "steps"),
        per_device_train_batch_size=train_args_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_args_cfg.get("per_device_eval_batch_size", 4),
        predict_with_generate=True,
        logging_steps=train_args_cfg.get("logging_steps", 100),
        save_steps=train_args_cfg.get("save_steps", 500),
        eval_steps=train_args_cfg.get("eval_steps", 500),
        save_total_limit=train_args_cfg.get("save_total_limit", 3),
        fp16=train_args_cfg.get("fp16", True),
        num_train_epochs=train_args_cfg.get("num_train_epochs", 3),
        learning_rate=train_args_cfg.get("learning_rate", 5e-5),
        weight_decay=train_args_cfg.get("weight_decay", 0.0),
        warmup_steps=train_args_cfg.get("warmup_steps", 0),
        seed=train_args_cfg.get("seed", 42),
        push_to_hub=False,
        load_best_model_at_end=train_args_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_args_cfg.get("metric_for_best_model", "eval_rougeL"),
        greater_is_better=train_args_cfg.get("greater_is_better", True),
        gradient_accumulation_steps=train_args_cfg.get("gradient_accumulation_steps", 1),
        save_strategy=train_args_cfg.get("save_strategy", "steps"),
    )

    from src.training.collator import get_data_collator
    data_collator = get_data_collator(tokenizer, model)

    from src.evaluation.metrics import compute_metrics
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    return trainer
