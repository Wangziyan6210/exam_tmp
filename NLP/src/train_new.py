import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


def main():
    os.makedirs("./models", exist_ok=True)

    dataset = load_dataset("imdb", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("./models")
    trainer.tokenizer.save_pretrained("./models")


if __name__ == "__main__":
    main()
