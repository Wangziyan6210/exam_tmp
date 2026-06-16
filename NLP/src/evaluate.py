import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_imdb_data


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    report = classification_report(labels, predictions, target_names=["negative", "positive"], output_dict=True)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision_positive": report["positive"]["precision"],
        "recall_positive": report["positive"]["recall"],
        "f1_positive": report["positive"]["f1-score"],
    }


def main():
    train_dataset, test_dataset = load_imdb_data()

    model = BertForSequenceClassification.from_pretrained("./models")

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()

    os.makedirs("./results", exist_ok=True)
    with open("./results/eval_results.txt", "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results ===\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

    print("=== Evaluation Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    print("\nResults saved to ./results/eval_results.txt")


if __name__ == "__main__":
    main()
