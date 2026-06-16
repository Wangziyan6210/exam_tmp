from datasets import load_dataset
from transformers import BertTokenizer


def load_imdb_data():
    dataset = load_dataset("imdb", trust_remote_code=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    return train_dataset, test_dataset
