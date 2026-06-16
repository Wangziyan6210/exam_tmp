import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import BertTokenizer, BertForSequenceClassification


def predict(text, model, tokenizer, device, max_length=512):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1).item()

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = probabilities[0][prediction].item()
    return sentiment, confidence


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("./models")
    model = BertForSequenceClassification.from_pretrained("./models").to(device)

    test_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible waste of time. Poor plot and bad acting.",
        "It was okay, nothing special but not terrible either.",
        "One of the best films I have ever seen. Brilliant direction!",
        "The movie started well but became boring halfway through.",
    ]

    print("=== Inference Results ===")
    for review in test_reviews:
        sentiment, confidence = predict(review, model, tokenizer, device)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()
