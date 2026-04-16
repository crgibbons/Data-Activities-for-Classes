from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I love how easy this setup is.",
    "This repository still needs some work."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(text)
    print(result)
    print()
