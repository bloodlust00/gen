from transformers import pipeline

sentiment_analyzer = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result

def main():
    sentences = [
        "I love this product! It's amazing.",
        "The service was terrible and I'm very disappointed.",
        "The movie was okay, not great but not bad either.",
        "This is the best day of my life!",
        "I feel so frustrated with this situation."
    ]
    for sentence in sentences:
        sentiment_result = analyze_sentiment(sentence)
        print(f"Sentence: {sentence}")
        print(f"Sentiment: {sentiment_result[0]['label']} (confidence: {sentiment_result[0]['score']:.4f})")
        print("-" * 60)

main()
