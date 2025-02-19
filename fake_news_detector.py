from transformers import pipeline

def detect_fake_news(text):
    """Detects if the input news text is fake or real using zero-shot classification."""
    classifier = pipeline("zero-shot-classification")
    candidate_labels = ["fake", "real"]
    result = classifier(text, candidate_labels)
    return result

if __name__ == "__main__":
    news_text = input("Enter the news article text: ")
    result = detect_fake_news(news_text)
    print("\nFake News Detection Result:")
    print(result)
