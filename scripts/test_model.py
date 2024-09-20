from scripts.fetch_data import fetch_telegram_messages
from scripts.train_model import train_model
from scripts.model_utils import classify_messages


def test_model():
    model, vectorizer = train_model()

    df = fetch_telegram_messages(limit=100)

    classified_messages = classify_messages(model, vectorizer, df)

    print(classified_messages)


if __name__ == "__main__":
    test_model()
