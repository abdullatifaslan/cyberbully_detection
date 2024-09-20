import os
import lightgbm as lgb
from scripts.fetch_data import fetch_telegram_messages
from scripts.model_utils import classify_messages, load_model
from scripts.preprocess_data import preprocess_data
from scripts.test_model import test_model


def main():
    if os.path.exists('model/lightgbm.pkl'):
        loaded_model = lgb.Booster(model_file=load_model())

        df = fetch_telegram_messages(limit=100)
        _, _, vectorizer = preprocess_data(df)
        classified_messages = classify_messages(loaded_model, vectorizer, df)
        print(classified_messages)

    test_model()


if __name__ == '__main__':
    main()
