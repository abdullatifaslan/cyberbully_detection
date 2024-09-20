from sklearn.model_selection import train_test_split
from config.config import LIGHTGBM_PARAMS, N_ESTIMATORS, TRAIN_TEST_SPLIT, RANDOM_STATE
from scripts.model_utils import train_lightgbm, save_model
from scripts.preprocess_data import preprocess_data
from scripts.preprocess_data import load_datasets


def train_model():
    df = load_datasets()
    X, y, vectorizer = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)

    model = train_lightgbm(X_train, X_test, y_train, y_test, LIGHTGBM_PARAMS, N_ESTIMATORS)
    save_model(model)

    return model, vectorizer

if __name__ == '__main__':
    train_model()