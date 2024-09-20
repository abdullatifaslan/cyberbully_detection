import pickle

import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score


def train_lightgbm(X_train, X_test, y_train, y_test, params, n_estimators):
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(params, train_data, num_boost_round=n_estimators, valid_sets=[test_data])

    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    accuracy = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    return model


def classify_messages(model, vectorizer, messages):
    X = vectorizer.transform(messages['Text']).toarray()

    y_pred = model.predict(X)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    messages['Predicted'] = y_pred_binary
    return messages


def load_model():
    try:
        with open('model/lightgbm_model.pkl', 'rb') as f:
            pickle.load(f)
    except FileNotFoundError as file_error:
        print(file_error)
    except Exception as error:
        print(error)


def save_model(model):
    try:
        with open('model/lightgbm_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    except Exception as error:
        print(error)
