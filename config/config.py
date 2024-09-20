API_ID = '24806729'
API_HASH = '0ddb21093e4fadf8c20386ac164665df'
TELEPHONE_NUMBER = '+90 531 961 98 23'
CHANNEL_NAME = 'cyberbully_test'

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 700,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'verbose': -1,
    'min_child_samples': 30,
    'class_weight': 'balanced',
    'device': 'gpu'
}

TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 1000

USE_GPU = True
