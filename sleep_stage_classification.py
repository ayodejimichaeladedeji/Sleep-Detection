from data.steps import get_steps_data_and_convert_to_df, fetch_steps_data_from_api
from data.heart_rate import get_heart_rate_data_and_convert_to_df, fetch_heart_rate_data_from_api
from data.sleep import get_sleep_data_and_convert_to_df, expand_sleep_data_from_mongo, fetch_sleep_data_from_api
from data.activity_level_calories import get_activity_level_calories_data_and_convert_to_df, fetch_activity_level_calories_data_from_api

from models.xgboost import xgboost
from models.random_forest import random_forest
from models.gradient_boost import gradient_boost
from models.voting_classifier import voting_classifier
from preprocessing.preprocessor import preprocessor_sleep_stage_classification, split_train_test_set, oversampling_with_smote


def fetch_oxygen_saturation_level_data_from_api():
    pass


if __name__ == '__main__':
    # fetch_steps_data_from_api()
    # fetch_heart_rate_data_from_api()
    # fetch_activity_level_calories_data_from_api()
    # fetch_sleep_data_from_api()
    # expand_sleep_data_from_mongo()

    steps_df = get_steps_data_and_convert_to_df()
    sleep_df = get_sleep_data_and_convert_to_df()
    heart_rate_df = get_heart_rate_data_and_convert_to_df()
    activity_level_calories_df = get_activity_level_calories_data_and_convert_to_df()

    X, y_encoded, label_encoder_classes = preprocessor_sleep_stage_classification(steps_df, heart_rate_df, activity_level_calories_df, sleep_df)

    X_resampled, y_resampled = oversampling_with_smote(X, y_encoded)

    X_train, X_test, y_train, y_test = split_train_test_set(X_resampled, y_resampled)

    random_forest(X_train, X_test, y_train, y_test, label_encoder_classes)
    gradient_boost(X_train, X_test, y_train, y_test, label_encoder_classes)
    voting_classifier(X_train, X_test, y_train, y_test, label_encoder_classes)
