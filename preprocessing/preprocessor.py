import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def preprocessor_sleep_wake_detection(steps_df, heart_rate_df, activity_level_calories_df, sleep_df):
    merged_df = pd.merge(steps_df, heart_rate_df, on=['created_time', 'created_date'], how='outer')
    merged_df = pd.merge(merged_df, activity_level_calories_df, on=['created_time', 'created_date'], how='outer')
    merged_df = pd.merge(merged_df, sleep_df, on=['created_time', 'created_date'], how='outer')

    final_df = merged_df[['created_time', 'created_date', 'sleep_level', 'calories_burned', 'activity_level', 'heart_rate_per_minute', 'steps_per_minute']]

    final_df.loc[final_df['activity_level'] > 1, 'sleep_level'] = 'awake'
    final_df.loc[final_df['steps_per_minute'] > 1, 'sleep_level'] = 'awake'

    final_df.to_csv('./csv/final_output.csv', index=False)

    final_df['sleep_level'] = final_df['sleep_level'].replace('rem', 'asleep')
    final_df['sleep_level'] = final_df['sleep_level'].replace('deep', 'asleep')
    final_df['sleep_level'] = final_df['sleep_level'].replace('light', 'asleep')

    final_df['sleep_level'] = final_df['sleep_level'].replace('wake', 'awake')
    final_df['sleep_level'] = final_df['sleep_level'].replace('restless', 'awake')

    final_df.replace('', np.nan, inplace=True)
    final_df.dropna(inplace=True)

    final_df.sort_values(by=['created_date', 'created_time'])

    final_df.to_csv('./csv/final_output_swd.csv', index=False)

    print(final_df["sleep_level"].value_counts())

    print(final_df.isna().sum())

    label_encoder = LabelEncoder()

    final_df['sleep_level'] = label_encoder.fit_transform(final_df['sleep_level'])

    y_encoded, X = final_df['sleep_level'], final_df.drop(['created_time', 'created_date', 'sleep_level'], axis=1)

    return X, y_encoded, label_encoder.classes_

def preprocessor_sleep_stage_classification(steps_df, heart_rate_df, activity_level_calories_df, sleep_df):
    merged_df = pd.merge(steps_df, heart_rate_df, on=['created_time', 'created_date'], how='outer')
    merged_df = pd.merge(merged_df, activity_level_calories_df, on=['created_time', 'created_date'], how='outer')
    merged_df = pd.merge(merged_df, sleep_df, on=['created_time', 'created_date'], how='outer')

    final_df_x = merged_df[['created_time', 'created_date', 'sleep_level', 'calories_burned', 'activity_level', 'heart_rate_per_minute', 'steps_per_minute']]
    final_df_x.to_csv('./csv/final_output.csv', index=False)
    final_df = final_df_x[final_df_x['sleep_level'].isin(['light', 'rem', 'deep'])].copy()

    final_df.replace('', np.nan, inplace=True)
    final_df.dropna(inplace=True)

    final_df.sort_values(by=['created_date', 'created_time'])

    final_df.to_csv('./csv/final_output_ssc.csv', index=False)

    print(final_df["sleep_level"].value_counts())

    print(final_df.isna().sum())

    label_encoder = LabelEncoder()

    final_df['sleep_level'] = label_encoder.fit_transform(final_df['sleep_level'])

    y_encoded, X = final_df['sleep_level'], final_df.drop(['created_time', 'created_date', 'sleep_level'], axis=1)

    return X, y_encoded, label_encoder.classes_


def split_train_test_set(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def oversampling_with_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(Counter(y_resampled))
    return X_resampled, y_resampled

def under_sampling(X, y):
    undersample_strategy = {0: 374, 1: 374, 2: 374}
    rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(Counter(y_resampled))
    return X_resampled, y_resampled
