import pandas as pd
from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.preprocessing import LabelEncoder

import chardet

def ctgan_oversampling():
    # with open('./csv/final_output.csv', 'rb') as file:
    #     result = chardet.detect(file.read())
    #     print(result)

    # datasets = load_csvs(
    #     folder_name='csv/',
    #     read_csv_parameters={
    #         'skipinitialspace': True,
    #         'encoding': 'ascii'
    #     })
    #
    # data = datasets['final_output']

    metadata = SingleTableMetadata()

    # metadata.detect_from_csv(filepath='./csv/final_output.csv')

    final_output = pd.read_csv('./csv/final_output.csv')

    metadata.detect_from_dataframe(final_output)

    # metadata.save_to_json(filepath='my_metadata_v1.json')

    synthesizer = CTGANSynthesizer(
        metadata,  # required
        enforce_rounding=True,
        epochs=300,
        verbose=True,
        enforce_min_max_values=True
    )

    synthesizer.fit(final_output)

    synthetic_data = synthesizer.sample(
        num_rows=2_000,
        batch_size=1_000
    )

    initial_df = pd.read_csv('./csv/final_output.csv')
    balanced_df = pd.concat([initial_df, synthetic_data])

    # Check the distribution after oversampling
    print(balanced_df['sleep_level'].value_counts())

    label_encoder = LabelEncoder()

    balanced_df['sleep_level'] = label_encoder.fit_transform(balanced_df['sleep_level'])

    y_encoded, X = balanced_df['sleep_level'], balanced_df.drop(['created_time', 'created_date', 'sleep_level'], axis=1)

    return X, y_encoded, label_encoder.classes_