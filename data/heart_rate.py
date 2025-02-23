import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from utilities.dates import get_dates

load_dotenv()

dates = get_dates()
url = os.getenv('heart_rate_url')
fitbit_id = os.getenv('fitbit_id')
authorization_token = os.getenv('authorization_token')

mongo_uri = os.getenv('MONGO_URI')
db_name = os.getenv('MONGO_DB_NAME')
collection_name = os.getenv('MONGO_COLLECTION_NAME_HEART_RATE')

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

def fetch_heart_rate_data_from_api():
    for d in dates:
        heart_rate_url = url.replace('{fitbit_id}', fitbit_id).replace('{date}', d)
        headers = {
            'Accept': 'application/json',
            "Authorization": f"Bearer {authorization_token}"
        }
        response = requests.get(heart_rate_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            persist_data(data)
        else:
            print(f"Failed to fetch data: {response.status_code}")
    print("Data inserted into MongoDB")

def persist_data(data):
    date = data["activities-heart"][0]["dateTime"]
    dataset = data['activities-heart-intraday']['dataset']

    if len(dataset) == 0:
        return

    documents = []
    for record in dataset:
        time_str = record['time']
        value = record['value']

        hour, minute, _ = time_str.split(':')

        document = {
            'created_date': date,
            'heart_rate_per_minute': value,
            'created_time': f"{hour}:{minute}"
        }

        documents.append(document)

    collection.insert_many(documents)

def get_heart_rate_data_and_convert_to_df():
    data = list(collection.find({}, {'_id': 0, 'heart_rate_per_minute': 1, 'created_time': 1, 'created_date': 1}))
    heart_rate_df = pd.DataFrame(data)
    return heart_rate_df
