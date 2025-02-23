import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from utilities.dates import get_dates
from datetime import datetime, timedelta

load_dotenv()

dates = get_dates()
fitbit_id = os.getenv('fitbit_id')
url = os.getenv('sleep_url')
authorization_token = os.getenv('authorization_token')

mongo_uri = os.getenv('MONGO_URI')
db_name = os.getenv('MONGO_DB_NAME')
sleep_collection_name = os.getenv('MONGO_COLLECTION_NAME_ACTIVITY_LEVEL_SLEEP')
expanded_sleep_collection_name = os.getenv('MONGO_COLLECTION_NAME_ACTIVITY_LEVEL_EXPANDED_SLEEP')

client = MongoClient(mongo_uri)
db = client[db_name]
sleep_collection = db[sleep_collection_name]
expanded_sleep_collection = db[expanded_sleep_collection_name]

def fetch_sleep_data_from_api():
    for d in dates:
        sleep_url = url.replace('{fitbit_id}', fitbit_id).replace('{date}', d)
        headers = {
            'Accept': 'application/json',
            "Authorization": f"Bearer {authorization_token}"
        }
        response = requests.get(sleep_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            persist_data(data)
        else:
            print(f"Failed to fetch data: {response.status_code}")
    print("Data inserted into MongoDB")

def persist_data(data):
    if len(data["sleep"]) == 0:
        return

    for sleep_data in data["sleep"]:
        date = sleep_data["dateOfSleep"]
        data = sleep_data["levels"]["data"]

        documents = []
        for record in data:
            time_from_split_date = record['dateTime'].split('T')
            sleep_level = record["level"]
            length = record["seconds"]

            hour, minute, _ = time_from_split_date[1].split(':')

            document = {
                'length(seconds)': length,
                'created_date': date,
                'sleep_level': sleep_level,
                'created_time': f"{hour}:{minute}"
            }

            documents.append(document)

        sleep_collection.insert_many(documents)

def expand_sleep_data_from_mongo():
    data = list(sleep_collection.find({}, {'_id': 0, 'length(seconds)': 1, 'sleep_level': 1, 'created_time': 1, 'created_date': 1}))
    sleep_data = []

    # Process each document
    for record in data:
        length_seconds = record['length(seconds)']
        sleep_level = record['sleep_level']
        created_time = record['created_time']
        created_date = record['created_date']

        start_time = datetime.strptime(f"{created_date} {created_time}", "%Y-%m-%d %H:%M")

        # Calculate the number of minutes
        minutes = length_seconds // 60

        for minute in range(minutes):
            new_entry_time = start_time + timedelta(minutes=minute)
            new_entry = {
                'sleep_level': sleep_level,
                'created_time': new_entry_time.strftime("%H:%M"),
                'created_date': created_date
            }
            sleep_data.append(new_entry)

    expanded_sleep_collection.insert_many(sleep_data)

def get_sleep_data_and_convert_to_df():
    data = list(expanded_sleep_collection.find({}, {'_id': 0, 'sleep_level': 1, 'created_time': 1, 'created_date': 1}))
    sleep_df = pd.DataFrame(data)
    # light_entries_count = sleep_df[sleep_df['sleep_level'] == 'light'].shape[0]
    # print(light_entries_count)
    return sleep_df
