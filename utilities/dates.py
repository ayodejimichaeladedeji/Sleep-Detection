from datetime import datetime, timedelta

def get_dates():
    start_date = datetime.strptime('2024-06-25', '%Y-%m-%d')
    end_date = datetime.strptime('2024-08-08', '%Y-%m-%d')

    date_list = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

    return date_list