from dotenv import load_dotenv
import os
import requests

# Using the Adzuna API: https://developer.adzuna.com/activedocs

load_dotenv()

APP_ID = os.getenv('APP_ID')
APP_KEY = os.getenv('APP_KEY')
MAX_DAYS = '30'
COUNTRY = 'UK'

def fetch_jobs():
    FIELD = input("\nPlease specify a field for the job search, e.g. Software Engineering: ")
    print(f'\nSearching for jobs in {FIELD}...')
    API_URL = f'https://api.adzuna.com/v1/api/jobs/gb/search/1?app_id={APP_ID}&app_key={APP_KEY}&what={FIELD}&location0={COUNTRY}&max_days_old={MAX_DAYS}&sort_by=date'

    # Make the GET request
    response = requests.get(API_URL)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the content of the response (HTML)
        print(f'\nHere is what I found:')
        print(response.text)
    else:
        print(f'Failed to retrieve content, status code: {response.status_code}')

fetch_jobs()