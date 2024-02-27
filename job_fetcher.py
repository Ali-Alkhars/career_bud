from dotenv import load_dotenv
import os
import requests

"""
This script has helper functions that use the Adzuna API (https://developer.adzuna.com/activedocs)
to fetch some available job offers.
"""

# Define the API call parameters
load_dotenv()
APP_ID = os.getenv('APP_ID')
APP_KEY = os.getenv('APP_KEY')
MAX_DAYS = '30'
COUNTRY = 'UK'
SORT = 'relevance'

def fetch_jobs():
    """
    Make a GET HTTP to the Adzuna API to fetch some job offers for the user.
    Call helper function to display the results.
    """
    FIELD = input("\nPlease specify a field for the job search, e.g. Software Engineering: ")
    print(f'\nSearching for jobs in {FIELD}...')
    API_URL = f'https://api.adzuna.com/v1/api/jobs/gb/search/1?app_id={APP_ID}&app_key={APP_KEY}&what={FIELD}&location0={COUNTRY}&max_days_old={MAX_DAYS}&sort_by={SORT}'

    # Make the GET request
    response = requests.get(API_URL)

    try:
        if response.status_code == 200:
            data = response.json()
            show_jobs(data['results'])
        else:
            print(f"\nFailed to retrieve data, status code: {response.status_code}\nI'm very sorry, maybe try again?")
    except Exception as e:
            print(e)
            print(f"\nI'm having some difficulties with retrieving job offers. I'm very sorry, maybe try again?")

def show_jobs(results):
    """Handle the logic of showing the user a specific number of results"""
    print(f'\nI found some potential jobs for you!')
    current_job = 0
    count = len(results)

    # Show first five results (or all if < 5)
    while current_job < count and current_job < 5:
        show_job(current_job+1, results[current_job])
        current_job += 1

    # Show more results at the user's request
    more = 'y'
    while more != 'n' and current_job < count:
        more = input("\nWould you like to see more offers? Enter 'y' or 'n': ").lower()
        if more != 'n':
            show_job(current_job+1, results[current_job])
            current_job += 1

def show_job(count, data):
    """Display a specific job offer to the user"""
    print("\n-------------------------------------------")
    print(f"\nJob {count}: {data['title']} at {data['company']['display_name']}")
    print(f"\nDescription: {data['description']}")
    print(f"\nPredicted salary: £{int(data['salary_min'])}")
    print(f"\nApply here: {data['redirect_url']}")
    print(f"\nJob post date: {data['created'].split('T')[0]}")