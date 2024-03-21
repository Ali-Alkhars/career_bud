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
        elif response.status_code == 400 or response.status_code == 401:
            print(f"\nFailed to retrieve data, please make sure that you setup a '.env' file with the API's APP_ID and APP_KEY.\n")
        else:
            print(f"\nFailed to retrieve data, status code: {response.status_code}\nI'm very sorry, maybe try again?")
    except Exception as e:
            print(f"Exception: {e}")
            print(f"\nI'm having some difficulties with retrieving job offers. I'm very sorry, maybe try again?")

def show_jobs(results):
    """Handle the logic of showing the user a specific number of results"""
    current_job = 0
    count = len(results)

    if count > 0:
        print(f'\nI found some potential jobs for you!')
    else:
        print(f"\nSorry, I couldn't find jobs in your specified field!")

    # Show first three results (or all if < 3)
    while current_job < count and current_job < 3:
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
    # Retrieve data safely
    job_title = data.get('title', '[MISSING JOB TITLE]')
    company_display_name = data.get('company', {}).get('display_name', '[MISSING COMPANY NAME]')
    job_description = data.get('description', '[MISSING JOB DESCRIPTION]')
    salary = data.get('salary_min', -99999999)
    apply_url = data.get('redirect_url', '[MISSING APPLY URL]')
    post_date = data.get('created', '[MISSING DATE]')

    print("\n-------------------------------------------")
    print(f"\nJob {count}: {job_title} at {company_display_name}")
    print(f"\nDescription: {job_description}")
    print(f"\nPredicted salary: £{int(salary)}")
    print(f"\nApply here: {apply_url}")
    print(f"\nJob post date: {post_date.split('T')[0]}")