import json
import random

"""
This script is intended to structure a specific data file into 
a Json dataset that will be used to train Language Models to
generate a signal to search for real-time job offers.
"""

def response_maker():
    """Return a random response from a pre-made list of responses"""
    job_search_key = "[JOBSEARCH]" # The key used to train models to search for jobs
    responses = [
        f"Looking for jobs... {job_search_key}",
        f"I'll try to find some available jobs for you {job_search_key}",
        f"I can help you with finding jobs {job_search_key}",
        f"Here are some available positions {job_search_key}",
        f"Let me look for some jobs for you {job_search_key}",
        f"Finding an appropriate job could be difficult. Here are some offers {job_search_key}",
        f"Let's find you a fitting job! {job_search_key}",
        f"Finding jobs together should be fun! {job_search_key}",
        f"Let's do some job hunting {job_search_key}",
        f"Searching for appropriate job offers... {job_search_key}",
        f"Job hunting is often tricky, let's do it together! {job_search_key}",
        f"It would be my pleasure to help you apply for your next job! {job_search_key}",
        f"Let's kickstart your career! Looking up job offers... {job_search_key}",
    ]
    return random.choice(responses)

def process_data(input_file, output_file):
    """Extract inputs and outputs and structure them in a JSON file"""

    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    entries = []
    for input in lines:
        if '\n' in input:
            input = input.replace('\n', '')
        entries.append({
            "input": input,
            "response": response_maker()
        })

    # Append to the existing JSON file
    with open(output_file, 'r+') as file:
        data = json.load(file)
        data.extend(entries)
        file.seek(0)
        json.dump(data, file, indent=4)

process_data('../Unfiltered-datasets/job_prompts.txt', '../Fine-tuning Datasets/job_propmts_dataset.json')