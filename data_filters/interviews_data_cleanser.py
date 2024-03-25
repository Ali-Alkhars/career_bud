import random
import json
import re
import csv

"""
This script is intended to structure specific data files into 
a Json dataset that will be used to train Language Models.
"""


def question_maker(topic="software engineering"):
    """
    Return a random question from a pre-made list of questions
    given a specific topic.
    """
    questions = [
        f'Can you give me interview questions about {topic}?',
        f'Prepare me for an interview for a software engineering role',
        f'I want interview questions about {topic}',
        f'Prepare me for an interview on {topic}',
        f'Could you question me about {topic}?',
        f'I need software engineering interview questions',
        f'I need programming interview questions',
        f'I need {topic} interview questions',
        f'Prepare me for an interview',
        f'What are common interview questions for {topic}?',
        f'Could we do a mock interview?',
        f'Provide me with interview questions about {topic}',
    ]
    return random.choice(questions)

def process_questions(input_file, output_file, regex, topic):
    """
    Extract the questions of a data file that are structured
    in an undesirable way, use a given regular expression 
    to get the correct structure to the JSON dataset.
    """
    # Regular expression to match the new question pattern
    pattern = re.compile(regex)

    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Extract questions from each line
    questions = []
    for line in lines:
        match = pattern.match(line)
        if match:
            question_text = match.group(1).strip()
            questions.append({
                "input": question_maker(topic),
                "response": question_text
            })

    # Append to the existing JSON file
    with open(output_file, 'r+') as file:
        data = json.load(file)
        data.extend(questions)
        file.seek(0)
        json.dump(data, file, indent=4)

def process_csv_questions(input_file, output_file):
    """
    Extract the questions of a data file that are structured
    as a csv to get the correct structure to the JSON dataset.
    """
    questions = []
    with open(input_file, newline='', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions.append({
                "input": row['Question'],
                "response": row['Answer']
            })
    
    # Append to the existing JSON file
    with open(output_file, 'r+') as file:
        data = json.load(file)
        data.extend(questions)
        file.seek(0)
        file.truncate()  # Clear the file before writing the updated data
        json.dump(data, file, indent=4)
