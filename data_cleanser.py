import random
import json
import re

"""
This script is intended to structure specific data files into 
a Json dataset that will be used to train Language Models.
"""


# Return a random question from a pre-made list of questions
# given a specific topic
def question_maker(topic="software engineering"):
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

# Extract the questions of a data file that are structured
# in an undesirable way, use a given regular expression 
# to get the correct structure to the JSON dataset
def process_questions(input_file, output_file, regex, topic):
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
                "topic": question_maker(topic),
                "question": question_text
            })

    # Append to the existing JSON file
    with open(output_file, 'r+') as file:
        data = json.load(file)
        data.extend(questions)
        file.seek(0)
        json.dump(data, file, indent=4)

# process_questions('Datasets/Interviews Datasets/1000 top JS interview questions/Questions.md', 'interviews_dataset.json', r'\[(.*?)\]', "JavaScript")
# process_questions('Datasets/Interviews Datasets/222 Java interview questions/Questions.md', 'interviews_dataset.json', r'- \d+ \. (.*)', "Java")
# process_questions('Datasets/Interviews Datasets/Developer interview questions/Questions.md', 'interviews_dataset.json', r'^(?!#|\[).+$', "Back-end")
# process_questions('Datasets/Interviews Datasets/Essential JS interview questions/Questions.md', 'interviews_dataset.json', r'^## Question \d+\.\s*(.*)$', "JavaScript")
process_questions('Datasets/Interviews Datasets/Front-end interview questions/questions/general-questions.md', 'interviews_dataset.json', r'^\* (.*)$', "Front-end")
