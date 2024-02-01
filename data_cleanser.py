import random

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