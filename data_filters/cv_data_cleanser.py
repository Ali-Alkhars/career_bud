import json
import random
from datasets import load_dataset

"""
This script is intended to structure specific data files into 
a Json dataset that will be used to train Language Models on 
CV improvement tips.
"""

def question_maker():
    """Return a random question from a pre-made list of questions"""
    questions = [
        f'Can you give me some tips to improve my CV?',
        f'Can you give me some tips to improve my Resume?',
        f'How can I improve my CV?',
        f'How can I improve my Resume?',
        f'I want ways to enhance my CV',
        f'I want ways to enhance my Resume',
        f'Could you suggest changes on me CV?',
        f'Could you suggest changes on me Resume?',
        f'Give me some CV tips',
        f'Give me some Resume tips',
        f'I need help with my CV',
        f'I need help with my Resume',
        f'I want you to help me update my CV',
        f'I want you to help me update my Resume',
        f'Could you improve my CV?',
        f'Could you improve my Resume?',
    ]
    return random.choice(questions)

def process_HF_questions():
    # extract the data
    data = load_dataset('gkrishnan/Resume_Best_Practices')
    text_data = data['train']['text'][0].replace('\n', ' ')  # Replacing \n with space

    # Split the text into sentences (by full stop).
    sentences = text_data.split('. ')
    
    # Group every two sentences together.
    grouped_sentences = ['. '.join(sentences[i:i+2]) + '.' for i in range(0, len(sentences)-1, 2)]
    
    # Create the structured dataset.
    structured_data = [{"input": question_maker(), "response": entry} for entry in grouped_sentences]
    
    # Write the structured dataset to a JSON file.
    with open('../Datasets/cv_dataset.json', 'w', encoding='utf-8') as file:
        json.dump(structured_data, file, ensure_ascii=False, indent=4)

def process_web_questions():
    # extract the data
    with open('../Unfiltered-datasets/CV/CV_improvements_data.txt', 'r', encoding='utf-8') as file:
        data = file.read()

    # Split the text into sentences (by full stop).
    sentences = data.split('. ')
    
    # Group every two sentences together.
    grouped_sentences = ['. '.join(sentences[i:i+2]) + '.' for i in range(0, len(sentences)-1, 2)]
    
    # Create the structured dataset.
    structured_data = [{"input": question_maker(), "response": entry} for entry in grouped_sentences]
    
    # Write the structured dataset to a JSON file.
    with open('../Datasets/cv_dataset.json', 'r+', encoding='utf-8') as file:
        allData = json.load(file)
        allData.extend(structured_data)
        file.seek(0)
        json.dump(allData, file, indent=4)

# Create the structured dataset
process_web_questions()
# process_HF_questions()