import random
import json

def question_maker(title="computer science"):
    questions = [
        f'Could you give me {title} online course recommendations?',
        f'I need some training on {title}',
        f'I want a course on {title}',
        f'How can I improve my {title} skills?',
        f'Give me some online course recommendations',
        f'I need some online courses',
        f'Help me learn more about {title}',
        f'Are there any online courses you recommend on {title}?',
        f'Show me courses on {title}',
        f'I am looking for courses on {title}',
        f'Recommend me some courses on {title}',
    ]
    return random.choice(questions)

def answer_maker(title, description):
    answer = f"Here is an online course I found on IBM SkillsBuild titled ({title}): {description}"
    return answer

def append_to_json_file(file_path, new_data):
    # Open the JSON file for reading and load its content
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("\n\nWarning!! Error while opening the JSON file\n\n")
        data = []

    # Append the new data
    data.append(new_data)

    # Open the JSON file for writing and update its content
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

while True:
    print("\n\n---------- New Entry ----------")
    title = input("\nEnter the course title: ")
    description = input("\nEnter the course description: ")

    entry = {
        "input": question_maker(title),
        "response": answer_maker(title, description)
    }

    # Append entry to the JSON file
    append_to_json_file('../Datasets/courses_dataset.json', entry)