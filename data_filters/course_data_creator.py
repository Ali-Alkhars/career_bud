import random
import json

def question_maker(topic="computer science"):
    questions = [
        f'Could you give me {topic} online course recommendations?',
        f'I need some training on {topic}',
        f'I want a course on {topic}',
        f'How can I improve my {topic} skills?',
        f'Give me some online course recommendations',
        f'I need some online courses',
        f'Help me learn more about {topic}',
        f'Are there any online courses you recommend on {topic}?',
        f'Show me courses on {topic}',
        f'I am looking for courses on {topic}',
        f'Recommend me some courses on {topic}',
    ]
    return random.choice(questions)

def answer_maker(title, content):
    answer = f"Here is an online course I found on IBM SkillsBuild titled ({title}): {content}"
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
    content = input("\nEnter the course description: ")

    entry = {
        "input": question_maker(title),
        "response": answer_maker(title, content)
    }

    # Append entry to the JSON file
    append_to_json_file('../Datasets/courses_dataset.json', entry)