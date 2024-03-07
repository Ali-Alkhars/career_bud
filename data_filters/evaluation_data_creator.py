import json

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
    prompt = input("\nEnter the prompt: ")
    ground_truth = input("\nEnter the ground truth: ")

    entry = {
        "prompt": prompt,
        "ground_truth": ground_truth
    }

    # Append entry to the JSON file
    append_to_json_file('../Datasets/evaluation_dataset.json', entry)