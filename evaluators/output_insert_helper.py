import json

"""
A helper script to make inserting model outputs to the
evaluation_results.json file user friendly, so then the
outputs could be easily evaluated.
"""

def insert_output(model_name, list_name):
    file_name = '../Evaluation Datasets/evaluation_results.json'

    # Read the existing data from the JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)

    new_entry = [input("\nModel output: ")]

    # Update the intended list
    data[model_name][list_name].extend(new_entry)

    # Write the updated data back to the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated the {list_name} of {model_name} successfully.")


# Run the function
model_name = "DialoGPT-CareerBud"
list_name = "manual_outputs"
while True:
    insert_output(model_name, list_name)
