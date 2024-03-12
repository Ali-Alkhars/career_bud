import json
from trained_chatbots import T5Chatbot, LlamaChatbot, DialoGPTChatbot
from base_chatbots import BaseT5Chatbot, BaseLlamaChatbot, BaseDialoGPTChatbot

"""
A helper script to make inserting model outputs to the
evaluation_results.json file user friendly, so then the
outputs could be easily evaluated.
"""

def manual_insert_output(model_name, list_name):
    """Insert the model's outputs manually"""
    file_name = './Evaluation Datasets/evaluation_results.json'

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

def auto_insert_output(model_name, list_name, prompts_file, model):
    """
    Insert the model's outputs automatically.
    """
    file_name = './Evaluation Datasets/evaluation_results.json'
    prompts = extract_prompts(prompts_file)

    print(f"\nStarted extracting outputs from {model_name}")
    # Extract model outputs for all prompts
    outputs = []
    for prompt in prompts:
        output = model.chat(prompt)
        outputs.append(output)

    print(f"Extracted outputs from {model_name} successfully!")
    # Read the existing data from the JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Update the intended list
    data[model_name][list_name].extend(outputs)

    # Write the updated data back to the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated the {list_name} of {model_name} successfully.")


def extract_prompts(file_name):
    # Read the JSON data from the file
    with open(file_name, 'r') as file:
        data = json.load(file)

    prompts = []

    # Loop through each entry in the JSON data
    for entry in data:
        # Extract the prompt and add it to the list
        prompts.append(entry['prompt'])
    return prompts

def run_manual_insertion(model_name, list_name):
    """Run the manual_insert_output in a loop"""
    while True:
        manual_insert_output(model_name, list_name)


auto_insert_output(
    model_name="microsoft/DialoGPT-medium", 
    list_name="auto_outputs", 
    prompts_file="./Evaluation Datasets/auto_evaluation_dataset.json",
    model=BaseDialoGPTChatbot()
)