import json
from trained_chatbots import T5Chatbot, LlamaChatbot, DialoGPTChatbot
from base_chatbots import BaseT5Chatbot, BaseLlamaChatbot, BaseDialoGPTChatbot

"""
A helper script to make inserting model job activation results to the
evaluation_results.json file user friendly. Or auto assessment of
job activation.
"""

def auto_calc_accuracy(model_name, model):
    """
    Calculate the job activation accuracy automatically.
    Update "evaluation_results.json" with the results.
    """
    activation_prompts = extract_prompts("activation")
    non_activation_prompts = extract_prompts("non_activation")

    tp = 0  # True positive
    tn = 0  # True negative
    fp = 0  # False positive
    fn = 0  # False negative
    activation_key = "[JOBSEARCH]"

    # Calculate the accuracy of activation prompts
    for prompt in activation_prompts:
        output = model.chat(prompt)
        if activation_key in output:
            tp += 1
        else:
            fn += 1

    # Calculate the accuracy of non_activation prompts
    for prompt in non_activation_prompts:
        output = model.chat(prompt)
        if activation_key not in output:
            tn += 1
        else:
            fp += 1

    # Calculate results and upload them to "evaluation_results.json"
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    results = {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy
    }
    extend_accuracy_results(model_name=model_name, results=results)

def manual_calc_accuracy(model_name):
    """
    Calculate the job activation accuracy, by entering results manually.
    Update "evaluation_results.json" with the results.
    """
    activation_prompts = extract_prompts("activation")
    non_activation_prompts = extract_prompts("non_activation")

    tp = 0  # True positive
    tn = 0  # True negative
    fp = 0  # False positive
    fn = 0  # False negative

    # Calculate the accuracy of activation prompts
    for prompt in activation_prompts:
        print(f"\n\nActivation prompt: \n{prompt}")
        result = input("\nResult ('t' for True positive. anything else for False negative): ").lower()
        if result == "t":
            tp += 1
        else:
            fn += 1

    # Calculate the accuracy of non_activation prompts
    for prompt in non_activation_prompts:
        print(f"\n\nNon-activation prompt: \n{prompt}")
        result = input("\nResult ('t' for True negative. anything else for False positive): ").lower()
        if result == "t":
            tn += 1
        else:
            fp += 1

    # Calculate results and upload them to "evaluation_results.json"
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    results = {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy
    }
    extend_accuracy_results(model_name=model_name, results=results)

def extend_accuracy_results(model_name, results):
    """
    Extend the "evaluation_results.json" file with the given
    accuracy results
    """
    file_name = "./Evaluation Datasets/evaluation_results.json"
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Update the intended list
    data[model_name]['job_activation_accuracy'].update(results)

    # Write the updated data back to the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated accuracy of {model_name} successfully.")

def extract_prompts(list_name):
    with open('./Evaluation Datasets/job_activation_prompts.json', 'r') as file:
        data = json.load(file)

    return data[list_name]

# auto_calc_accuracy(model_name="meta-llama/Llama-2-7b-chat-hf", model=BaseLlamaChatbot())
manual_calc_accuracy("DialoGPT-CareerBud")