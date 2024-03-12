import evaluate
import json

"""
This script is used to evaluate the outputs of
models automatically on three metrics:
- BLEU
- ROUGE
- METEOR
"""

def eval_outputs(model_name):
    """
    Evaluate the outputs (predictions) of the given model 
    (both auto & manually generated evaluation data).
    Upload the results in the "evaluation_results.json" file.
    """
    file_name = '../Evaluation Datasets/evaluation_results.json'
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Get the model's outputs and references for the auto generated prompts and ground truths
    auto_predictions = data[model_name]["auto_outputs"]
    auto_references = extract_references('../Evaluation Datasets/auto_evaluation_dataset.json')

    # Get the model's outputs and references for the manually generated prompts and ground truths
    manual_predictions = data[model_name]["manual_outputs"]
    manual_references = extract_references('../Evaluation Datasets/manual_evaluation_dataset.json')
    
    metrics = ["bleu", "rouge", "meteor"]
    for m in metrics:
        metric = evaluate.load(m)
        auto_results = metric.compute(predictions=auto_predictions, references=auto_references)
        manual_results = metric.compute(predictions=manual_predictions, references=manual_references)

        extend_metric_results(file_name=file_name, model_name=model_name, metric_name=f"auto_{m}", results=auto_results)
        extend_metric_results(file_name=file_name, model_name=model_name, metric_name=f"manual_{m}", results=manual_results)

def extend_metric_results(file_name, model_name, metric_name, results):
    """
    Extend the "evaluation_results.json" file with the given
    metric result
    """
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Update the intended list
    data[model_name][metric_name].update(results)

    # Write the updated data back to the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated {metric_name} of {model_name} successfully.")


def extract_references(file_name):
    """
    Extract the references (ground_truths) from the given
    evaluation dataset path.
    """
    # Read the JSON data from the file
    with open(file_name, 'r') as file:
        data = json.load(file)

    references = []

    # Loop through each entry in the JSON data
    for entry in data:
        # Extract the prompt and add it to the list
        references.append(entry['ground_truth'])
    return references

eval_outputs('Gemini')