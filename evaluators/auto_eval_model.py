import evaluate
import json
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

"""
This script is used to evaluate the outputs of
models automatically on four metrics:
- BLEU
- ROUGE
- METEOR
- SimCSE
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

def eval_simCSE_score(model_name):
    """
    Evaluate the outputs (predictions) of the given model 
    (both auto & manually generated evaluation data) on the 
    simCSE evaluation metric.
    Upload the results in the "evaluation_results.json" file.
    """
    simCSE_model_name = "princeton-nlp/sup-simcse-roberta-large"    # simCSE model used to check embeddings similarity
    tokenizer = AutoTokenizer.from_pretrained(simCSE_model_name)
    model = AutoModel.from_pretrained(simCSE_model_name)

    # Evaluate for both manually and auto created evaluation datasets
    evaluation_dataset_types = ['auto', 'manual']
    for dataset_type in evaluation_dataset_types:
        # Get the model's outputs and references
        results_file_name = '../Evaluation Datasets/evaluation_results.json'
        with open(results_file_name, 'r') as file:
            data = json.load(file)

        outputs = data[model_name][f"{dataset_type}_outputs"]
        ground_truths = extract_references(f'../Evaluation Datasets/{dataset_type}_evaluation_dataset.json')

        # Tokenize model outputs and ground_truths
        predictions = tokenizer(outputs, padding=True, truncation=True, return_tensors="pt")
        references = tokenizer(ground_truths, padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            prediction_embeddings = model(**predictions, output_hidden_states=True, return_dict=True).pooler_output
            reference_embeddings = model(**references, output_hidden_states=True, return_dict=True).pooler_output

        # Get the similarities
        similarities = []
        for i in range(len(prediction_embeddings)):
            entry_similarity = 1 - cosine(reference_embeddings[i], prediction_embeddings[i])
            similarities.append(entry_similarity)

        # Calculate the average similarity score
        score = sum(similarities) / len(similarities)
        print(f"\n{model_name} similarity score for {dataset_type} dataset: {score}")

        # Insert the result
        add_metric_result(model_name=model_name, result={f"{dataset_type}_simCSE": score})

def add_metric_result(model_name, result):
    """
    Extend the "evaluation_results.json" file with the given
    metric's results.
    """
    file_name = "../Evaluation Datasets/evaluation_results.json"
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Add the intended list
    data[model_name].update(result)

    # Write the updated data back to the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Metric added to {model_name} successfully.")

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

# eval_outputs('Gemini')
eval_simCSE_score("Gemini")