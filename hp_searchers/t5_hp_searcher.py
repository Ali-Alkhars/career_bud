import datetime
import evaluate
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import evaluate

"""
This scripts attempts to find the optimal training parameters
for the T5 model on the combined CareerBud dataset. 

The code runs hyperparameter search using the optuna library
integrated into the Trainer API.
https://huggingface.co/docs/transformers/en/hpo_train
"""

# Load the JSON dataset
with open('../Datasets/careerbud_dataset.json', 'r') as file:
    data = json.load(file)

# Convert each item to the specified format for inputs and targets
# Here we train the model to answer without context.
formatted_questions = [{'inputs': f"question: {item['input']} context: ", 'targets': item['response']} for item in data]

# Convert the list of dictionaries into a Hugging Face Dataset
dataset = Dataset.from_dict({
    'inputs': [item['inputs'] for item in formatted_questions],
    'targets': [item['targets'] for item in formatted_questions]
})

# Split data into 90% training and 10% testing
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def encode(examples):
    """Encode the dataset"""
    model_inputs = tokenizer(examples['inputs'], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['targets'], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

encoded_dataset = dataset.map(encode, batched=True)

def model_init(trial):
    """Initialise the model"""
    return AutoModelForSeq2SeqLM.from_pretrained('t5-small')

def hp_space(trial):
    """Define the hyperparameter space"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 8),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 16, 32, 64]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [4, 16, 32, 64]),
    }

def compute_objective(metrics):
    """Define how to compute the objective from the metrics"""
    return metrics["eval_bleu"]

def compute_metrics(eval_pred):
    """Define evaluate function for tracking BLEU score"""
    predictions, labels = eval_pred

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    # From: https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Manually check for empty predictions so BLEU doesn't crash
    filtered_predictions = [pred for pred in decoded_predictions if pred]
    filtered_references = [[ref] for ref in decoded_labels if ref]
    if not filtered_predictions or not filtered_references:
        return {"bleu": 0.0}

    # Load BLEU metric
    bleu_metric = evaluate.load('bleu')
    # Compute BLEU score, ensuring that inputs are correctly formatted
    results = bleu_metric.compute(predictions=decoded_predictions, references=decoded_labels)

    return {"bleu": results["bleu"]}

# Set basic training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="../T5-hp",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
)

# Initialize the Trainer with the model, training arguments, and other necessary inputs
trainer = Seq2SeqTrainer(
    args=training_args,
    model_init=model_init,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(f'HP search initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Perform hyperparameter search
best_trial = trainer.hyperparameter_search(
    direction="maximize", 
    hp_space=hp_space,
    compute_objective=compute_objective,
    backend="optuna"
)

print(f"\n\n{best_trial}\n\n")

print(f'HP search done! Timestamp: {datetime.datetime.now()}')
