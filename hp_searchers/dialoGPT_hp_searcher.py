import datetime
import evaluate
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import evaluate

"""
This scripts attempts to find the optimal training parameters
for the DialoGPT model on the combined CareerBud dataset. 

The code runs hyperparameter search using the optuna library
integrated into the Trainer API.
https://huggingface.co/docs/transformers/en/hpo_train
"""

# Load the JSON dataset
with open('../Datasets/careerbud_dataset.json', 'r') as file:
    data = json.load(file)

# Convert each item to the specified string format and collect them
formatted_questions = [{'questions': f"{item['input']} {item['response']}"} for item in data]

# Convert the list of strings into a Hugging Face Dataset
dataset = Dataset.from_dict({'questions': [item['questions'] for item in formatted_questions]})

# Split data into 90% training and 10% testing
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer.pad_token = tokenizer.eos_token

def encode(examples):
    """Encode the dataset"""
    encoded = tokenizer(examples['questions'], truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]
    return encoded

encoded_dataset = dataset.map(encode, batched=True)

def model_init(trial):
    """Initialise the model"""
    return GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')

def hp_space(trial):
    """Define the hyperparameter space"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 16, 32, 64]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [4, 16, 32, 64]),
    }

def compute_objective(metrics):
    """Define how to compute the objective from the metrics"""
    return metrics["eval_bleu"]

def compute_metrics(eval_pred):
    """Define evaluate function for tracking BLEU score"""
    predictions, labels = eval_pred
    
    # Decode the predictions
    decoded_predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in predictions.argmax(-1)]
    
    # Since the labels are already in the encoded form, we need to decode them as well
    decoded_labels = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)] for g in labels]
    
    # Initialize the BLEU metric
    bleu_metric = evaluate.load('bleu')
    
    # Compute BLEU score
    results = bleu_metric.compute(predictions=decoded_predictions, references=decoded_labels)
    
    return {"bleu": results["bleu"]}

# Set basic training parameters
training_args = TrainingArguments(
    output_dir="../DialoGPT-hp",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
)

# Initialize the Trainer with the model, training arguments, and other necessary inputs
trainer = Trainer(
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
