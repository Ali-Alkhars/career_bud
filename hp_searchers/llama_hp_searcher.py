import datetime
import evaluate
import json
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import numpy as np
import evaluate

"""
This scripts attempts to find the optimal training parameters
for the Llama-2 model on the combined CareerBud dataset. 

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
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def model_init(trial):
    """Initialise the model"""
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

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
    logits, labels = eval_pred
    # Decode the predictions
    predictions = np.argmax(logits, axis=-1)
    
    # Convert the integer predictions and labels back to text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Labels are -100 for non-target values, so we need to filter those out
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Since BLEU expects a list of references for each prediction, we need to adjust the format
    decoded_labels = [[label] for label in decoded_labels]

    # Initialize the BLEU metric
    bleu_metric = evaluate.load('bleu')

    # Calculate BLEU scores
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"bleu": result["bleu"]}

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Set basic training parameters
training_args = TrainingArguments(
    output_dir="../Llama-2-hp",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=False,
)

# Initialize the Trainer with the model, training arguments, and other necessary inputs
trainer = SFTTrainer(
    args=training_args,
    model_init=model_init,
    peft_config=peft_parameters,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="questions",
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
