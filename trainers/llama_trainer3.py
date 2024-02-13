import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from datasets import DatasetDict, Dataset
import numpy as np
import evaluate
import datetime

"""
This script uses the Hugging Face Trainer API to
train the Llama-2-chat model on a particular dataset.

Doesn't work (You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of)
"""

# Load dataset
with open("../interviews_dataset.json", "r") as file:
    data = json.load(file)

# Convert each item to the specified format for inputs and targets
formatted_questions = [{'inputs': f"Topic: {item['topic']} Question: {item['question']}", 'targets': item['question']} for item in data]

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

# Load the tokenizer and model
model_name = 'meta-llama/Llama-2-7b-chat-hf'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", load_in_4bit=True, device_map="auto")

# Prepare the dataset
def encode(examples):
    model_inputs = tokenizer(examples['inputs'], max_length=128, truncation=True, padding="max_length")
    # Llama-2 expects the data in a specific format, adjust the preprocessing as needed
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['targets'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets
encoded_dataset = dataset.map(encode, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="../Llama-2-interviews",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,                      # Log every 10 steps
    logging_dir='../logs',                  # Directory for logs
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Initialize the BLEU metric
    bleu_metric = evaluate.load('bleu')
    # Compute BLEU score
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print(f'Trainer initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Train the model
trainer.train()

print(f'Training done! Timestamp: {datetime.datetime.now()}')

# Save the model
trainer.save_model("Llama-2-interviews")
