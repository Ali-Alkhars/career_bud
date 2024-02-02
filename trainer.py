from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import datetime
import evaluate

"""
This script uses the Hugging Face Trainer API to
train a language model on a particular dataset.
"""

model_name = 'microsoft/DialoGPT-medium'

# Load dataset
data_path = "interviews_dataset.json"
raw_dataset = load_dataset('json', data_files=data_path)

# Split dataset into training and testing
train_test_split = raw_dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Preprocess data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Tokenize the inputs and labels
    tokenized_inputs = tokenizer(examples["topic"], truncation=True, padding="max_length", max_length=128)
    # Assume that "question" should be predicted from "topic"
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=128)["input_ids"]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./DialoGPT-interviews",    # Directory for model outputs
    evaluation_strategy="epoch",           # Evaluate after each epoch
    learning_rate=5e-5,                    # Suggested learning rate for fine-tuning
    per_device_train_batch_size=4,         # Adjust based on your GPU memory
    per_device_eval_batch_size=4,          # Adjust based on your GPU memory
    num_train_epochs=3,                    # Number of training epochs
    weight_decay=0.01,                     # Regularization
    logging_dir='./logs',                  # Directory for logs
    logging_steps=10,                      # Log every 10 steps
    load_best_model_at_end=True,           # Load the best model at the end of training
    save_strategy="epoch",                 # Save model checkpoint after each epoch
    metric_for_best_model="f1",      # Use f1 to identify the best model
)

# Define evaluate function for tracking F1 score
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Flatten the outputs if they're not already
        flattened_predictions = [pred.strip() for pred in decoded_preds]
        flattened_references = [label.strip() for label in decoded_labels]

        # Compute F1 score
        return f1_metric.compute(predictions=flattened_predictions, references=flattened_references)
    except Exception as e:
        print(f"An error occurred during metric computation: {e}")
        return {"F1": None} 


# Initialise the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print(f'Trainer initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Train the model
trainer.train()

print(f'Training done! Timestamp: {datetime.datetime.now()}')

# Save the model
trainer.save_model("DialoGPT-interviews")
