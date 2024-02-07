from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np

"""
This script uses the Hugging Face Trainer API to
train the DialoGPT model on a particular dataset.
(old tokenizer)
"""

model_name = 'microsoft/DialoGPT-medium'

# Load dataset
data_path = "../interviews_dataset.json"
raw_dataset = load_dataset('json', data_files=data_path)

# Split dataset into training and testing
train_test_split = raw_dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'].select(range(20)),
    'test': train_test_split['test'].select(range(10))
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

# Define the Training Arguments
training_args = TrainingArguments(
    output_dir="../DialoGPT-interviews",    # Directory for model outputs
    evaluation_strategy="epoch",           # Evaluate after each epoch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,                    # Number of training epochs (just like paper)
    weight_decay=0.01,                     # Regularization
    logging_dir='../logs',                  # Directory for logs
    logging_steps=10,                      # Log every 10 steps
    load_best_model_at_end=True,           # Load the best model at the end of training
    save_strategy="epoch",                 # Save model checkpoint after each epoch
    metric_for_best_model="eval_bleu",      # Use BLEU to identify the best model
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions
    # The predictions are logits, convert them to token IDs first (use argmax) and then to text
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in np.argmax(predictions, axis=-1)]
    
    # Decode labels
    # Since labels are directly provided as input_ids in tokenized_datasets, we can decode them directly
    decoded_labels = [tokenizer.decode(lbl, skip_special_tokens=True, clean_up_tokenization_spaces=True) for lbl in labels]
    
    # Initialize the BLEU metric from the 'evaluate' library
    bleu_metric = evaluate.load('bleu')

    # Compute BLEU score
    # Note: BLEU expects a list of predicted sentences and a list of lists of reference sentences
    results = bleu_metric.compute(predictions=decoded_predictions, references=[[label] for label in decoded_labels])
    
    # Return the BLEU score
    return {"bleu": results["bleu"]}

# Initialise the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("DialoGPT-interviews")