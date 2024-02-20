import datetime
import evaluate
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

"""
This script uses the Hugging Face Trainer API to
train the T5 model on courses_dataset.
"""

# Load the JSON dataset
with open('../Datasets/courses_dataset.json', 'r') as file:
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

# Load the tokenizer and model
model_name = '../T5-interviews'
tokenizer_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Encode the dataset
def encode(examples):
    model_inputs = tokenizer(examples['inputs'], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['targets'], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

encoded_dataset = dataset.map(encode, batched=True)

# Define the Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="../T5-IC",
    evaluation_strategy="epoch",
    learning_rate=3e-4,                    # As recommended by the Hugging Face T5 Docs
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate = True,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='../logs',
    logging_steps=10,                      # Log every 10 steps
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
    save_strategy="epoch",
)

# Define evaluate function for tracking BLEU score
def compute_metrics(eval_pred):
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

# Initialise the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

print(f'Trainer initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Train the model
trainer.train()

print(f'Training done! Timestamp: {datetime.datetime.now()}')

# Save the model
trainer.save_model("T5-IC")