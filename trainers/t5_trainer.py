import datetime
import evaluate
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

"""
This script uses the Hugging Face Trainer API to
train the T5 model on a particular dataset.
"""

# Load the JSON dataset
with open('../interviews_dataset.json', 'r') as file:
    data = json.load(file)

# Convert each item to the specified format for inputs and targets
formatted_questions = [{'inputs': f"question: {item['topic']} context: ", 'targets': item['question']} for item in data]

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
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Encode the dataset
def encode(examples):
    model_inputs = tokenizer(examples['inputs'], padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['targets'], padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

encoded_dataset = dataset.map(encode, batched=True)

# Define the Training Arguments
training_args = TrainingArguments(
    output_dir="../T5-interviews",
    evaluation_strategy="epoch",
    learning_rate=1e-4,                 # As recommended by the Hugging Face T5 Docs
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='../logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
    save_strategy="epoch",
)

# Define evaluate function for tracking BLEU score
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]
    
    bleu_metric = evaluate.load('bleu')
    results = bleu_metric.compute(predictions=decoded_predictions, references=decoded_labels)
    
    return {"bleu": results["bleu"]}

# Initialise the Trainer
trainer = Trainer(
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
trainer.save_model("T5-interviews")