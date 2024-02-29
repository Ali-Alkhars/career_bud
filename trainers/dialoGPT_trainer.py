import datetime
import evaluate
import json
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

"""
This script uses the Hugging Face Trainer API to
train the DialoGPT model on the combined CareerBud dataset.
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


# Load the tokenizer and model
model_name = 'microsoft/DialoGPT-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Encode the dataset
def encode(examples):
    encoded = tokenizer(examples['questions'], truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]
    return encoded

encoded_dataset = dataset.map(encode, batched=True)

# Define the Training Arguments
training_args = TrainingArguments(
    output_dir="../DialoGPT-CareerBud-Checkpoints",    # Directory for model outputs
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

# Define evaluate function for tracking BLEU score
def compute_metrics(eval_pred):
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
trainer.save_model("DialoGPT-CareerBud")