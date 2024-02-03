from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
data_path = "interviews_dataset.json"
raw_dataset = load_dataset('json', data_files=data_path)

# Split dataset into training and testing
train_test_split = raw_dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Preprocess dataset
def tokenize_function(examples):
    # Tokenize the inputs and labels
    tokenized_inputs = tokenizer(examples["topic"], truncation=True, padding="max_length", max_length=256)
    # Assume that "question" should be predicted from "topic"
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=256)["input_ids"]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./DialoGPT-interviews",
    num_train_epochs=3,  # Adjust based on validation performance
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    logging_steps=10,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
)

# Define compute_metrics function for F1 score
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {"f1": f1_score(labels, predictions)}

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(f'Trainer initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Fine-tune the model
trainer.train()

print(f'Training done! Timestamp: {datetime.datetime.now()}')


trainer.save_model("DialoGPT-interviews")
