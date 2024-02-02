from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

model_name = 'microsoft/DialoGPT-medium'

# Load dataset
data_path = "interviews_dataset.json"
raw_dataset = load_dataset('json', data_files=data_path, field='data')

# Split dataset into training and testing
train_test_split = raw_dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Preprocess data
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Combine "topic" and "question" for each entry
    return tokenizer(examples['topic'], examples['question'], truncation=True, padding="max_length", max_length=512)

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
    metric_for_best_model="accuracy",      # Use accuracy to identify the best model
)

# Define compute_metrics function for tracking accuracy
from datasets import load_metric

accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

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
