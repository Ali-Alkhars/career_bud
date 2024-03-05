import torch
import json
import datetime
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import numpy as np
import evaluate

"""
This script uses the Hugging Face Trainer API to
train the Llama-2-chat model on the combined CareerBud dataset.

Taken from: https://deci.ai/blog/fine-tune-llama-2-with-lora-for-question-answering/
Useful: https://huggingface.co/blog/4bit-transformers-bitsandbytes
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


# Model and tokenizer names
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Define evaluate function for tracking BLEU score (using evaluate library)
def compute_metrics(eval_pred):
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
# As suggested by https://deci.ai/blog/fine-tune-llama-2-with-lora-for-question-answering/
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Define the Training Arguments
train_params = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,

    output_dir="../Llama-2-CareerBud-checkpoints",
    evaluation_strategy="epoch",           # Evaluate after each epoch
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=False,
    group_by_length=True,
    load_best_model_at_end=True,           # Load the best model at the end of training
    save_strategy="epoch",                 # Save model checkpoint after each epoch
    metric_for_best_model="eval_bleu",      # Use BLEU to identify the best model
)

# Initialise the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_parameters,
    dataset_text_field="questions",
    tokenizer=tokenizer,
    args=train_params,
    compute_metrics=compute_metrics
)

print(f'Trainer initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Train the model
trainer.train()

print(f'Training done! Timestamp: {datetime.datetime.now()}')

# Save the model
trainer.save_model("Llama-2-CareerBud")