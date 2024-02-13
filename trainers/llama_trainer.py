import datetime
import json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import torch
from trl import SFTTrainer

"""
This script uses the Hugging Face Trainer API to
train the Llama-2-chat model on a particular dataset.
Taken from: https://www.youtube.com/watch?v=MDA3LUKNl1E
And: https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19

Doesn't work (Some deep data setup issues)
"""

# Load the JSON dataset
with open('../interviews_dataset.json', 'r') as file:
    data = json.load(file)

# Convert each item to the specified string format and collect them
formatted_questions = [{'questions': f"### Input:{item['topic']}  ### Response:{item['question']}"} for item in data]

# Convert the list of strings into a Hugging Face Dataset
dataset = Dataset.from_dict({'questions': [item['questions'] for item in formatted_questions]})

# Split data into 90% training and 10% testing
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Load the tokenizer and model
model_name = 'meta-llama/Llama-2-7b-chat-hf'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_auth_token=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1 

# Setup LoRA
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CASUAL_LM"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# model.add_adapter(peft_config)

# Define the Training Arguments
# training_args = TrainingArguments(
#     output_dir="../Llama-2-interviews",    # Directory for model outputs
#     evaluation_strategy="epoch",           # Evaluate after each epoch
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     gradient_accumulation_steps=4,
#     optim="paged_adamw_32bit",
#     learning_rate=1e-4,
#     fp16=True,
#     max_grad_norm=0.3,
#     warmup_ratio=0.05,
#     group_by_length=True,
#     save_safetensors=True,
#     lr_scheduler_type="cosine",
#     seed=43,
#     num_train_epochs=3,                    # Number of training epochs
#     weight_decay=0.01,                     # Regularization
#     logging_dir='../logs',                  # Directory for logs
#     logging_steps=10,                      # Log every 10 steps
#     load_best_model_at_end=True,           # Load the best model at the end of training
#     save_strategy="epoch",                 # Save model checkpoint after each epoch
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
# )
training_args = TrainingArguments(
    output_dir="../Llama-2-interviews",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500
)

# Initialise the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="questions",
    max_seq_length=512
)

print(f'Trainer initialised and now starting. Timestamp: {datetime.datetime.now()}')

# Train the model
trainer.train()

print(f'Training done! Timestamp: {datetime.datetime.now()}')

# Save the model
trainer.save_model("Llama-2-interviews")