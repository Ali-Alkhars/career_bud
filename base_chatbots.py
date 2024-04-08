import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import logging

"""
This script contains a class to run each of CareerBud's base models:
- t5-small
- meta-llama/Llama-2-7b-chat-hf
- microsoft/DialoGPT-medium
"""

class BaseT5Chatbot:
    def __init__(self, name='t5-small'):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the irrelevant warnings
        self.tokenizer = T5Tokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def chat(self, question):
        # Prepend with "question:" or "answer the question:"
        input_text = f"question: {question} context: "
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Generate an answer
        output_ids = self.model.generate(input_ids, max_length=500, num_beams=3, early_stopping=True)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return answer

    def name(self):
        return 'T5'


class BaseLlamaChatbot:
    def __init__(self, name="meta-llama/Llama-2-7b-chat-hf"):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the irrelevant warnings
        self.tokenizer = AutoTokenizer.from_pretrained(name, torch_dtype="auto")
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", load_in_4bit=True, device_map="auto")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def chat(self, input_text):
        # Tokenize the new input sentence
        new_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # Generate a response
        chat_ids = self.model.generate(new_input_ids.to(self.device), max_length=500, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the response
        response = self.tokenizer.decode(chat_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return response

    def name(self):
        return 'Llama-2'

class BaseDialoGPTChatbot:
    def __init__(self, name="microsoft/DialoGPT-medium"):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the misleading left_padding warning
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name)

    def chat(self, user_input):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Generate a response
        chat_ids = self.model.generate(new_user_input_ids, max_length=500, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the response
        response = self.tokenizer.decode(chat_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return response

    def name(self):
        return 'DialoGPT'
