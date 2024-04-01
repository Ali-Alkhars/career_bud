import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import logging

class T5Chatbot:
    def __init__(self, model_name='ali-alkhars/T5-CareerBud', tokenizer_name='t5-small'):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the irrelevant warnings
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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


class LlamaChatbot:
    def __init__(self, tokenizer_name="meta-llama/Llama-2-7b-chat-hf", model_name="ali-alkhars/Llama-2-CareerBud"):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the irrelevant warnings
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, torch_dtype="auto")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", load_in_4bit=True, device_map="auto")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chat_history_ids = None

    def chat(self, input_text):
        # Tokenize the new input sentence
        new_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # Generate a response
        chat_ids = self.model.generate(new_input_ids.to(self.device), max_length=500, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the response
        response = self.tokenizer.decode(chat_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return response

    def chat_with_history(self, input_text):
        # Tokenize the new input sentence
        new_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

        # Generate a response
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=500, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    def name(self):
        return 'Llama-2'

class DialoGPTChatbot:
    def __init__(self, model_name="ali-alkhars/DialoGPT-CareerBud", tokenizer_name="microsoft/DialoGPT-medium"):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the misleading left_padding warning
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None
        self.max_history_saved = 5  # keep track of only 5 messages then reset
        self.current_history_saved = 0

    def chat(self, user_input):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Generate a response
        chat_ids = self.model.generate(new_user_input_ids, max_length=500, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the response
        response = self.tokenizer.decode(chat_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return response


    def chat_with_history(self, user_input):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids
        self.current_history_saved += 1

        # Generate a response from the model
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=500, pad_token_id=self.tokenizer.eos_token_id)

        # Return the model's response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # reset chat history if limit was exceeded
        if self.current_history_saved >= self.max_history_saved:
            self.chat_history_ids = None
            self.current_history_saved = 0
            print("ALERT: I'll have to reset my chat history after this response!!")

        return response

    def name(self):
        # Uncomment if saving chat history
        # return f'({self.current_history_saved if self.current_history_saved != 0 else "5"}/{self.max_history_saved}) DialoGPT'
        return 'DialoGPT'
