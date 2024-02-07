import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
# from transformers.utils import logging
# import warnings
import logging

class T5Chatbot:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.context = "Ali is an Arab. He comes from Saudi Arabia. Ali is 22 years old, even though he feels 18."

    def chat(self, question):
        # Prepend with "question:" or "answer the question:"
        input_text = f"question: {question} context: {self.context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Generate an answer
        output_ids = self.model.generate(input_ids, max_length=150, num_beams=3, early_stopping=True)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return answer

    def name(self):
        return 'T5'


class LlamaChatbot:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

    def chat(self, input_text):
        # Tokenize the new input sentence
        new_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

        # Generate a response
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    def name(self):
        return 'Llama-2'

class DialoGPTChatbot:
    def __init__(self, model_name="DialoGPT-interviews-unevaluated-new", tokenizer_name="microsoft/DialoGPT-medium"):
        logging.getLogger("transformers").setLevel(logging.ERROR) # Stop the misleading left_padding warning
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None
        self.max_history_saved = 5  # keep track of only 5 messages then reset
        self.current_history_saved = 0

    def chat(self, user_input):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids
        self.current_history_saved += 1

        # Generate a response from the model
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Return the model's response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # reset chat history if limit was exceeded
        if self.current_history_saved >= self.max_history_saved:
            self.chat_history_ids = None
            self.current_history_saved = 0
            print("ALERT: I'll have to reset my chat history after this response!!")

        return response

    def name(self):
        return f'({self.current_history_saved if self.current_history_saved != 0 else "5"}/{self.max_history_saved}) DialoGPT'


choice = input("Choose a model (1 for T5), (2 for DialoGPT), (3 for Llama-2): ")
print()
input_text = ''
bot = None

if choice == '1':
    bot = T5Chatbot()

elif choice == '2':
    bot = DialoGPTChatbot()

elif choice == '3':
    bot = LlamaChatbot()
    
else:
    print("Incorrect input! Try again")

while input_text != 'quit' and bot != None:
    input_text = input("User: ")
    if input_text == 'quit':
        response = 'Goodbye! Talk to you soon'
    else:
        response = bot.chat(input_text)

    print(f'{bot.name()} Career Bud: {response} \n\n')



