import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer

class T5Chatbot:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.context = ""

    def add_context(self, text):
        self.context += text + " "

    def generate_response(self, input_text):
        self.add_context(input_text)
        input_ids = self.tokenizer.encode(self.context, return_tensors='pt').to(self.device)
        
        # Generate a response
        output_ids = self.model.generate(input_ids, max_length=512, num_return_sequences=1, num_beams=5, temperature=0.9)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return response

class LlamaChatbot:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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


choice = input("Choose a model (1 for T5), (2 for Llama-2): ")
print()
input_text = ''

if choice == '1':
    bot = T5Chatbot()
    while input_text != 'quit':
        input_text = input("User: ")
        response = bot.generate_response(input_text)
        print(response, "\n\n")
elif choice == '2':
    bot = LlamaChatbot()
    while input_text != 'quit':
        input_text = input("User: ")
        response = bot.chat(input_text)
        print(response, "\n\n")
else:
    print("Incorrect input! Try again")



