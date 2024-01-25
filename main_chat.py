import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer, RobertaForQuestionAnswering

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

class RoBERTaChatbot:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaForQuestionAnswering.from_pretrained('roberta-large')
        self.fixed_context = "Ali is an Arab. He comes from Saudi Arabia. Ali is 22 years old, even though he feels 18."

    def chat(self, question):
        # Combine the fixed context with the question
        input_text = self.fixed_context + " " + question
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the answer
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        return answer

    def name(self):
        return 'RoBERTa'

class DialoGPTChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
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
        return 'DialoGPT'

# Encounters error: RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
class PhiChatbot:
    def __init__(self, model_name="microsoft/phi-2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

    def chat(self, user_input):
        # inputs = self.tokenizer(user_input, return_tensors="pt", return_attention_mask=False)
        # outputs = self.model.generate(**inputs, max_length=200)
        # response = self.tokenizer.batch_decode(outputs)[0]

        with torch.no_grad():
            token_ids = self.tokenizer.encode(user_input, add_special_tokens=False ,return_tensors="pt")
            output_ids = self.model.generate(
                token_ids.to(self.device),
                max_new_tokens=512,
                do_sample=True,
                temperature = 0.3
            )

        response = self.tokenizer.decode(output_ids[0][token_ids.size(1) :])
        return response

    def name(self):
        return 'Phi-2'


choice = input("Choose a model (1 for T5), (2 for DialoGPT), (3 for Llama-2), (4 for RoBERTa), (5 for Phi-2): ")
print()
input_text = ''
bot = None

if choice == '1':
    bot = T5Chatbot()

elif choice == '2':
    bot = DialoGPTChatbot()

elif choice == '3':
    bot = LlamaChatbot()

elif choice == '4':
    bot = RoBERTaChatbot()

elif choice == '5':
    bot = PhiChatbot()
    
else:
    print("Incorrect input! Try again")

while input_text != 'quit' and bot != None:
    input_text = input("User: ")
    if input_text == 'quit':
        response = 'Goodbye! Talk to you soon'
    else:
        response = bot.chat(input_text)

    print(f'{bot.name()} Career Bud: {response} \n\n')



