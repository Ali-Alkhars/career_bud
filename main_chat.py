import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer, RobertaForQuestionAnswering

class T5Chatbot:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.context = ""

    def add_context(self, text):
        self.context += text + " "

    def chat(self, question):
        # Prepend with "question:" or "answer the question:"
        input_text = f"question: {question} context: {self.context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Update context for next question
        self.add_context(question)

        # Generate an answer
        output_ids = self.model.generate(input_ids, max_length=150, num_beams=3, early_stopping=True)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return answer

    # def generate_response(self, input_text):
    #     self.add_context(input_text)
    #     input_ids = self.tokenizer.encode(self.context, return_tensors='pt').to(self.device)
        
    #     # Generate a response
    #     output_ids = self.model.generate(input_ids, max_length=512, num_return_sequences=1, num_beams=5, temperature=0.9)
    #     response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
    #     return response

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

class RoBERTaChatbot:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaForQuestionAnswering.from_pretrained('roberta-large')
        self.context = ""

    def chat(self, question):
        # Append the question to the context
        input_text = self.context + " " + question
        inputs = self.tokenizer(input_text, return_tensors='pt')

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the answer
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        # Update the context
        self.context += " " + question + " " + answer
        return answer


choice = input("Choose a model (1 for T5), (2 for Llama-2), (3 for RoBERTa): ")
print()
input_text = ''
bot = None

if choice == '1':
    bot = T5Chatbot()

elif choice == '2':
    bot = LlamaChatbot()

elif choice == '3':
    bot = RoBERTaChatbot()
    
else:
    print("Incorrect input! Try again")

while input_text != 'quit' and bot != None:
    input_text = input("User: ")
    response = bot.chat(input_text)
    print("Career Bud: ", response, "\n\n")



