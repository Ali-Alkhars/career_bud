import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer, RobertaForQuestionAnswering

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


choice = input("Choose a model (4 for RoBERTa), (5 for Phi-2): ")
print()
input_text = ''
bot = None

if choice == '4':
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



