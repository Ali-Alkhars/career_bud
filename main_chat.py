from trained_chatbots import T5Chatbot, DialoGPTChatbot, LlamaChatbot
from job_fetcher import fetch_jobs

"""
This script handles running the chatbots in a user friendly way.
"""

choice = input("Choose a model (1 for T5), (2 for DialoGPT), (3 for Llama-2): ")
print()
input_text = ''
bot = None
job_search_key = "[JOBSEARCH]" # The key used to train models to search for jobs

if choice == '1':
    bot = T5Chatbot()

elif choice == '2':
    bot = DialoGPTChatbot()

elif choice == '3':
    bot = LlamaChatbot()
    
else:
    print("Incorrect input! Try again")

while input_text != 'quit' and bot != None:
    job_search = False
    input_text = input("\nUser: ")
    if input_text == 'quit':
        response = 'Goodbye! Talk to you soon'
    else:
        response = bot.chat(input_text)
        # Check if user asked for job search
        if job_search_key in response:
            response = response.replace(job_search_key, "")    # remove the key from the output
            job_search = True

    print(f'{bot.name()} Career Bud: {response} \n\n')
    if job_search:
        fetch_jobs()



