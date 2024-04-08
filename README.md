# CareerBud

**CareerBud** is a chatbot fine-tuned to assist you boost your professional career!

## CareerBud's abilities
- Recommending IBM SkillsBuild courses
- Suggesting CV improvement tips
- Providing job interview preparation questions
- Providing real-time job offers (requires [Adzuna API keys](#how-to-get-adzuna-api-keys))

## CareerBud Base models
- [T5 small](https://huggingface.co/t5-small)
- [DialoGPT Medium](https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Julien%21+How+are+you%3F)
- [Llama 2 Chat 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

## CareerBud models and datasets
The three CareerBud models and the datasets used for fine-tuning and evaluation could be accessed [HERE](https://huggingface.co/ali-alkhars)

## CareerBud User Guide
Follow these thourough steps to start chatting with CareerBud!

1. **Create a virtual environment** by typing the following command in a terminal situated in the main directory: `python3 -m venv myenv` and activate it using `source myenv/bin/activate`.

2. **Install the required libraries**: `pip3 install -r requirements.txt`

3. \[optional but highly recommended\] to be able to chat with Llama 2 and receive job offers:
   - To chat with CareerBud Llama 2 you'll need to run CareerBud on a device that has a GPU
   - For CareerBud to be able to fetch you current job offers, you need to request Adzuna API keys. You can get the keys by following [THESE](#how-to-get-adzuna-api-keys) extra steps.

4. **Start CareerBud**: `python3 main_chat.py`, choose a model and start chatting! You could end the chat by typing `quit`.

## How to Get Adzuna API Keys
1. Fill [THIS](https://developer.adzuna.com/signup) contact form
2. Verify your account through the received email
3. View your API Keys by navigating to Dashboard > API Access Details
4. Go back to the CareerBud code. Create a new file in the main directory named `.env` fill it with the following fields:
   ```
   APP_ID=[Add here your application ID]
   APP_KEY=[Add here your application key]
   ```
