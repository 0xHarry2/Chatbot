# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # # Load pre-trained model and tokenizer
# # model_name = "microsoft/DialoGPT-medium"
# # model = AutoModelForCausalLM.from_pretrained(model_name)
# # tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load model directly
# # from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
# # Function to generate a response
# def generate_response(input_text):
#     new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if 'chat_history_ids' in globals() else new_user_input_ids

#     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# # Chat with the bot
# while True:
#     user_input = input("You: ")
#     response = generate_response(user_input)
#     print(f"Bot: {response}")

# import os
# import requests

# def get_response_from_huggingface(prompt, temperature=1.0, max_length=50, top_k=50, top_p=0.9):
#     API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
    
#     headers = {
#         'Authorization': f'Bearer {"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"}',
#         'Content-Type': 'application/json'
#     }
    
#     data = {
#         'inputs': prompt,
#         'parameters': {
#             'temperature': temperature,
#             'max_length': max_length,
#             'top_k': top_k,
#             'top_p': top_p
#         }
#     }
    
#     response = requests.post(API_URL, headers=headers, json=data)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {'error': f'Request failed with status code {response.status_code}'}

# def main():
#     print("Welcome to the GPT-J console application!")
    
#     while True:
#         prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
#         if prompt.lower() == 'exit':
#             break
        
#         response = get_response_from_huggingface(prompt)
        
#         if 'error' in response:
#             print(f"Error: {response['error']}")
#         else:
#             print(f"Response: {response[0]['generated_text']}")

# if __name__ == "__main__":
#     main()

# import os
# # from langchain_openai import ChatOpenAI
# # from langchain_core.prompts import PromptTemplate
# # from langchain.chains.llm import LLMChain
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI

# # Load the API key from environment variables
# api_key = "jGQ0lUjQHRq1jfAo2zZRE5fTUsrUx2jfyTTppjknRJ6BwDVy"
# if not api_key:
#     raise ValueError("CHATFIREWORK_API_KEY environment variable not set")

# # Initialize the OpenAI LLM with the API key and model details
# llm = ChatOpenAI(api_key=api_key, model="llama-v2-13b-chat")

# # Define the prompt template
# prompt_template = PromptTemplate(
#     input_variables=["prompt"],
#     template="{prompt}"
# )

# # Initialize the LLMChain with the LLM and prompt template
# # llm_chain = LLMChain(llm=llm, prompt=prompt_template)
# llm_chain = prompt_template   | llm | StrOutputParser()
# def main():
#     print("Welcome to the LLaMA-v2-13b-chat console application with LangChain!")

#     while True:
#         user_input = input("\nEnter your prompt (or type 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             break
        
#         # Generate response using LangChain
#         response = llm_chain.run(prompt=user_input)
        
#         print(f"Response: {response}")

# if __name__ == "__main__":
#     main()

# from langchain.llms import HuggingFaceHub
# from langchain import PromptTemplate, LLMChain

# # Initialize the LLaMA model from HuggingFace
# llm = HuggingFaceHub(
#     model_name="llama-v2-13b-chat",
#     api_token="YOUR_HUGGINGFACE_API_TOKEN"
# )

# # Define a prompt template
# prompt = PromptTemplate(
#     input_variables=["input"],
#     template="You are a helpful assistant. {input}"
# )

# # Create an LLMChain with the model and prompt
# chain = LLMChain(llm=llm, prompt=prompt)

# # Function to get response from the chatbot
# def get_response(user_input):
#     response = chain.run(input=user_input)
#     return response

# def main():
#     print("Chatbot is running. Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Goodbye!")
#             break
#         response = get_response(user_input)
#         print(f"Bot: {response}")

# if __name__ == "__main__":
#     main()

# from transformers import Conversation, pipeline

# # Initialize the conversational pipeline with LLaMA model
# conversational_pipeline = pipeline("conversation", model="facebook/llama-large")

# # Function to interact with the chatbot
# def chat_with_llama(input_text):
#     conversation = Conversation(input_text)
#     response = conversational_pipeline(conversation)
#     return response

# # Main function to run the chatbot
# def main():
#     print("Welcome to LLaMA Chatbot! Type 'exit' to end the conversation.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Goodbye!")
#             break
#         response = chat_with_llama(user_input)
#         print(f"LLaMA: {response['generated_responses'][0]['text']}")

# if __name__ == "__main__":
#     main()

# from langchain.llms.huggingface_hub import HuggingFaceHub
# from langchain_core.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain

# # Initialize the LLaMA model from HuggingFace
# llm = HuggingFaceHub(
#     model_name="facebook/llama-large",
#     huggingfacehub_api_token="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"
# )
# hf = HuggingFaceHub(repo_id="gpt2", huggingfacehub_api_token="my-api-key")
# # Define a prompt template
# prompt = PromptTemplate(
#     input_variables=["input"],
#     template="You are a helpful assistant. {input}"
# )

# # Create an LLMChain with the model and prompt
# chain = LLMChain(llm=llm, prompt=prompt)

# # Function to get response from the chatbot
# def get_response(user_input):
#     response = chain.run(input=user_input)
#     return response

# # Main function to run the chatbot
# def main():
#     print("LLaMA Console Chatbot. Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Goodbye!")
#             break
#         response = get_response(user_input)
#         print(f"LLaMA: {response}")

# if __name__ == "__main__":
#     main()

from langchain.chains.llm import LLMChain
# from langchain. import LangChainAPI

# Initialize LangChain API with your API key
api_key = "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"
# api = LangChainAPI(api_key)

# Initialize LLMChain with the LLaMA-v2-13b-chat model
model_name = "llama-v2-13b-chat"
llama_model = LLMChain(api=api_key, model_name=model_name)

# Function to interact with the chatbot
def chat_with_llama(input_text):
    response = llama_model(input_text)
    return response

# Main function to run the chatbot
def main():
    print("LLaMA-v2-13b-chat Console Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_with_llama(user_input)
        print(f"LLaMA: {response}")

if __name__ == "__main__":
    main()