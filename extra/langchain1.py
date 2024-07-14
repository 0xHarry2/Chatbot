# import requests

# # Your Hugging Face API key
# HUGGINGFACE_API_KEY = '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'

# # Function to get a response from the Hugging Face API
# def get_response_from_huggingface(prompt):
#     API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_length": 150,
#             "temperature": 0.7,
#             "do_sample": True,
#         }
#     }
#     response = requests.post(API_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     data = response.json()
#     return data[0]['generated_text']

# # Define a basic chat loop
# def chat():
#     print("Chatbot: Hi! How can I help you today? (Type 'exit' to end the conversation)")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             print("Chatbot: Goodbye!")
#             break
#         prompt = f"User: {user_input}\nChatbot:"
#         response = get_response_from_huggingface(prompt)
#         print(f"Chatbot: {response}")

# if __name__ == "__main__":
#     chat()

# import requests
# import gradio as gr

# # Replace 'your_huggingface_api_key' with your actual Hugging Face API key
# HUGGINGFACE_API_KEY = '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'
# # Function to get a response from the Hugging Face API
# def get_response_from_huggingface(prompt):
#     API_URL = "https://api-inference.huggingface.co/models/emre/llama-2-13b-code-chat"
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_length": 150,
#             "temperature": 0.5,
#             "do_sample": True,
#         }
#     }
#     response = requests.post(API_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     data = response.json()
#     return data[0]['generated_text']

# # Define a function for the Gradio interface
# def chatbot_interface(user_input):
#     prompt = f"User: {user_input}\nChatbot:"
#     response = get_response_from_huggingface(prompt)
#     return response

# # Create the Gradio interface
# # gr.ChatInterface(chatbot_interface).launch()
# iface = gr.Interface(
#     fn=chatbot_interface,
#     inputs=gr.Textbox(lines=7, placeholder="Enter your message here..."),
#     outputs="text",
#     title="Chatbot",
#     description="Type a message and press enter to chat with the bot.",
# )

# # Launch the Gradio interface
# if __name__ == "__main__":
#     iface.launch(share=True)
# from langchain.llms import HuggingFaceHub
# from langchain import PromptTemplate, LLMChain

# # Initialize the LLaMA model from HuggingFace
# llm = HuggingFaceHub(
#     model_name="llama-v2-13b-chat",
#     api_token="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"
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