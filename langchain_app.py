# from langchain_openai import Chat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_fireworks.chat_models import ChatFireworks
# import gradio as gr
from flask import Flask, request, jsonify
from flask_cors import CORS
import waitress
# Initialize LangChain API with your API key
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)
api_key1 = "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"

chat = ChatFireworks(
    model="accounts/fireworks/models/llama-v2-13b-chat",
    temperature=0.5,
    max_tokens=1000,
    api_key="jGQ0lUjQHRq1jfAo2zZRE5fTUsrUx2jfyTTppjknRJ6BwDVy"
)
# Function to interact with the chatbot
@app.route('/chat', methods=['POST'])
def chat_with_llama():
    data = request.get_json()
    input_text  = data.get('question')
    print("input_text.............:",input_text)
    response = chat.invoke(input_text)
    print("output:..................:",response.content)
    send_message = {"response": response.content}
    return jsonify(send_message)

# Main function to run the chatbot
# def main():
#     print("LLaMA-v2-13b-chat Console Chatbot. Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Goodbye!")
#             break
#         response = chat_with_llama(user_input)
#         print(f"LLaMA: {response}")

# if __name__ == "__main__":
#     main()

# iface = gr.Interface(
#     fn=chat_with_llama,
#     inputs=gr.Textbox(lines=7, placeholder="Enter your message here..."),
#     outputs="text",
#     title="Chatbot",
#     description="Type a message and press enter to chat with the bot.",
# )

# Launch the Gradio interface
# if __name__ == "__main__":
#     iface.launch(share=True)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)