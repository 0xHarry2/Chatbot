# from langchain_openai import Chat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_fireworks.chat_models import ChatFireworks
import fireworks.client
from fireworks.client.image import ImageInference, Answer

# import gradio as gr
from flask import Flask, request, jsonify
from flask_cors import CORS
import waitress
# Initialize LangChain API with your API key
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)
api_key1 = ""

chat = ChatFireworks(
    model="accounts/fireworks/models/llama-v2-13b-chat",
    temperature=0.5,
    max_tokens=1000,
    api_key=""
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

fireworks.client.api_key = ""
inference_client = ImageInference(model="stable-diffusion-xl-1024-v1-0")
@app.route('/image', methods=['POST'])
def imageGeneration(prompt):
    answer : Answer = inference_client.text_to_image(
        prompt=prompt,
        cfg_scale=10,
        height=1024,
        width=1024,
        sampler=None,
        steps=25,
        seed=3,
        safety_check=False,
        output_image_format="PNG",
        # Add additional parameters here
    )

    if answer.image is None:
        raise RuntimeError(f"No return image, {answer.finish_reason}")
    else:
        answer.image.save("output.png")
    return answer

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)
