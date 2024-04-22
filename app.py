from flask import Flask, render_template, request
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import *
from src import helper
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import deque

load_dotenv()

app = Flask(__name__)

index = helper.pinecone_init()
embeddings = helper.get_embeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro")
model = genai.GenerativeModel('gemini-pro')
vectorstore, PROMPT = helper.load_vectorstore(prompt_template=prompt_template, embeddings=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
chatting = model.start_chat()

MAX_HISTORY_LENGTH = 10
chat_history = deque(maxlen=MAX_HISTORY_LENGTH)

@app.route("/")

def index():
    return render_template('chat.html')
    


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
    except KeyError:
        return "Error: 'msg' field not found in the request form."
    
    chat_history.append(("User", msg))
    info = retriever.get_relevant_documents(msg)
    formatted_prompt = prompt_template.format(question=msg, context=info, history=chat_history)
    response = chatting.send_message(formatted_prompt)
    
    chat_history.append(("Bot", response.text))
    return str(response.text)

if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)