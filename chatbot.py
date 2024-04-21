from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader

load_dotenv()
app = Flask(__name__)
load_dotenv()

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize global variables
vectorstore = None
conversation_chain = None
chat_history = []

def main(pdf_folder):
    """Main function to initialize vectorstore and conversation_chain."""
    global vectorstore, conversation_chain
    try:
        raw_text = get_pdf_text(pdf_folder)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)
    except Exception as e:
        print(f"Error in main function: {str(e)}")

def get_pdf_text(pdf_folder):
    """Extract text from PDF files in the specified folder."""
    try:
        text = ""
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_reader = PdfReader(open(pdf_path, "rb"))
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error in get_pdf_text function: {str(e)}")

def get_text_chunks(text):
    """Split text into chunks for processing."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error in get_text_chunks function: {str(e)}")

def get_vectorstore(text_chunks):
    """Generate vectorstore from text chunks."""
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error in get_vectorstore function: {str(e)}")

def get_conversation_chain(vectorstore):
    """Create a conversation chain for handling user queries."""
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        print(f"Error in get_conversation_chain function: {str(e)}")

@app.route("/", methods=["POST"])
def answer_question():
    """Handle POST requests containing user questions."""
    try:
        data = request.get_json()
        if "question" in data:
            user_question = data["question"]
            response = handle_user_input(user_question)
            print("answer", response)
            return jsonify({"response": response})
        else:
            return jsonify({"error": "Missing 'question' parameter in the request"}), 400
    except Exception as e:
        print(f"Error in answer_question function: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

def handle_user_input(user_question):
    """Process user input and generate a response."""
    try:
        global chat_history
        default_prompt = "You are a support bot, For general greetings like ""hi"" or ""hello,"" provide general response like ""Hello!  I'm here to assist you""."
        user_question_with_prompt = default_prompt + user_question
        response = conversation_chain({'question': user_question_with_prompt, 'chat_history': chat_history})
        chat_history.append({"role": "system", "content": "You asked: " + user_question})

        if 'answer' in response:
            response_text = response['answer']
            chat_history.append({"role": "assistant", "answer": response_text})
            return response_text
        else:
            return "Error: Unexpected response format"
    except Exception as e:
        print(f"Error in handle_user_input function: {str(e)}")
        return "An error occurred while processing the user input"

if __name__ == '__main__':
    pdf_folder = "pdf" #PDF Folder name will contain the pdf files 
    main(pdf_folder)
    app.run(port=8000)
