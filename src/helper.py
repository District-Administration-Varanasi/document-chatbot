from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone.data.index import Index
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import pinecone
from .translator import trans
import os
from dotenv import load_dotenv
import json
from pathlib import Path


def load_pdfs(data: str) -> list:
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(extracted_data: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def pinecone_init():
    PINECONE_API_KEY, PINECONE_INDEX_NAME = get_secrets()
    try:
        print("Setting up the connection to Pinecone Database...")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"Error setting up the connection to Pinecone Database: {e}")
        exit()
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Current State of the Vector Database: ->")
    print(index.describe_index_stats())
    return index
    

def store_data(text_chunks: list, embeddings: HuggingFaceEmbeddings, index: Index, batch_size: int = 300) -> bool:
    PINECONE_API_KEY, PINECONE_INDEX_NAME = get_secrets()
    print("Storing the data on the Pinecone Vector DataBase")
    total_documents = len(text_chunks)
    ma = 0
    with tqdm(total=total_documents, desc="Storing documents") as pbar:
        for i in range(0, total_documents, batch_size):
            batch = text_chunks[i:i + batch_size]
            try:
                PineconeVectorStore.from_documents(batch, embeddings, index_name=PINECONE_INDEX_NAME)
                pbar.update(len(batch))
            except Exception as e:
                print(f"Error storing documents: {e}")
                return False
    print("Finished Uploading the data on the Pinecone Vector DataBase.")
    print("State of DataBase After Storing: -> ")
    print(index.describe_index_stats())
    return True


def get_chunks_from_pdf(path: str) -> list:
    try:
        print("Extracting data...")
        extracted_data = load_pdfs(data=path)
    except Exception as e:
        print(f"Error during loading the data: {e}")
    print("Translating The pdfs from hindi to english...")
    try:
        translations, data = trans(data = extracted_data)
        try:
            with open(Path('translation\\translations.json'), 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error during Saving the 'translations.json' file: {e}")
    except Exception as e:
        print(f"Error during translation of the pdfs: {e}")
    print("Translated and saved all the pdfs...")
    print("Creating text chunks...")
    try:
        text_chunks = text_split(extracted_data=data)
    except Exception as e:
        print(f"Error during splitting the text: {e}")
    return text_chunks


def get_embeddings(emb_model: str = None) -> HuggingFaceEmbeddings:
    try:
        print("Getting the embedding model...")
        if emb_model is None:
            embeddings = download_hugging_face_embeddings()
        else:
            embeddings = download_hugging_face_embeddings(emb_model)
    except Exception as e:
        print(f"Error during fetching the embedding model: {e}")
        exit()
    return embeddings


def get_secrets() -> tuple:
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    return PINECONE_API_KEY, PINECONE_INDEX_NAME


def load_vectorstore(prompt_template: str, embeddings: HuggingFaceEmbeddings) -> tuple:
    PINECONE_API_KEY, PINECONE_INDEX_NAME = get_secrets()
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context", "history"])
    return vectorstore, PROMPT


    