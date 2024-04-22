import os
import pinecone
from tqdm import tqdm
from dotenv import load_dotenv
import src.helper as helper
from langchain_pinecone import PineconeVectorStore

def store_vectors():
    chunks = helper.get_chunks_from_pdf(path="test_documents")
    embeddings = helper.get_embeddings()
    index = helper.pinecone_init()
    done = helper.store_data(text_chunks=chunks, embeddings=embeddings, index=index)
    if done == False:
        exit()


if __name__ == "__main__":
    store_vectors()
    
    
    
    
# # load all pdfs from "original_doc_test"

# '''
# Use These function... that is pre implemented to load the pdf
# def load_pdfs(data: str) -> list:
#     loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
#     return loader.load()
# '''


# # the pdfs are in hindi...perform ocr and convert the entire data to english
# '''
# Suggest proper codes and procedure to do this
# '''




# # use hugging face embeddings to embed the english data
# '''
# use these preimplemented funtions
# def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
#     return HuggingFaceEmbeddings(model_name=model_name)
# def text_split(extracted_data: list) -> list:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     return text_splitter.split_documents(extracted_data)
# def get_chunks_from_pdf(path: str) -> list:
#     try:
#         print("Extracting data...")
#         extracted_data = load_pdfs(data="data")
#     except Exception as e:
#         print(f"Error during loading the data: {e}")
#         exit()
#     print("Creating text chunks...")
#     text_chunks = text_split(extracted_data=extracted_data)
#     return text_chunks
# def get_embeddings(emb_model: str = None) -> HuggingFaceEmbeddings:
#     try:
#         print("Getting the embedding model...")
#         if emb_model is None:
#             embeddings = download_hugging_face_embeddings()
#         else:
#             embeddings = download_hugging_face_embeddings(emb_model)
#     except Exception as e:
#         print(f"Error during fetching the embedding model: {e}")
#     return embeddings

# '''

# # store the embedded data on the pinecone vector database
# '''
# use these preimplemented funtions
# def store_data(text_chunks: list, embeddings: HuggingFaceEmbeddings, index: Index, batch_size: int = 300) -> bool:
#     PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME = get_secrets()
#     print("Storing the data on the Pinecone Vector DataBase")
#     total_documents = len(text_chunks)
#     with tqdm(total=total_documents, desc="Storing documents") as pbar:
#         for i in range(0, total_documents, batch_size):
#             batch = text_chunks[i:i + batch_size]
#             try:
#                 PineconeVectorStore.from_documents(batch, embeddings, index_name=PINECONE_INDEX_NAME)
#                 pbar.update(len(batch))
#             except Exception as e:
#                 print(f"Error storing documents: {e}")
#                 return False
#     print("Finished Uploading the data on the Pinecone Vector DataBase.")
#     print("State of DataBase After Storing: -> ")
#     print(index.describe_index_stats())
#     return True

# def pinecone_init():
#     PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME = get_secrets()
#     try:
#         print("Setting up the connection to Pinecone Database...")
#         pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
#     except Exception as e:
#         print(f"Error setting up the connection to Pinecone Database: {e}")
#         exit()
#     index = pc.Index(PINECONE_INDEX_NAME)
#     print("Current State of the Vector Database: ->")
#     print(index.describe_index_stats())
#     return index

# def get_secrets() -> tuple:
#     load_dotenv()
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     PINECONE_ENV = os.getenv("PINECONE_ENV")
#     PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
#     return PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME


# '''




