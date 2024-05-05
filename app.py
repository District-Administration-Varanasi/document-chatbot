import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from constants import CHROMA_SETTINGS

checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto" ,
    torch_dtype = torch.float32
)
@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'texxt2text-generation',
        model = base_model,
        tokenizer=tokenizer,
        max_length = 256,
        do_sample = True,
        tempratue = 0.3,
        top_p = 0.95
    )
    local_llm = huggingface_pipeline(pipeline=pipe)
    return local_llm

def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db",embeddings_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriver()
    qa = RetrievalQA.from_Chain_type(
        llm = llm,
        chain_type = "stuff",
        return_source_documents = True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title('Search your pdf 📄')
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Questioning and Answering project that reponds to questions about your pdf file.
            """
        )
    question = st.text_area("Enter your Question")
    if st.button("Search"):
        st.info("Your question: "+question)
        st.info("Your Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

if __name__ == '__main__':
    main()