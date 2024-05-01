from top2vec import Top2Vec
import umap.plot
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import plotly.graph_objects as go
import umap
import itertools
import numpy as np
import pandas as pd
from umap import UMAP
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from transformers import pipeline
import os
import torch
import transformers
from glob import glob
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser

# Initialize Top2Vec model
model = Top2Vec.load("/kaggle/input/to2vec/top2vec.pkl")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define function to show image
def show_image(url):
    response = requests.get(image_url)
    if response.status_code == 200:
        try:
            image = Image.open(BytesIO(response.content))
            plt.imshow(image)
            plt.axis('off')  
            plt.show()
        except:
            print("exception")
    else:
        print("Failed to fetch the image. Status code:", response.status_code)

# Process QAnon posts JSON file
path = "/kaggle/input/q4anon/QAnon-posts.json"
text_list = []
with open(path, 'r') as file:
    data = json.load(file)
    for key,vals in data.items():
        text_list.append(vals)

text = []
for t in range(len(text_list[0])):
    try:
        text.append(text_list[0][t]['text'])
    except:
        None

# Clean text using NLTK
punctuation = string.punctuation
stop_words = set(stopwords.words('english'))

def clean(text):
    text_list = []
    link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|Q'
    number_pattern = r'>>\d+'
    for t in text:
        links = re.findall(link_pattern, t)
        for link in links:
            try:
                response = requests.get(link)
                soup = BeautifulSoup(response.content, 'html.parser')
                text_content = soup.get_text()
                first_100_words = ' '.join(text_content.split()[:100])
                t = t.replace(link, first_100_words)
            except Exception as e:
                t = t.replace(link, '')
        t = re.sub(number_pattern, '', t)
        t = t.lower()
        t = ''.join(char for char in t if char not in punctuation)
        word_tokens = word_tokenize(t)
        filtered_sentence = ' '.join(word for word in word_tokens if word not in stop_words)
        text_list.append(filtered_sentence)
    return text_list

text_list = clean(text)

# Initialize UMAP model
umap_args = {
    "n_neighbors": 20,
    "n_components": 2, 
    "metric": "cosine",
    'min_dist':0.01,
    'spread':1
}
umap_model = umap.UMAP(**umap_args).fit(model.document_vectors)

# Initialize Plotly figure for UMAP plot
umap.plot.points(umap_model, labels=model.doc_top_reduced, theme="fire")

# Define function to visualize bar chart
def visualize_barchart(topics_list: list, top_n_topics: int = 5, n_words: int = 5, title: str = "Top Topic Words", width: int = 250, height: int = 250):
    colors = itertools.cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    if top_n_topics:
        topics_list = topics_list[:top_n_topics]
    rows = int(np.ceil(len(topics_list) / 4))
    fig = make_subplots(rows=rows, cols=4, subplot_titles=[f"Topic {i+1}" for i in range(len(topics_list))])
    for i, topic_words in enumerate(topics_list):
        words = [word for word in topic_words[:n_words]]
        scores = [1 for _ in range(n_words)]  # Dummy scores for visualization
        row = i // 4 + 1
        col = i % 4 + 1
        fig.add_trace(
            go.Bar(x=scores, y=words, orientation='h', marker_color=next(colors)),
            row=row, col=col
        )
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=22, color="white")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    return fig

# Get topics from Top2Vec model
topics = model.get_topics(5, reduced=True)

# Visualize bar chart for topics
fig = visualize_barchart(topics, top_n_topics=3, n_words=5, title="Top Words in Topics")

# Initialize BERT-based classifier pipeline
model_id = "FacebookAI/roberta-large-mnli"
classifier = pipeline('zero-shot-classification', model=model_id)

# Define sequence to classify and candidate labels
sequence_to_classify = "@g0ssipsquirrelx Wrong, ISIS follows the example of Mohammed and the Quran exactly"
candidate_labels = ["aggreesive","racist","sexist","fraud","political"]

# Classify sequence with candidate labels
classifier(sequence_to_classify, candidate_labels)

# Initialize BART-based summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define article text
ARTICLE = """
New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison. Her next court appearance is scheduled for May 18.
"""

# Summarize article using BART
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

# Initialize FAISS vector store
db = FAISS.from_documents(
    pages,
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
)

# Create retriever from FAISS vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

# Define RAG pipeline chain
rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

# Generate and invoke query
query = prompt(context="why is pelosi net worth plummeting while poor keep being poor?", question="what are context and key sentimental themes here?")
response = rag_chain.invoke(query)

# Print question and response
print("Question:", response["question"])
print(response["text"].replace('\\n', '\n'))