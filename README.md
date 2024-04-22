# Gov_Doc

## Description

This project is a Flask web application that utilizes a combination of language models and document embeddings for text-based conversation and document retrieval. It integrates Google's GenerativeAI models and Pinecone Vector Database to provide conversational AI capabilities and efficient document search.

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Set up environment variables:

   Create a `.env` file in the root directory and add the following:

   ```plaintext
   PINECONE_API_KEY=<your_pinecone_api_key>
   PINECONE_INDEX_NAME=<your_pinecone_index_name>
   ```

2. Run the Flask application:

```bash
python app.py
```


Sure, let's add that information to the `Usage` section:

---

## Usage

1. Set up environment variables:

   Create a `.env` file in the root directory and add the following:

   ```plaintext
   PINECONE_API_KEY=<your_pinecone_api_key>
   PINECONE_INDEX_NAME=<your_pinecone_index_name>
   ```

2. Run the Flask application:

```bash
python app.py
```
![Example-1](https://raw.githubusercontent.com/Azazel0203/gov_doc/main/static/ex1.jpg)
![Example -2](https://raw.githubusercontent.com/Azazel0203/gov_doc/main/static/ex2.jpg)
![Example -3](https://raw.githubusercontent.com/Azazel0203/gov_doc/main/static/ex3.jpg)

3. Run the script to store document indexes:

```bash
python store_indexes.py
```
![Image Alt Text](https://raw.githubusercontent.com/Azazel0203/gov_doc/main/static/out1.jpg)
This step is necessary to ensure that the document indexes are properly stored in the Pinecone Vector Database for efficient retrieval.


4. Access the application in your web browser at `http://localhost:8000`.
--- 

## Features

- **Chat Interface**: Engage in conversation with the integrated generative AI model.
- **Document Search**: Retrieve relevant documents based on user queries.
- **PDF Processing**: Extract text from PDF documents, translate from Hindi to English, and split into manageable chunks.
- **Pinecone Integration**: Store document embeddings for efficient retrieval and search.

## Functionality

### `chat()`

This function handles the chat interaction between the user and the AI model. It processes user input, retrieves relevant documents, and generates a response using the generative AI model.

```python
@app.route("/get", methods=["GET", "POST"])
def chat():
    # Process user input
    msg = request.form["msg"]
    chat_history.append(("User", msg))
    
    # Retrieve relevant documents
    info = retriever.get_relevant_documents(msg)
    
    # Generate response
    formatted_prompt = prompt_template.format(question=msg, context=info, history=chat_history)
    response = chatting.send_message(formatted_prompt)
    
    chat_history.append(("Bot", response.text))
    return str(response.text)
```

### `store_vectors()`

This function extracts text from PDF documents, translates from Hindi to English, splits into chunks, retrieves embeddings, and stores them in the Pinecone Vector Database.

```python
def store_vectors():
    chunks = helper.get_chunks_from_pdf(path="test_documents")
    embeddings = helper.get_embeddings()
    index = helper.pinecone_init()
    done = helper.store_data(text_chunks=chunks, embeddings=embeddings, index=index)
    if done == False:
        exit()
```

## Data Processing Pipeline

1. **PDF Extraction**: PDF documents are loaded and their text content is extracted.

```python
# Load PDFs and extract text
chunks = helper.get_chunks_from_pdf(path="test_documents")
```

2. **Translation**: Hindi text, if present, is translated to English.

```python
# Translate Hindi text to English
translations, data = helper.trans(data=extracted_data)
```

3. **Text Chunking**: Text is divided into manageable chunks for efficient processing.

```python
# Split text into chunks
text_chunks = helper.text_split(extracted_data=data)
```

4. **Embedding Generation**: Text chunks are converted into embeddings using Hugging Face models.

```python
# Retrieve embeddings
embeddings = helper.get_embeddings()
```

5. **Pinecone Storage**: Embeddings are stored in the Pinecone Vector Database for fast retrieval.

```python
# Store vectors in Pinecone
done = helper.store_data(text_chunks=chunks, embeddings=embeddings, index=index)
```

## Example Usage

```python
# Import necessary modules
import src.helper as helper

# Extract text from PDFs, translate, and store vectors
helper.store_vectors()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements or bug fixes.

## License

[MIT License](LICENSE)

---
