
# RAG Chatbot with PDF Upload

This project implements a Retrieval-Augmented Generation (RAG) chatbot using a PDF document as the knowledge base. It allows users to upload a PDF file, processes its content, and then uses the LLaMA model to generate context-aware answers to user queries.

## Features
- **PDF Upload**: Users can upload PDF files, and the content is processed and split into smaller chunks for better retrieval performance.
- **Vector Store**: Uses FAISS to store and retrieve vector embeddings of document chunks.
- **Embeddings**: Leverages HuggingFace's Sentence Transformers for creating embeddings.
- **LLaMA Integration**: Utilizes a pre-trained LLaMA model (`llama-2-7b-chat.ggmlv3.q4_0.bin`) for generating responses.
- **Streamlit UI**: A user-friendly interface built with Streamlit.

---

## Installation and Setup

### Prerequisites
1. Python 3.8 or higher installed.
2. Required Python libraries:
   - `streamlit`
   - `langchain-community`
   - `sentence-transformers`
   - `CTransformers`
   - `faiss`

### Clone the Repository
```bash
git clone https://github.com/dineshramv13/Llama_chatbot
cd RAG-PDF-Chatbot
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Model File
- Download the `llama-2-7b-chat.ggmlv3.q4_0.bin` model file.
- Place the model file in the `models/` directory.


---

## Usage

### Running the App
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your browser at `http://localhost:8501`.

### Uploading a Document
1. Upload a PDF file using the file uploader.
2. The app will process the document and initialize the RAG pipeline.

### Asking Questions
1. Enter your query in the input box.
2. The chatbot will generate a response based on the document's content.

---

## Code Overview

### Key Functions
1. **`process_uploaded_file(uploaded_file)`**:
   - Processes the uploaded PDF file and splits it into manageable chunks for vector storage.

2. **`setup_faiss(documents)`**:
   - Sets up FAISS as the vector store using HuggingFace embeddings.

3. **`get_llama_response(query, retriever)`**:
   - Uses the LLaMA model to generate a response based on the retrieved context.

---

## Example Workflow
1. Upload a PDF document (e.g., "user_manual.pdf").
2. Ask a question like:
   ```
   What is the warranty period mentioned in the manual?
   ```
3. Receive a chatbot-generated response based on the uploaded document.

---

## Limitations
- The model's response is limited by the context retrieved and the LLaMA model's token limit.
- Large PDFs may take time to process depending on their size and complexity.

---

## Future Enhancements
- Add support for other file formats (e.g., `.docx`).
- Integrate with a larger model for more accurate responses.
- Optimize the vector retrieval process for better performance.

---

