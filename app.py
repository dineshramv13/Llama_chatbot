


import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers


def process_uploaded_file(uploaded_file):
    #Processes the uploaded PDF file and returns extracted text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()

    os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = text_splitter.split_documents(pages)
    return documents


def setup_faiss(documents):
    #Set up FAISS vector store with the extracted documents
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def get_llama_response(query, retriever):
    #Generate a response using the LLaMA model with RAG
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama"
    )

    docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in docs])

    max_context_tokens = 400  # Token limit for context
    context = " ".join(context.split()[:max_context_tokens])

    prompt = (
        f"You are an assistant. Use the following context to answer the query."
        f"If the answer isn't in the context, say 'Answer not available in the context.'\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query}\n"
        f"Answer:"
    )

    response = llm.invoke(prompt)
    return response


# Streamlit App UI
st.title("RAG Chatbot with PDF Upload ")

uploaded_file = st.file_uploader("Upload your document (PDF and text only):", type=["pdf"])
if uploaded_file:
    st.info("Processing uploaded document...")
    documents = process_uploaded_file(uploaded_file)
    if documents:
        vector_store = setup_faiss(documents)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        st.success("Document processed and RAG pipeline initialized!")

        query = st.text_input("Ask a question about the document:")


        response = get_llama_response(query, retriever)
        st.write("**Chatbot Response:**")
        st.write(response)
