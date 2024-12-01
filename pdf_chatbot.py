import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set API keys
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Function to process PDF and create FAISS index
def process_pdf(pdf_path):
    # Read text from PDF
    reader = PdfReader(pdf_path)
    raw_text = ''
    for page in reader.pages:
        raw_text += page.extract_text() or ''
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    
    # Create embeddings and store in FAISS
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Load QA chain for querying
def get_answer(vector_store, query):
    docs = vector_store.similarity_search(query)
    if docs:
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response
    else:
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
