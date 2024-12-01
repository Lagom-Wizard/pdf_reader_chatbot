import streamlit as st
from pdf_chatbot import process_pdf, get_answer

st.title("PDF-Based Chatbot")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Load and process the PDF
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    vector_store = process_pdf("temp.pdf")
    st.success("PDF uploaded and processed successfully!")

# Chat interface
user_query = st.text_input("Ask a question about the PDF:")
if user_query:
    if 'vector_store' in locals():
        response = get_answer(vector_store, user_query)
        st.write("Answer:", response)
    else:
        st.warning("Please upload and process a PDF first.")
