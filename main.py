import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from PyPDF2.errors import PdfReadError

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError(
        "Google API Key not found. Please check your environment settings.")

genai.configure(api_key=google_api_key)


logging.basicConfig(level=logging.INFO)


def get_pdf_content(docs):
    text = ""
    for pdf in docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                    else:
                        logging.warning(
                            f"Could not extract text from page {page_num + 1} in {pdf.name}")
                except PdfReadError as e:
                    logging.error(
                        f"Error extracting text from page {page_num + 1} in {pdf.name}: {e}")
        except Exception as e:
            logging.error(f"Error processing file {pdf.name}: {e}")
    if not text:
        logging.error("No text could be extracted from the provided PDF(s).")
    return text


def get_text__in_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)


def get_store_in_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logging.info("Vector store created and saved locally.")


def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context", and do not provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True)
        logging.info("Vector store loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        st.error(
            "An error occurred while loading the document index. Please try again.")
        return

    docs = vector_store.similarity_search(user_question)

    chain = get_conversation_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response.get("output_text", "No response generated."))


def main():

    st.set_page_config(page_title="PDFs-Chat", layout="wide")
    st.header("Chat with PDF 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")

        docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):

                if not docs:
                    st.error("Please upload at least one PDF file.")
                    return

                raw_text = get_pdf_content(docs)

                if raw_text:
                    text_chunks = get_text__in_chunks(raw_text)
                    get_store_in_vector(text_chunks)
                    st.success(
                        "Processing complete! You can now ask questions.")
                else:
                    st.error(
                        "No text was extracted from the uploaded PDFs. Please try again with a different file.")


if __name__ == "__main__":
    main()
