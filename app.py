import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.chains import RetrievalQA


# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI PDF Chatbot (Gemini + RAG)")
st.write("Upload a PDF and ask questions about its contents.")

# ----------------------------------
# GEMINI API KEY
# ----------------------------------
api_key = st.sidebar.text_input(
    "Enter Google Gemini API Key",
    type="password",
)

if not api_key:
    st.warning("Please enter your Gemini API Key.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ----------------------------------
# FILE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    st.success(f"Loaded {len(documents)} pages.")

    # ----------------------------------
    # SPLIT DOCUMENT
    # ----------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(documents)

    # ----------------------------------
    # CREATE EMBEDDINGS
    # ----------------------------------
    with st.spinner("Creating Vector Database..."):

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        vectorstore = FAISS.from_documents(
            docs,
            embeddings,
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # ----------------------------------
    # LOAD GEMINI MODEL
    # ----------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    # ----------------------------------
    # CREATE QA CHAIN
    # ----------------------------------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    st.success("PDF processed successfully!")

    st.divider()

    question = st.text_input("Ask a question about the PDF")

    if st.button("Ask"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:

            with st.spinner("Thinking..."):

                result = qa.invoke({"query": question})

            st.subheader("Answer")

            st.write(result["result"])

            st.subheader("Source Pages")

            shown = set()

            for doc in result["source_documents"]:

                page = doc.metadata.get("page", "Unknown")

                if page not in shown:

                    shown.add(page)

                    st.markdown(f"### 📄 Page {page + 1}")

                    st.info(doc.page_content[:500] + "...")
