import streamlit as st 
import os 
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model="gemma2-9b-it")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on provided context only. Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
""")

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Loading and embedding documents..."):
            try:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.loader = PyPDFDirectoryLoader("./research_papers")
                st.session_state.docs = st.session_state.loader.load()
                if not st.session_state.docs:
                    st.warning("No PDF documents found in /research_papers.")
                    return
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.success("Document embedding created successfully!")
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")

# Streamlit UI
st.title("PDF Question Answering with Groq + Langchain")

user_prompt = st.text_input("Enter your question here")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please generate document embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Processing your question..."):
            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_prompt})
            end = time.process_time()

        st.write("**Response:**", response.get("answer", "No answer returned."))
        st.caption(f"Processed in {end - start:.2f} seconds.")

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write("--------------------------------------------------------")
