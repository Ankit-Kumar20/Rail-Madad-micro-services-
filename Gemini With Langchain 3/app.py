import streamlit as st
from langchain.chains import create_history_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##setup Streamlit
st.title("conversational RAG With PDF uploads and chat history")
st.write("Upload PDF's and Chat with thier content")

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
session_id = st.text_input("Session ID",value="default_session")

if 'store' not in st.session_state:
    st.session_state.store ={}
    