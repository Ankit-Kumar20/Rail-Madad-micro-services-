import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
# import pytesseract
from PIL import Image
import google.generativeai as genai

# pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key='AIzaSyB8XuTdGSDDGuCZa2fb1lvRbnNgoi2h9SU')

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##setup Streamlit
st.title("conversational RAG With PDF uploads and chat history")
st.write("Upload PDF's and Chat with thier content")

##
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
session_id = st.text_input("Session ID",value="default_session")

if 'store' not in st.session_state:
    st.session_state.store ={}


user_input = st.text_input("Your question:")
uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response_img = model.generate_content(image)
    
    #split and create embedding for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(response_img.text)
    vectorstore = Chroma.from_texts(splits,embeddings)
    retriever = vectorstore.as_retriever()

    image_prompt=(
            "Give a chat history and the latest user question"
            "which might refernece context in the chat history,"
            "formulate a standalone question whcih can be understood"
            "without the chat history,Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    image_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",image_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,image_prompt)

    system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer,say that you"
            "don't know.Use three sentences maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )
    qa_prompt =ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"), 
            ]
        )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    # def get_session_history(session:str)->BaseChatMessageHistory:
    #         if session_id not in st.session_state.store:
    #             st.session_state.store[session_id]=ChatMessageHistory()
    #         return st.session_state.store[session_id]
        
    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    session_history=get_session_history(session_id)
    response_img = conversational_rag_chain.invoke(
        {"input":user_input},
        config ={
            "configurable":{"session_id":session_id}
        }
    )
    ##st.write(st.session_state.store)
    st.write("Assistant:",response_img['answer'])
    st.write("Chat History:",session_history.messages)

if user_input is not None:

    #split and create embedding for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_text(user_input)
    try:
        vectorstore = Chroma.from_texts(splits,embeddings)
        retriever = vectorstore.as_retriever()
 
        txt_prompt=(
                "Give a chat history and the latest user question"
                "which might refernece context in the chat history,"
                "formulate a standalone question whcih can be understood"
                "without the chat history,Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
        txt_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",txt_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,txt_prompt)

        system_prompt = (
                "You are an assistant for question-answering tasks."
                "Use the following pieces of retrieved context to answer"
                "the question. If you don't know the answer,say that you"
                "don't know.Use three sentences maximum and keep the "
                "answer concise"
                "\n\n"
                "{context}"
            )
        qa_prompt =ChatPromptTemplate.from_messages(
                [
                    ("system",system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}"), 
                ]
            )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        # def get_session_history(session:str)->BaseChatMessageHistory:
        #         if session_id not in st.session_state.store:
        #             st.session_state.store[session_id]=ChatMessageHistory()
        #         return st.session_state.store[session_id]
            
        conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

        session_history=get_session_history(session_id)
        response_txt = conversational_rag_chain.invoke(
            {"input":user_input},
            config ={
                "configurable":{"session_id":session_id}
            }
        )
        ##st.write(st.session_state.store)
        st.write("Assistant:",response_txt['answer'])
        st.write("Chat History:",session_history.messages)
    except:
         st.write("")
