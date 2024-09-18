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


def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

def get_category(response):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_text(response)
    vectorstore = Chroma.from_texts(splits,embeddings)
    retriever = vectorstore.as_retriever()

    cat_prompt = (
        "Assign it to one of the following categories."
        "categories: 'Cleanliness and Hygiene', 'Ticketing Issues', 'Train Delay and Cancellations', 'Catering and Food Quality', 'Amenities and Facilities'"
        "Give me the category to which the provided complaint belongs to."
    )
    cat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",cat_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
    history_aware_retriever = create_history_aware_retriever(llm,retriever,cat_prompt)

    system_prompt = (
            "You are an AI assistant responsible for categorizing and responding to complaints related to their Train journey. Users may describe various issues, which need to be categorized"
            "into one of the following: 'Cleanliness and Hygiene', 'Ticketing Issues', 'Train Delay and Cancellations', 'Catering and Food Quality', 'Amenities and Facilities', and 'Other" "Passenger Behavior'. Analyze the user's complaint, determine the most relevant category, and provide a concise response or acknowledgment of the issue. If the complaint does" "not fit any category, suggest that the user provide more details or select a different category."
            "Give onw word answer"
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

    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    session_history=get_session_history(session_id)
    response_cat = conversational_rag_chain.invoke(
        {"input":response},
        config ={
            "configurable":{"session_id":session_id}
        }
    )

    return response_cat['answer']
    #print(response_cat['answer'])


user_input = st.text_input("Your question:")
uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])


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
        "describe the given image, effectively"
        "You are an advanced AI assistant trained to interpret and describe images in a detailed, accurate, and contextually relevant manner. Your task is to provide clear and comprehensive descriptions of any image input, focusing on the following aspects:"
        "Describing the setting, background, and overall composition of the image."
        "Noting any prominent colors, textures, patterns, and lighting effects."
        "Inferring the possible actions, emotions, or interactions visible in the image."
        "Providing insights into the style, atmosphere, and any notable artistic or technical features."
        "Always aim to be neutral and objective in your description, avoiding any assumptions that go beyond what is visually present. Keep the descriptions concise, yet detailed enough to give a clear mental image to someone who cannot see it."
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
        "Analyze the image and classify the complaint into one of these categories: 'Cleanliness and Hygiene', 'Ticketing Issues', 'Train Delay and Cancellations', 'Catering and Food Quality', 'Amenities and Facilities', and 'Other" "Passenger Behavior'. "
        "Give answer in 4 lines."
        "always tell that your complaint has been routed to the respective department in the last line."
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
        {"input":response_img.text},
        config ={
            "configurable":{"session_id":session_id}
        }
    )
    ##st.write(st.session_state.store)
    st.write("Assistant:",response_img['answer'])
    st.write("Chat History:",session_history.messages)
    st.write("Category: ", get_category(response_img['answer']))


if user_input is not None:

    #split and create embedding for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_text(user_input)
    try:
        vectorstore = Chroma.from_texts(splits,embeddings)
        retriever = vectorstore.as_retriever()
 
        txt_prompt=(
            "You have received a text-based complaint from a user. Carefully read the complaint and assign it to one of the "
            "following categories: 'Cleanliness and Hygiene', 'Ticketing Issues', 'Train Delay and Cancellations', 'Catering and Food Quality', 'Amenities and Facilities', and 'Other" "Passenger Behavior'. After categorizing the complaint, provide a brief response acknowledging the issue and offering any next steps if relevant."
            "formulate a standalone answer in hindi language which can be understood"
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
                "You are an AI assistant responsible for categorizing and responding to complaints related to their Train journey. Users may describe various issues, which need to be categorized" "into one of the following: 'Cleanliness and Hygiene', 'Ticketing Issues', 'Train Delay and Cancellations', 'Catering and Food Quality', 'Amenities and Facilities', and 'Other" "Passenger Behavior'. Analyze the user's complaint, determine the most relevant category, and provide a concise response or acknowledgment of the issue. If the complaint does" "not fit any category, suggest that the user provide more details or select a different category."
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
        st.write("category: ",get_category(response_txt['answer']))

    except:
         st.write("")
