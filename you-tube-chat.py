import streamlit as st
import os
import os.path

from dotenv import load_dotenv
from llama_index.response.pprint_utils import pprint_response
from llama_index.llms import OpenAI
from llama_index import download_loader
from llama_index import VectorStoreIndex, load_index_from_storage, ServiceContext
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_hub.youtube_transcript import is_youtube_video

load_dotenv()

storage_path = "./vectorstore"

llm = OpenAI(temperature=0.1, model="gpt-4-turbo-preview")
service_context = ServiceContext.from_defaults(llm=llm)

documents = None

with st.sidebar:
    st.title("Youtube")
    urlTextValue = st.text_input(label="Youtube URL")
    st.button(label="Load URL", on_click=lambda: loadYoutubeURL(urlTextValue))

st.title("Ask the Youtube")
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]


def cleanIndex():
    global documents
    if documents != None :
        st.write("Found Index and Vectors, Cleaning up now...")
        ids = [str(i) for i in range(1, len(docs) + 1)]
        index = VectorStoreIndex.from_documents(documents)
        docsToDelete = index.similarity_search("")
        print(docsToDelete[0].metadata)
        st.write("Count before cleanup", documents._collection.count())
        docsToDelete._collection.delete(ids=[ids[-1]])
        st.write("count after cleanup", documents._collection.count())

def loadYoutubeURL(url):
    global documents
    cleanIndex()
    if is_youtube_video(url) == True :
        with st.spinner("Loading the Index..."):
            print(url)
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(ytlinks=[url])
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=storage_path)
            chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True, verbose=True)
            print(chat_engine)
            st.session_state["chat_engine"] = chat_engine
    else :
        st.error("Please check the youtube URL, it doesn't seem to be valid", icon="🚨")


if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("Prompt recieved")
            chat_engine = st.session_state["chat_engine"]
            print(chat_engine)
            if chat_engine != None :
                response = chat_engine.chat(prompt)
                st.write(response.response)
                pprint_response(response, show_source=True)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) 
            else :
                st.write("Please load a youtube video first...")