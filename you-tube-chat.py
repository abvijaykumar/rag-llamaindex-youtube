import streamlit as st
import os
import os.path

from dotenv import load_dotenv
from llama_index.response.pprint_utils import pprint_response
from llama_index.llms import OpenAI
from llama_index import download_loader
from llama_index import VectorStoreIndex, load_index_from_storage, ServiceContext
#from llama_hub.youtube_transcript import YoutubeTranscriptReader
#from llama_index.readers import YoutubeTranscriptReader
from llama_hub.youtube_transcript import YoutubeTranscriptReader

load_dotenv()

storage_path = "./vectorstore"

llm = OpenAI(temperature=0.1, model="gpt-4-turbo-preview")
service_context = ServiceContext.from_defaults(llm=llm)


# with st.sidebar:
#     st.title("Youtube")
#     question_text = st.text_area(label="Youtube URL")

# with st.sidebar:
#     st.button(label="Load URL", on_click=lambda: loadYoutubeURL())
    

url="https://www.youtube.com/watch?v=NBIRzSGHFro"

loader = YoutubeTranscriptReader()
documents = loader.load_data(ytlinks=[url])

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=storage_path)    


st.title("Ask the Youtube")
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            pprint_response(response, show_source=True)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) 