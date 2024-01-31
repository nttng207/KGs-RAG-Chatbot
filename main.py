import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


import streamlit as st

import openai

from llama_index import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    StorageContext,
)
from llama_index.llms import  OpenAI
from llama_index.graph_stores import Neo4jGraphStore
# from llama_index import HuggingFaceEmbedding, LangchainEmbedding
import openai
import os
from llama_index.memory import ChatMemoryBuffer

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext, set_global_service_context

openai.api_key = st.secrets.openai_key
llm = OpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    engine="td2"
)


embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-roberta-large-v1"
)
# embed_model = LangchainEmbedding(HuggingFaceEmbedding("sentence-transformers/all-roberta-large-v1")) #local:BAAI/bge-small-en-v1.5
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

set_global_service_context(service_context)

username = "neo4j"
password = "7q0G7tP_ZxJGK-HH87pNoLMqcCO4sDN2cQnPIASwzLw"
url = "neo4j+s://39992655.databases.neo4j.io"
database = "neo4j"

space_name = "rag_workshop"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)
path_cur = os.getcwd()
storage_graph = StorageContext.from_defaults(persist_dir = path_cur+'\graph_chatbot',graph_store = graph_store)


kg_index = load_index_from_storage(
    storage_context = storage_graph,
    service_context = service_context,
    max_triplets_per_chunk = 10,
    llm = llm
)


kg_query_engine = kg_index.as_query_engine(
    include_text=False,
    response_mode="tree_summarize", #test response !!!
    #embedding_mode="hybrid", #
    similarity_top_k=5,
)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


chat_engine = kg_index.as_chat_engine(
    chat_mode="react",
    memory=memory,
    verbose=True,
)

# utils

#### page

st.set_page_config(
    page_title="Graph RAG Chat Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Demo: Graph RAG Chat Bot")


st.info(
    "This is a demo application RAG Chatbot using for Seminar Object. Dataset is Employment-Manual-January-2023.pdf.",
    icon="📃",
)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me question from the knowledge Employment-Manual-January-2023. **",
        }
    ]

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print('prompt is ',prompt)
            response = chat_engine.chat(prompt)
            st.write("Answer:", response.response)
            st.write("Source:", response.source_nodes)
            # st.write("Answer:", response.source_document)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)

