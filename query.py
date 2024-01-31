import sys
import os
import logging
import random
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    download_loader,
    PromptHelper,
    StorageContext,
    load_index_from_storage,
    StorageContext,
    KnowledgeGraphIndex,
)
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding, LangchainEmbedding
from llama_index.response.notebook_utils import display_response
from pathlib import Path
from llama_index import download_loader
from llama_index.llms import LlamaCPP
from llama_index.text_splitter import SentenceSplitter
from langchain_community.llms import CTransformers


llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

#embed_model = OpenAIEmbedding() #sentence-transformers/all-roberta-large-v1
embed_model = HuggingFaceEmbedding("sentence-transformers/all-roberta-large-v1") #local:BAAI/bge-small-en-v1.5
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

username = "neo4j"
password = "7q0G7tP_ZxJGK-HH87pNoLMqcCO4sDN2cQnPIASwzLw"
url = "neo4j+s://39992655.databases.neo4j.io"
database = "neo4j"

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)


# Storage Context
storage_context = StorageContext.from_defaults(
    persist_dir="./graph_chatbot", graph_store=graph_store
)

# KG Index
kg_index = load_index_from_storage(
    storage_context=storage_context,
    service_context=service_context,
    max_triplets_per_chunk=10,
    verbose=True,
)

kg_index_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)


# Graph RAG Retriever

#


# Graph RAG Query Engine

# graph_rag_query_engine = RetrieverQueryEngine.from_args(
#     graph_rag_retriever,
#     service_context=service_context,
#     verbose=True,
# )

# # Query tools

# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=graph_rag_query_engine,
#         metadata=ToolMetadata(
#             name="......",
#             description=".....",
#         ),
#     )
# ]

# Chatbot
from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
# chat_engine = ReActAgent.from_tools(
#     query_engine_tools, llm=llm, memory=memory, verbose=True
# )

chat_engine = kg_index.as_chat_engine(
    chat_mode="react",
    memory=memory,
    verbose=True,
)

response = chat_engine.chat('According to the labor code what is Dolisa ?')
print(response)