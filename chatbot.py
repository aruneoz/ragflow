import streamlit as st
import requests
import json
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from neumai.Chunkers import RecursiveChunker

from ragflow.pipelines.TriggerSyncTypeEnum import TriggerSyncTypeEnum
from datetime import datetime
from typing import List

from langchain.llms.vertexai import VertexAI
from neumai.DataConnectors import WebsiteConnector
from neumai.Loaders import HTMLLoader
from ragflow.tasks import data_extraction
from ragflow.pipelines.TriggerSyncTypeEnum import TriggerSyncTypeEnum
import ragflow.tasks
from ragflow.utils.connectors.GCSBlobConnectors import  GCSBlobConnector
from flask import Flask
from neumai.Shared import Selector, NeumDocument, CloudFile
from ragflow.pipelines import Pipeline, TriggerSyncTypeEnum
from ragflow.utils.connectors.SourceConnector import SourceConnector
from ragflow.utils.embeddings import VertexAIEmbed
from ragflow.utils.loaders.UnstructuredLoader import UnstructuredLoader
from ragflow.utils.sink import AlloyDBVectorStore

gcs_bucket = os.environ['GCS_BUCKET']
gcs_pdf_batch_size = os.environ['PDF_BATCH_SIZE']
database_host = os.environ['VECTOR_HOST']
database_port = os.environ['VECTOR_PORT']
database_user = os.environ['VECTOR_USER']
database_pwd = os.environ['VECTOR_PWD']
database_name = os.environ['VECTOR_DB']
database_table_name = os.environ['VECTOR_COLLECTIONS']

gcs_connector = GCSBlobConnector(
    connection_string="gcs_demo",
    bucket_name=f"{gcs_bucket}",
    batch_size=f"{gcs_pdf_batch_size}",

  )

source = SourceConnector(
    data_connector=gcs_connector,
    loader=UnstructuredLoader(strategy="table",model_name='yolox'),
    chunker=RecursiveChunker(chunk_size=500,
                             chunk_overlap=50,
                             batch_size=1000,
                             separators=["\n\n", " ", ""])
  )

pinecone_sink = AlloyDBVectorStore.AlloyDBSink(
    database_host=database_host,  # "34.170.81.51",
    database_port=database_port,  # "5432",
    database_user=database_user,  # "admin",
    database_pwd=database_pwd,  # "sUmmertime123$",
    database_name=database_name,  # "pgvector",
    database_table_name=database_table_name,  # "neumai_vector_test",

  )

vertexai_embed = VertexAIEmbed.VertexAIEmbed(api_key="<VertexAI AI KEY>")

pipeline = Pipeline.Pipeline(
    sources=[source],
    embed=vertexai_embed,
    sink=pinecone_sink,

  )

def multiturn_generate_content(context):
    config = {
        "max_output_tokens": 2048,
        "temperature": 0.1,
        "top_p": 1
    }
    print(context)
    model = GenerativeModel("gemini-pro")
    chat = model.start_chat()
    return(chat.send_message(context, generation_config=config))

query_params = st.experimental_get_query_params()

if "context" not in st.session_state:
    st.session_state["context"] = ""
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "You are a helpful assistant with the knowledge of providing instructions from a tutorail document that answers questions based on the following context:{} . Analyze step by step and answer the question relevant to the context and if there are similar context , the pick the one which is most relevant. The output must be formatted in the MD format and elaborate the answers as step by step instructions. If the context contains the Image Description section and user requests diagram or schematic , then return no response.  Limit your answers to the context provided."

with st.sidebar:
    st.title("Sabre Chatbot")
    st.markdown("This is the demonstration of RAGFlow framework based on NeumAI , helps you connect and synchronize your data to a vector database. Simply set up a pipeline and let RAGFlow automatically synchronize your data.")
    st.markdown("This app allows you to chat with the data connected to your pipeline")
    include_context = st.toggle("Include context in messages", True)
    system_prompt = st.toggle("Change system prompt", False)
    st.markdown("Developed by Arun Santhanagopalan")
    if system_prompt:
        initial_value = st.session_state["system_prompt"]
        st.session_state["system_prompt"] = st.text_area(label="System Prompt", value=initial_value, height=200)

st.title("Multimodal Chat with your Document")
st.caption("Simple chatbot based on Neum AI and VertexAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "system", "content": st.session_state["system_prompt"].format(st.session_state["context"])})
    st.session_state["messages"].append({"role": "assistant", "content": "How can I help you?", "context":""})

for msg in st.session_state.messages:
    if(msg["role"] != "system"):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg["context"] != "" and include_context:
                with st.expander("Context"):
                    st.text(msg["context"])

if prompt := st.chat_input():

    # Get context from Neum AI


    json_response = pipeline.search(query=prompt, number_of_results=5)
    context=""
    for result in json_response:
      unfiltered = context + '\n\n'.join({result.metadata['text']})
      print(unfiltered)
    #print(f"Search Result: {result.metadata['text']}")
      #if ('text' in result.metadata and (result.metadata['elementCategory']== 'NarrativeText' or result.metadata['elementCategory']== 'Image' or result.metadata['elementCategory']== 'Table' )):
      context = context + '\n\n'.join({result.metadata['text']})

        # if ('text_as_html' in result.metadata):
        #     context = context + '\n\n'.join({result.metadata['text_as_html']})
        # if ('imageDescription' in result.metadata):
        #     context = context + '\n\n Image Description:'.join({result.metadata['imageDescription']})
        #     imgUrl = result.metadata['image_path']
      # if('text_as_html' in result.metadata):
      #     context = context + '\n\n'.join({result.metadata['text']}) + '\n\n'.join({result.metadata['text_as_html']})





    # Extract the 'results' list from the JSON response
    # Additional metadata fields are available as well as id and score.
    #results = [ json.loads(result)['metadata']['text'] for result in json_response]
    st.session_state["context"] = context
    st.session_state["messages"][0] = {"role": "system", "content": st.session_state["system_prompt"].format(st.session_state["context"])}

    # Request to Open AI
    # client = OpenAI(
    #     api_key=openai_api_key
    # )

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    messages = [{k: v for k, v in item.items() if k != 'context'} for item in st.session_state.messages]
    response = multiturn_generate_content(str(messages))
    msg = response
    st.session_state.messages.append({"role":"assistant", "content":msg.text, "context":st.session_state["context"]})
    with st.chat_message("assistant"):
        st.markdown(msg.text)
        # if 'imgUrl' in locals():
        #     st.image(imgUrl)
        if include_context:
            with st.expander("Context"):
                st.text(st.session_state["context"])