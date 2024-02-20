"""
@Time    : 2024/01/19 17:45
@Author  : asanthan
@File    : main.py
@Descriptor: This is backend pipeline API service for implementing RAGFlow Multimodal Framework
"""


import os
from datetime import datetime
from typing import List

from langchain.llms.vertexai import VertexAI
from neumai.DataConnectors import WebsiteConnector
from neumai.Loaders import HTMLLoader
from tasks import data_extraction
from pipelines.TriggerSyncTypeEnum import TriggerSyncTypeEnum
import tasks
from utils.connectors.GCSBlobConnectors import  GCSBlobConnector
from flask import Flask
from neumai.Shared import Selector, NeumDocument, CloudFile
#
# from utils.loaders.DocAILoader import DocAILoader
from utils.loaders.UnstructuredLoader import UnstructuredLoader
from neumai.Chunkers.RecursiveChunker import RecursiveChunker
from neumai.Sources import SourceConnector
from utils.embeddings import VertexAIEmbed
from utils.sink import AlloyDBVectorStore
from pipelines import Pipeline, TriggerSyncTypeEnum


def create_app():
  global source, pinecone_sink, gcsconnector, vertexai_embed
  flask_app = Flask(__name__)
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
    loader=UnstructuredLoader(),
    chunker=RecursiveChunker(chunk_size=500,
                             chunk_overlap=50,
                             batch_size=1000,
                             separators=["\n\n", " ", ""])
  )

  print("Going to connect to Vector Store ..." + database_host )

  pgVector_sink = AlloyDBVectorStore.AlloyDBSink(
    database_host=database_host,  # "34.170.81.51",
    database_port=database_port,  # "5432",
    database_user=database_user,  # "admin",
    database_pwd=database_pwd,  #
    database_name=database_name,  # "pgvector",
    database_table_name=database_table_name,  # "neumai_vector_test",

  )

  vertexai_embed = VertexAIEmbed.VertexAIEmbed(api_key="<VertexAI AI KEY>")

  @flask_app.route('/rag/create')
  def run_rag():
    # tasks.data_extraction(pipeline_model=pipeline.as_pipeline_model(),extract_type=TriggerSyncTypeEnum.full)
    print("Inside ")
    pipeline = Pipeline.Pipeline(
      sources=[source],
      embed=vertexai_embed,
      sink=pgVector_sink,

    )
    from kombu.exceptions import TimeoutError

    print("Going to Invoke Pipeline Task******************")
    response=data_extraction.apply_async(
      kwargs={"pipeline_model": pipeline.as_pipeline_model(), "extract_type": TriggerSyncTypeEnum.TriggerSyncTypeEnum.full},
      queue="data_extraction",retry=True, retry_policy={
    'max_retries': 3,
    'retry_errors': (TimeoutError, ),
    })

    return f"Task Scheduled withj id {response}"


  return flask_app


# gcs_connector =  GCSBlobConnector(
#       connection_string = "gcs_demo",
#       bucket_name="neumai",
#       batch_size=5,
#
#   )



# source = SourceConnector(
#   data_connector = gcs_connector,
#   loader = DocAILoader(gcs_output="gs://bnsf",doc_ai_parser="projects/779370283097/locations/us/processors/e79a65ddabeb5d3e"),
#   chunker = RecursiveChunker()
# )
#


# website_connector =  WebsiteConnector(
#     url = "https://www.neum.ai/post/retrieval-augmented-generation-at-scale",
#     selector = Selector(
#         to_metadata=['url']
#     )
# )
#
# source = SourceConnector(
#   data_connector=website_connector,
#   loader=HTMLLoader(),
#   chunker=RecursiveChunker()
# )



from neumai.SinkConnectors import PineconeSink












#
# #tasks.data_extraction(pipeline_model=pipeline.as_pipeline_model(),extract_type=TriggerSyncTypeEnum.full)
#
# data_extraction.apply_async(
# 	kwargs={"pipeline_model":pipeline.as_pipeline_model(), "extract_type":TriggerSyncTypeEnum.full},
# 	queue="data_extraction"
# )

#print(f"Vectors stored: {pipeline.run()}")
# query="Scope 2 GHG — location-based in Metric tons CO2e"
# results=pipeline.search(query=query, number_of_results=3)
#
# #results=pipeline.search(query="Information related to Termination Clause ", number_of_results=3)
# context=""
# for result in results:
#   #print(f"Search Result: {result.metadata['text']}")
#   context = context + '\n\n'.join({result.metadata['text']})
#
# llm = VertexAI(
#     model_name="text-bison",
#     max_output_tokens=256,
#     temperature=0.1,
#     top_p=0.8,
#     top_k=40,
#     verbose=True,
# )
#
# print(f"Answer the following query {query} from the following context {context}")
#
# print(llm(f"Answer the following query {query} from the following context {context}"))