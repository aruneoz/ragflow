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
from ragflow.utils.loaders.PDFLoaderV1 import PDFLoaderV1

from ragflow.tasks import data_extraction
from ragflow.pipelines.TriggerSyncTypeEnum import TriggerSyncTypeEnum
from ragflow.utils.connectors.GCSBlobConnectors import  GCSBlobConnector
from flask import Flask, request
from neumai.Shared import Selector, NeumDocument, CloudFile
#
# from utils.loaders.DocAILoader import DocAILoader
from ragflow.utils.loaders.UnstructuredLoader import UnstructuredLoader
from neumai.Chunkers.RecursiveChunker import RecursiveChunker
from ragflow.utils.connectors.SourceConnector import SourceConnector
from ragflow.utils.embeddings import VertexAIEmbed
from ragflow.utils.sink import AlloyDBVectorStore
from ragflow.pipelines import Pipeline, TriggerSyncTypeEnum


def create_app():
  global source, pinecone_sink, gcsconnector, vertexai_embed
  flask_app = Flask(__name__)
  # gcs_bucket = os.environ['GCS_BUCKET']
  # gcs_pdf_batch_size = os.environ['PDF_BATCH_SIZE']
  database_host = os.environ['VECTOR_HOST']
  database_port = os.environ['VECTOR_PORT']
  database_user = os.environ['VECTOR_USER']
  database_pwd = os.environ['VECTOR_PWD']
  database_name = os.environ['VECTOR_DB']
  database_table_name = os.environ['VECTOR_COLLECTIONS']



  @flask_app.route('/rag/create', methods = ['POST'])
  def run_rag():
    # tasks.data_extraction(pipeline_model=pipeline.as_pipeline_model(),extract_type=TriggerSyncTypeEnum.full)
    print("Inside ")
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):

        data = request.json

        gcs_connector = GCSBlobConnector(
          connection_string="gcs_demo",
          bucket_name=f'{data.get("bucket_name")}',
          batch_size=f'{data.get("pdf_batch_size")}',

        )

        if data.get("pdf_loader_strategy").lower() == 'basic':
          print("Inside Basic")
          source = SourceConnector(
            data_connector=gcs_connector,
            loader=PDFLoaderV1(),
            chunker=RecursiveChunker(chunk_size=data.get("doc_chunk_size"),
                                     chunk_overlap=data.get("doc_chunk_overlap"),
                                     )
          )
        else:
          source = SourceConnector(
            data_connector=gcs_connector,
            loader=UnstructuredLoader(strategy="multimodal",model_name=f'{data.get("pdf_table_model")}'),
            chunker=RecursiveChunker(chunk_size=data.get("doc_chunk_size"),
                                     chunk_overlap=data.get("doc_chunk_overlap"),
                                     )
          )

        print("Going to connect to Vector Store ..." + database_host)

        pgVector_sink = AlloyDBVectorStore.AlloyDBSink(
          database_host=database_host,  # "34.170.81.51",
          database_port=database_port,  # "5432",
          database_user=database_user,  # "admin",
          database_pwd=database_pwd,  #
          database_name=database_name,  # "pgvector",
          database_table_name=data.get("vector_db_table_name"),  # "neumai_vector_test",

        )

        vertexai_embed = VertexAIEmbed.VertexAIEmbed(api_key="<VertexAI AI KEY>", chunk_size=data.get("doc_chunk_size"))


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

        return f"Task Scheduled with id {response}"

    else:

        return "Content type is not supported."


  return flask_app


