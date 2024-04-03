"""
@Time    : 2024/01/09 17:45
@Author  : asanthan
@Descriptor: This is a Demonstration of Distributed RAG Pipeline to process any doc , any layout including multimodal LLM Vertex AI Embeddings
"""


import time
from typing import List, Tuple, Optional
from neumai.EmbedConnectors.EmbedConnector import EmbedConnector
from langchain.embeddings.vertexai import VertexAIEmbeddings
from neumai.Shared.NeumDocument import NeumDocument
from neumai.Shared.Exceptions import OpenAIConnectionException
from pydantic import BaseModel, Field


def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch:],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]

class VertexAIConnectionException(Exception):
    """Raised if establishing a connection to GCS Blob fails"""
    pass

    # Utility functions for Embeddings API with rate limiting

class VertexAIEmbed(EmbedConnector):
    """
    VertexAI Embed Connector

    Facilitates embedding and processing data using GCP VertexAI services. This connector is designed to interact directly with GCP VertexAI's APIs, requiring an API key for authentication and operation.

    Attributes:
    -----------

    max_retries : Optional[int]
        The maximum number of retries for connection attempts with OpenAI's services. Default is 20 retries.

    chunk_size : Optional[int]
        The size of chunks for processing data, defined by the number of items. Default is 512, suitable for large data sets.
    """
    api_key: str = Field(..., description="API key for VertexAI services.")

    max_retries: Optional[int] = Field(20, description="Maximum number of retries for the connection.")

    chunk_size: Optional[int] = Field(512, description="Size of chunks for processing data.")

    @property
    def required_properties(self) -> List[str]:
        return ["api_key"]
    @property
    def embed_name(self) -> str:
        return 'VertexAIEmbed'


    @property
    def optional_properties(self) -> List[str]:
        return ['max_retries', 'chunk_size']

    def validation(self) -> bool:
        """config_validation connector setup"""
        try:
            # Embedding
            EMBEDDING_QPM = 100
            EMBEDDING_NUM_BATCH = 5
            embeddings = CustomVertexAIEmbeddings(
                requests_per_minute=EMBEDDING_QPM,
                num_instances_per_batch=EMBEDDING_NUM_BATCH,
            )
        except Exception as e:
            raise VertexAIConnectionException(f"OpenAI couldn't be initialized. See exception: {e}")
        return True





    def embed(self, documents: List[NeumDocument]) -> Tuple[List, dict]:
        """Generate embeddings with OpenAI"""
        try:
            # Embedding
            EMBEDDING_QPM = 100
            EMBEDDING_NUM_BATCH = 5

            embedding = CustomVertexAIEmbeddings(
                requests_per_minute=EMBEDDING_QPM,
                num_instances_per_batch=EMBEDDING_NUM_BATCH,
            )
            embeddings = []
            print("Inside Embed......")
            print(documents[0])
            texts = [x.content for x in documents]
            # do we want to persist some embeddings if they were able to be wrriten but not another "batch" of them? or should we treat all texts as an atomic operation
            embeddings = embedding.embed_documents(texts=texts)
            # cost_per_token = 0.000000001 # ADA-002 as of Sept 2023
            info = {
                "estimated_cost": str("Not implemented"),
                "total_tokens": str("Not implemented"),
                "attempts_used": str("Not implemented")
            }
        except Exception as e:
            raise VertexAIConnectionException(f"Embedding couldn't be initialized. See exception: {e}")

        return embeddings, info


    def embed_query(self, query: str) -> List[float]:
        """Generate embeddings for a single query using OpenAI"""
        # Embedding
        EMBEDDING_QPM = 100
        EMBEDDING_NUM_BATCH = 5

        embedding = CustomVertexAIEmbeddings(
            requests_per_minute=EMBEDDING_QPM,
            num_instances_per_batch=EMBEDDING_NUM_BATCH,
        )
        return embedding.embed_query(query)