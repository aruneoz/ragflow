from langchain.document_loaders import WikipediaLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from testset_generator import TestsetGenerator


topic = "python programming"



from langchain.vectorstores import PGVector
CONNECTION_STRING = "postgresql+psycopg2://admin:sUmmertime123$@34.170.81.51:5432/pgvector"
COLLECTION_NAME="neumai_vector_test"



generator_llm = VertexAI(
    location="europe-west3",
    max_output_tokens=256,
    max_retries=20,
)
embedding_model = VertexAIEmbeddings()

store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model,
)