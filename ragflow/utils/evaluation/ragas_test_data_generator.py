from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import RunConfig
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain.document_loaders import PyPDFLoader
#
#load documents again to avoid any kind of bias
loader = PyPDFLoader(file_path="/Users/asanthan/work/development/llm/ragflow/ragflow/data/pdf/crs_help_materials_SynXis.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
documents = text_splitter.split_documents(documents)
len(documents)
#
#
generator_llm = VertexAI(model="gemini-pro")
critic_llm = VertexAI(model="gemini-pro")
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko")

generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )
run_config=RunConfig(max_retries=1, max_wait=90)
testset = generator.generate_with_langchain_docs(documents, test_size=1, distributions={simple: 1.0}, with_debugging_logs=True,raise_exceptions=True,run_config=run_config)
df=testset.to_pandas()
print(df.head(10))