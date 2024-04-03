from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from llama_index.core.llama_dataset import (
    LabelledRagDataset,
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
)

from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.prompts import PromptType

from llama_index.llms.vertex import Vertex

reader = SimpleDirectoryReader(input_files=["/Users/asanthan/work/development/llm/ragflow/ragflow/data/pdf/crs_help_materials_SynXis.pdf"])
documents = reader.load_data()

q_n_a_prompt = """You are a Question and Answer Assistant and your task is to generate synthetic question and answers for RAG evaluation from the Context information  below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query based only from the context information and try to be accurate.\n"
    output the 
    "Query: {query_str}\n"
    "Answer: """

DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    q_n_a_prompt, prompt_type=PromptType.QUESTION_ANSWER
)
llm = Vertex(model="text-bison",temperature=0.3)

dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=llm,
    num_questions_per_chunk=1,
    text_qa_template=DEFAULT_TEXT_QA_PROMPT,
    show_progress=True
)

rag_dataset = dataset_generator.generate_dataset_from_nodes()
rag_dataset.to_pandas().to_csv("test.csv")