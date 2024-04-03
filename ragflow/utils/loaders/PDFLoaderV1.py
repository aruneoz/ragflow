from typing import List, Generator, Any, Optional
from neumai.Shared.NeumDocument import NeumDocument
from neumai.Shared.LocalFile import LocalFile
from neumai.Shared import CloudFile
from neumai.Loaders.Loader import Loader
from langchain.document_loaders import PyPDFLoader
from pydantic import Field
from unstructured.cleaners.core import clean, clean_non_ascii_chars
# from ragas.testset.generator import TestsetGenerator
# from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
# from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


class PDFLoaderV1(Loader):
    """
    PyPDF Loader

    Loads PDF files leveraging PyPDF.

    Attributes:
    -----------

    synthetic_data : bool  Boolean flag to indicate whether we need to generate synthetic data

    dataset_table   : str  Table to store synthetic data

    embedding_model : str Embedding model for synthetic data

    generator_model: str Generator model for synthetic data

    critique_model: str Critique model for synthetic data

    project_id: str Project id for the Vertex AI services

    location: str Location for the Vertex AI Services

    test_set: int No of test data to generate

    """

    syn_data: bool = Field(..., description="Boolean flag to indicate whether we need to generate synthetic data [required]")
    dataset_table: str = Field(..., description="Table to store synthetic data [required]")

    embedding_model: Optional[str] = Field(default='textembedding-gecko', description="Optional embedding model for synthetic data.")

    generator_model: Optional[str] = Field(default='text-bison',
                                           description="Optional generator model for synthetic data..")

    critique_model: Optional[str] = Field(default='gemini-pro',
                                           description="Optional Critique model for synthetic data.")

    project_id: Optional[str] = Field(default='greenfielddemos', description="Optional project id.")

    location: Optional[str] = Field(default='us-central1', description="Optional location.")

    test_set: Optional[int] = Field(default=10, description="Optional no.of test sets to generate.")

    @property
    def loader_name(self) -> str:
        return "PDFLoaderV1"

    @property
    def required_properties(self) -> List[str]:
        return ["syn_data", "dataset_table"]

    @property
    def optional_properties(self) -> List[str]:
        return []

    @property
    def available_metadata(self) -> List[str]:
        return []

    @property
    def available_content(self) -> List[str]:
        return []

    # Probably worth re-writing directly on top of pypdf to get access
    # to more metadata including images, tables, etc.
    def load(self, file: CloudFile) -> Generator[NeumDocument, None, None]:
        """Load data into Document objects."""
        loader = PyPDFLoader(file_path=file.file_identifier)
        documents = loader.load()

        if(self.syn_data):
            self.load_synthentic_data(documents)

        # join the file and document metadata objects
        for doc in documents:
            metadata = doc.metadata
            metadata['filename'] = file.file_identifier
            metadata['elementCategory'] = 'NarrativeText'
            print(metadata)
            # del metadata['text']
            yield NeumDocument(id=file.id, content=clean(clean_non_ascii_chars(doc.page_content),extra_whitespace=True, dashes=True,bullets=True), metadata=metadata)

    def config_validation(self) -> bool:
        return True

    # def load_synthentic_data(self, documents: Any):
    #     generator_llm = VertexAI(model=self.generator_model)
    #     critic_llm = VertexAI(model=self.critique_model)
    #     embeddings = VertexAIEmbeddings(model_name=self.embedding_model,project=self.project_id,location=self.location)
    #
    #     generator = TestsetGenerator.from_langchain(
    #         generator_llm,
    #         critic_llm,
    #         embeddings
    #     )
    #
    #     generator.adapt(language="english", evolutions=[simple, multi_context, reasoning])
    #     generator.save(evolutions=[simple, multi_context, reasoning])
    #
    #     # Change resulting question type distribution
    #     distributions = {
    #         simple: 0.5,
    #         multi_context: 0.4,
    #         reasoning: 0.1
    #     }
    #
    #     # use generator.generate_with_llamaindex_docs if you use llama-index as document loader
    #     testset = generator.generate_with_langchain_docs(documents, 10, distributions,True)
    #     df=testset.to_pandas()
    #     print(df.head(10))



