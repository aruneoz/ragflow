"""
@Time    : 2024/01/09 17:45
@Author  : asanthan
@Descriptor: This is a Demonstration of Distributed RAG Pipeline to process any doc , any layout including multimodal LLM Unstructured.io PDF Loader
"""
import base64
import os
from collections import Counter
from typing import List, Generator
import pathlib
import gcsfs
from neumai.Shared import CloudFile
from neumai.Shared.NeumDocument import NeumDocument
from neumai.Shared.LocalFile import LocalFile
from neumai.Loaders.Loader import Loader
from langchain.document_loaders import UnstructuredPDFLoader,UnstructuredFileLoader
from unstructured.cleaners.core import clean, clean_non_ascii_chars
from unstructured.documents.elements import NarrativeText, Title, Image, Table
from unstructured.partition.pdf import partition_pdf
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel
from unstructured.partition.text_type import sentence_count

from utils.interop.DocumentTransformer import document_transformer_to_llamaIndex


class UnstructuredLoaderException(Exception):
    """Raised if establishing a connection to GCS Blob fails"""
    pass
class UnstructuredLoader(Loader):
    """
    UnstructuredLoader Loader

    Loads PDF files leveraging UnstructuredLoader.

    Attributes:
    -----------

    None

    """

    @property
    def loader_name(self) -> str:
        return "UnstructuredLoader"

    @property
    def required_properties(self) -> List[str]:
        return []

    @property
    def optional_properties(self) -> List[str]:
        return []

    @property
    def available_metadata(self) -> List[str]:
        return []

    @property
    def available_content(self) -> List[str]:
        return []



    # Gemini Vision Pro

    def generateImageDescription(self,image1):
        model = GenerativeModel("gemini-pro-vision")
        with open(image1, "rb") as image_file:
            image_data = image_file.read()

        encoded_image = base64.b64encode(image_data).decode("utf-8")
        image = generative_models.Part.from_data(data=base64.b64decode(encoded_image), mime_type="image/jpeg")

        responses = model.generate_content(
            [image, """You are Expert AI Agent , whose task is to understand the provided images and generate a Image Caption and Image brief description of the image limit to 512 characters. If you can't answer then generate empty text """],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0,
                "top_p": 1,
                "top_k": 32
            },
        )
        return responses

    # Probably worth re-writing directly on top of pypdf to get access
    # to more metadata including images, tables, etc.
    def load(self, file: CloudFile) -> Generator[NeumDocument, None, None]:
        """Load data into Document objects."""
        try:
            print(f"processing {file.file_identifier} ")
                #fs = gcsfs.GCSFileSystem(project='greenfielddemos')
                #with fs.open(file.file_identifier, 'rb') as f:
                    # loader = UnstructuredPDFLoader(file.file_path,strategy="fast",include_metadata=True,chunking_strategy="by_title",multipage_sections=True)
                    # documents = loader.load()
                    # print(documents[0].metadata.values())
            model_name = "yolox"
                    #elements_fast = partition_pdf(file.file_path, strategy="hi_res",mode="elements",include_metadata=True,chunking_strategy="by_title",multipage_sections=True,pdf_infer_table_structure=True)
            elements_fast = partition_pdf(file.file_identifier, strategy="hi_res", mode="elements", include_metadata=True,combine_text_under_n_chars=512,
                                                  model_name=model_name, infer_table_structure=True , extract_images_in_pdf=True, image_output_dir_path=f"images/{file.id}/")

            tables = [el for el in elements_fast if el.category == "Table"]
                    # join the file and document metadata objects
                    #print(elements_fast)
            docs = []
            imgDescription=""
            docCat=""
            for doc in elements_fast:
                        # yield NeumDocument(id=file.id, content=doc.text, metadata=doc.metadata.to_dict())

                        neuamDoc = ""
                        if(len(doc.text.split())> 5):
                            # files = [f for f in pathlib.Path().glob(f"images/{file.id}/figure-{doc.metadata.page_number}-*.jpg")]
                            #print(doc.metadata.image_path)
                            if(doc.category!=None):
                                docCat = doc.category
                            else:
                                if(isinstance(doc, NarrativeText)):
                                    docCat='NarrativeText'
                                if (isinstance(doc, Title)):
                                    docCat = 'Title'
                                docCat='Unknown'

                            print(doc.category)
                            metadata = doc.metadata.to_dict()
                            metadata['elementCategory'] = docCat

                            if("image_path" in metadata):
                                print("Inside Image")
                                response = self.generateImageDescription(doc.metadata.image_path)
                                imgDescription = response.text

                                if(imgDescription!=""):
                                    print(imgDescription)
                                    doc.text = doc.text + 'Image Description:' + imgDescription
                                    metadata['imageDescription'] = imgDescription
                                    imgDescription=""


                            docCat = ""

                            neuamDoc = NeumDocument(id=doc.id,
                                                   content=clean_non_ascii_chars(doc.text),
                                                   metadata=metadata)
                                #self.generateRagDatasetGenerator(neuamDoc,file.id+"-"+doc.id)
                            #if(isinstance(doc, NarrativeText) or isinstance(doc, Table) or isinstance(doc, Image)):
                            yield neuamDoc




        except Exception as e:
            raise UnstructuredLoaderException(f"Error Processing Documents. See Exception: {e}")
        return True

    def config_validation(self) -> bool:
        return True