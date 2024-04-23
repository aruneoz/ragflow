"""
@Time    : 2024/01/09 17:45
@Author  : asanthan
@Descriptor: This is a Demonstration of Distributed RAG Pipeline to process any doc , any layout including multimodal LLM Unstructured.io PDF Loader
"""
import base64
from typing import List, Generator
from pydantic import Field
from neumai.Shared import CloudFile
from neumai.Shared.NeumDocument import NeumDocument
from neumai.Loaders.Loader import Loader
from unstructured.cleaners.core import clean, clean_non_ascii_chars
from unstructured.documents.elements import NarrativeText, Title, Image, Table
from unstructured.partition.pdf import partition_pdf
from vertexai.generative_models import GenerativeModel
from vertexai.preview import generative_models


# from langchain_google_vertexai import VertexAI

#from ragflow.utils.interop.DocumentTransformer import document_transformer_to_llamaIndex


class UnstructuredLoaderException(Exception):
    """Raised if establishing a connection to GCS Blob fails"""
    pass
class UnstructuredLoader(Loader):
    """
    UnstructuredLoader Loader

    Loads PDF files leveraging UnstructuredLoader.

   strategy : str
        What is the Strategy you want to use for PDF (basic, tables , multimodal)
   model_name : str
        Name of the model to use for PDF parser

    """

    strategy: str = Field(..., description="Strategy to use for PDF Layout Parser [required]")

    model_name: str = Field(..., description="Model to use [required]")


    @property
    def loader_name(self) -> str:
        return "UnstructuredLoader"

    @property
    def required_properties(self) -> List[str]:
        return ["strategy", "model_name"]

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

            #model_name = "yolox"
            model_name = self.model_name
            print("Using the following Strategy : " + self.strategy + "  with model " + self.model_name)

            if(self.strategy.lower()=="multimodal"):
                elements_fast = partition_pdf(file.file_identifier, strategy="hi_res", mode="elements", include_metadata=True,combine_text_under_n_chars=512,
                                                  model_name=model_name, infer_table_structure=True , extract_images_in_pdf=True, image_output_dir_path=f"images/{file.id}/")
            elif(self.strategy.lower()=="table"):

                elements_fast = partition_pdf(file.file_identifier, strategy="hi_res", mode="elements",
                                              include_metadata=True, combine_text_under_n_chars=512,
                                              model_name=model_name, infer_table_structure=True)
            else:
                elements_fast = partition_pdf(file.file_identifier, strategy="hi_res", mode="elements",
                                              include_metadata=True, combine_text_under_n_chars=512,
                                              model_name=model_name, infer_table_structure=False,
                                              extract_images_in_pdf=False, image_output_dir_path=f"images/{file.id}/")

            # tables = [el for el in elements_fast if el.category == "Table"]
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