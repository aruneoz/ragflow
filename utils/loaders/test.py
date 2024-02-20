import hashlib
import os
import traceback
from io import BytesIO
from typing import Optional
#from PIL import Image
import google
from PyPDF2 import PdfReader, PdfWriter
from fitz import fitz
from google.api_core.client_options import ClientOptions
from google.cloud import documentai, storage
from google.cloud.documentai_toolbox.wrappers import document
from google.cloud.documentai_v1 import Document
import base64
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part , Image
from unstructured.cleaners.core import clean, clean_non_ascii_chars
from unstructured.partition.pdf import partition_pdf


def generate(image1):
    model = GenerativeModel("gemini-pro-vision")
    responses = model.generate_content(
        [image1, """You are Expert AI Agent , whose task is to understand the provided images and generate the following
1. If the image contains charts , convert the charts to table data then generate the following section
      image_caption:
      image_data:
      image_file_name:
2. If the image contains instructions for repairs or installation or troubleshooting, then translate instructions into text description
    image_caption:
      image_description: describe what is in the image
      image_file_name:


If you cannot find any of the section , generate empty json  for each section.

Output the extract content as json string."""],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0,
            "top_p": 1,
            "top_k": 32
        },
    )
    return responses
model_name = "yolox"
#doc = fitz.open("../../data/Flood-Endorsement-Comparison.pdf")
elements_fast = partition_pdf("../../data/1.pdf", strategy="hi_res", mode="elements", include_metadata=True,combine_text_under_n_chars=512,
                                                                       model_name=model_name, infer_table_structure=True,extract_images_in_pdf=True,image_output_dir_path="images/testing.pdf/")



tables = [el for el in elements_fast if el.category == "Table"]
print(elements_fast.__dict__)
narritive = [el for el in elements_fast if el.category == "NarrativeText"]
print(narritive[1].text)

print(tables[0].metadata.text_as_html)
# directory="images/testing.pdf/"
#
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         print(f)
#         responses = generate(Part.from_image(Image.load_from_file(f)))
#         print(responses.text)
#         #os.remove(filename)
#



# # tables = [el for el in elements_fast if el.category == "Table"]
# images = [el for el in elements_fast if el.category == "Image"]
# processed_pages=[]
# #processed_images=[]
# #
# for image in images:
#     print("Processing the Page Number : " + str(image.metadata.page_number-1))
#     print(image.text)
#     print("Processing the Page Number : " + str(doc.page_count))
#     page=doc.load_page(image.metadata.page_number-1)
#     imageList = page.getImageList()
#     if imageList:
#         if(image.metadata.page_number not in processed_pages):
#
#             for image_index, img in enumerate(page.getImageList(), start=1):
#                 xref = img[0]
#                 # extract the image bytes
#                 base_image = doc.extractImage(xref)
#                 image_bytes = base_image["image"]
#                 responses = generate(Part.from_data(image_bytes, mime_type="image/png"))
#                 print(responses.text)
#                 processed_pages.append(image.metadata.page_number)

# for image in images:
#     print("Processing the Page Number : " + str(image.metadata.page_number-1))
#     print(image.text)
#     print("Processing the Page Number : " + str(doc.page_count))
#     page=doc.load_page(image.metadata.page_number-1)
#     pixmap = page.get_pixmap(dpi=300)
#     img = pixmap.pil_tobytes(format="PNG", optimize=True, dpi=(150, 150))
#     # pixmap.save("test.png")
#     # image = Image.load_from_file("test.png")
#     if(image.metadata.page_number not in processed_pages):
#         responses = generate(Part.from_data(img, mime_type="image/png"))
#         print(responses.text)
#         processed_pages.append(image.metadata.page_number)
#
# # for page in doc:
# #     page = doc.load_page(0)
# #     pixmap = page.get_pixmap(dpi=300)
# #     img = pixmap.pil_tobytes(format="PNG", optimize=True, dpi=(150, 150))
# #     # pixmap.save("test.png")
# #     # image = Image.load_from_file("test.png")
# #     responses=generate(Part.from_data(img,mime_type="image/png"))
# #     print(responses.text)
#
