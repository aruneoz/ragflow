"""
@Time    : 2024/01/09 17:45
@Author  : asanthan
@Descriptor: This is a Demonstration of Distributed RAG Pipeline to process any doc , any layout including multimodal LLM GCS Source Connector
"""


from datetime import datetime

from google.oauth2 import service_account
from neumai.DataConnectors import DataConnector
from typing import List, Generator, Optional
from google.cloud import storage
from neumai.Shared.LocalFile import LocalFile
from neumai.Shared.CloudFile import CloudFile
from neumai.Shared.Selector import Selector
from PyPDF2 import PdfReader, PdfWriter
import os
from pydantic import Field
import gcsfs

class GCSBlobConnectionException(Exception):
    """Raised if establishing a connection to GCS Blob fails"""
    pass
class GCSBlobConnector(DataConnector):
    """
    GCS Blob data connector

    Extracts data from a GCS Blob container.

    Attributes:
    -----------

    connection_string : str
        Connection string to the Azure Blob
    bucket_name : str
        Name of the GCS Blob bucket you want to extract data from
    selector : Optional[Selector]
        Optional selector object to define what data  should be used to generate embeddings or stored as metadata with the vector.
    #batch_size: str
        PDF batch size for the Loaders

    """

    connection_string: str = Field(..., description="Connection string to connect to GCS Blob [required]")

    bucket_name: str = Field(..., description="Bucket name to connect to [required]")

    batch_size: int = Field(..., description="Batch size for PDF processing [required]")

    selector: Optional[Selector] = Field(Selector(to_embed=[], to_metadata=[]), description="Selector for data connector metadata")

    @property
    def connector_name(self) -> str:
        return "GCSBlobConnector"

    @property
    def required_properties(self) -> List[str]:
        return ["connection_string", "bucket_name","batch_size"]

    @property
    def optional_properties(self) -> List[str]:
        return []

    @property
    def available_metadata(self) -> str:
        return ['name', 'updated','time_created']

    @property
    def schedule_avaialable(self) -> bool:
        return True

    @property
    def auto_sync_available(self) -> bool:
        return False

    @property
    def compatible_loaders(self) -> List[str]:
        return ["AutoLoader", "HTMLLoader", "MarkdownLoader", "CSVLoader", "JSONLoader", "PDFLoader" ,"UnstructuredLoader"]

    def connect_and_list_full(self) -> Generator[CloudFile, None, None]:
        # container = ContainerClient.from_connection_string(
        #     conn_str=self.connection_string, container_name=self.container_name
        # )
        #credentials = service_account.Credentials.from_service_account_file('greenfielddemos-af461fe8c10e.json')
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(self.bucket_name)

        #Process files
        #file_list = container.list_blobs()
        for blob in blobs:
            name = blob.name
            metadata = {
                "id": blob.id,
                "last_modified": blob.updated.isoformat(),

            }
            selected_metadata  = {k: metadata[k] for k in self.selector.to_metadata if k in metadata}
            yield CloudFile(file_identifier=name, metadata=selected_metadata, id = name)

    def connect_and_list_delta(self, last_run:datetime) -> Generator[CloudFile, None, None]:
        #credentials = service_account.Credentials.from_service_account_file('greenfielddemos-af461fe8c10e.json')
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(self.bucket_name)

        #Process files
        file_list = storage_client.list_blobs()
        for file in file_list:
            last_update_date = file.updated
            if(last_run < last_update_date):
                name = file.name
                metadata = {
                    "id": file.id,
                    "last_modified": file.updated.isoformat(),
                    "filename": file.name
                }
                selected_metadata  = {k: metadata[k] for k in self.selector.to_metadata if k in metadata}
                yield CloudFile(file_identifier=name, metadata=selected_metadata, id=name)

    def connect_and_download(self,  cloudFile:CloudFile) -> Generator[LocalFile, None, None]:

        #credentials = service_account.Credentials.from_service_account_file('greenfielddemos-af461fe8c10e.json')
        storage_client = storage.Client()

        bucket = storage_client.bucket(self.bucket_name)


        #client = BlobClient.from_connection_string(conn_str=self.connection_string, container_name=self.container_name, blob_name=cloudFile.file_identifier)
        #with tempfile.TemporaryDirectory() as temp_dir:
        nfs_mount=  os.environ['NFS_MOUNT'] #"/Users/asanthan/ragspersistentstore"
        file_path = f"{nfs_mount}/{self.bucket_name}/{cloudFile.file_identifier}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob = bucket.blob(cloudFile.file_identifier)
        blob.download_to_filename(file_path)
        print("Downloaded the file" + blob.name  + " downloaded to " + file_path)

        if cloudFile.file_identifier.endswith('.pdf'):
            input_pdf = PdfReader(file_path)
            num_batches = round((len(input_pdf.pages)) / self.batch_size)
            if(num_batches==0):
                num_batches=1
            print("Batch Size :" + str(self.batch_size)  +  "  Pages " + str(len(input_pdf.pages)))
            print("The no.of batches :" + str(num_batches))
            for b in range(num_batches):
                    writer = PdfWriter()

                    # Get the start and end page numbers for this batch
                    start_page = b * self.batch_size
                    end_page = min((b + 1) * self.batch_size, len(input_pdf.pages))

                    # Add pages in this batch to the writer
                    for i in range(start_page, end_page):
                        writer.add_page(input_pdf.pages[i])

                    # Save the batch to a separate PDF file
                    batch_filename = f'{cloudFile.file_identifier}-batch{b + 1}.pdf'
                    file_path_batch = f"{nfs_mount}/{self.bucket_name}/{batch_filename}"
                    #fs = gcsfs.GCSFileSystem(project='greenfielddemos')

                    with open(file_path_batch, 'wb') as output_file:
                        writer.write(output_file)
                    #yield CloudFile()
                    #yield LocalFile(file_path=file_path_batch, metadata=cloudFile.metadata, id=cloudFile.id)
                    yield CloudFile(file_identifier=file_path_batch, metadata=cloudFile.metadata, id=batch_filename)
            # else:
            #   #yield LocalFile(file_path=file_path, metadata=cloudFile.metadata, id=cloudFile.id)
            #   yield CloudFile(file_identifier=file_path, metadata=cloudFile.metadata, id=cloudFile.id)

    def config_validation(self) -> bool:
        if not all(x in self.available_metadata for x in self.selector.to_metadata):
            raise ValueError("Invalid metadata values provided")

        try:
            #credentials = service_account.Credentials.from_service_account_file('./greenfielddemos-af461fe8c10e.json')
            storage_client = storage.Client()
            blobs = storage_client.list_blobs(self.bucket_name)
        except Exception as e:
            raise GCSBlobConnectionException(f"Connection to GCS Blob Storage failed, check credentials. See Exception: {e}")
        return True
