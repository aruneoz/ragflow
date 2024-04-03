from neumai.Loaders import (
    AutoLoader,
    Loader,
    MarkdownLoader,
    JSONLoader,
    CSVLoader,
    PDFLoader,
    HTMLLoader,
)
from ragflow.utils.loaders.LoaderEnum import LoaderEnum
from ragflow.utils.loaders.UnstructuredLoader import UnstructuredLoader
from ragflow.utils.loaders.PDFLoaderV1 import PDFLoaderV1

class LoaderFactory:
    """Class that leverages the Factory pattern to get the appropriate loader
    """

    def get_loader(loader_name: str, loader_information: dict) -> Loader:
        if loader_information == None:
            return AutoLoader()
        loader_name_enum: LoaderEnum = LoaderEnum.as_loader_enum(loader_name)
        if loader_name_enum == LoaderEnum.autoloader:
            return AutoLoader(**loader_information)
        elif loader_name_enum == LoaderEnum.htmlloader:
            return HTMLLoader(**loader_information)
        elif loader_name_enum == LoaderEnum.markdownloader:
            return MarkdownLoader(**loader_information)
        elif loader_name_enum == LoaderEnum.jsonloader:
            return JSONLoader(**loader_information)
        elif loader_name_enum == LoaderEnum.csvloader:
            return CSVLoader(**loader_information)
        elif loader_name_enum == LoaderEnum.pdfloader:
            return PDFLoader(**loader_information)
        elif loader_name_enum == LoaderEnum.pdfloaderv1:
            return PDFLoaderV1(**loader_information)
        elif loader_name_enum == LoaderEnum.unstructuredloader:
            return UnstructuredLoader(**loader_information)
        else:
            return AutoLoader(**loader_information)
