from typing import List
import os
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.staging.base import elements_to_json, elements_from_json
from unstructured.staging.base import convert_to_dict
from unstructured.partition.pdf import partition_pdf
import warnings
warnings.filterwarnings("ignore")


class PdfLoader:

    def __init__(self) -> None:
        self.loader_dict = {
            "PyPDFLoader": self.load_PyPDF,
            "UnstructuredPDFFromLangChain": self.load_UnstructuredPDFFromLangChain,
            "Unstructured": self.load_unstructured,
        }

    def load_PyPDF(self, file_name: str) -> List:
        loader = PyPDFLoader(file_name, extract_images=True)
        pages = loader.load()
        for page in pages:
            print("Page:", page.metadata["page"])
            print(page.page_content)
        return pages

    def load_UnstructuredPDFFromLangChain(self, file_name: str) -> List:
        """
        Load data using UnstructuredPDFLoader in LangChain.
        """
        loader = UnstructuredPDFLoader(file_name, mode="elements")
        elements = loader.load()
        for element in elements:
            print(element)
        return elements

    def load_unstructured(self, file_name: str) -> List:
        """
        Load data using unstructured.
        brew install poppler
        brew install tesseract
        """
        elements = partition_pdf(filename=file_name,
                                 infer_table_structure=True,
                                 strategy='ocr_only',
                                 )

        # tables = [el for el in elements if el.category == "Table"]

        return elements

    def load(self, file_name: str, loader_name="Unstructured") -> List:
        """
        Load data using PyPDF.
        """
        if loader_name in self.loader_dict:
            self.loader_name = loader_name
            result = self.loader_dict[self.loader_name](file_name)
            return result
        else:
            return []

    def save_elements_json(self, elements, filename, json_path):
        convert_to_dict(elements)
        output = json_path + "/" + os.path.splitext(filename)[0] + ".json"
        elements_to_json(elements, filename=output)
        elements = elements_from_json(filename=output)

    def save_elements(self, elements, filename):
        convert_to_dict(elements)
        output = filename + ".json"
        elements_to_json(elements, filename=output)
        elements = elements_from_json(filename=output)

    def load_elements_json(self, filename):
        elements = elements_from_json(filename=filename)
        return elements


if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__))
    filename = "tester_pdf"
    input_file = os.path.join(file_path, filename + ".pdf")
    loader = PdfLoader()
    elements = loader.load(input_file, loader_name="Unstructured")
    loader.save_elements(elements, filename)
    # loader.load_elements_json(filename + ".json")
