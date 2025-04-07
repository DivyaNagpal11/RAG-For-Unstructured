from typing import List
import os
from langchain_community.document_loaders import UnstructuredExcelLoader
from unstructured.partition.auto import partition
from openpyxl import load_workbook
from unstructured.staging.base import elements_to_json, elements_from_json
from unstructured.staging.base import convert_to_dict
import warnings
warnings.filterwarnings("ignore")


class ExcelDataLoader:

    def load_excel(self, file_name: str, mode="single") -> List:
        loader = UnstructuredExcelLoader(file_name, mode=mode)
        data = loader.load()
        return data

    def load(self, file_name: str, mode="single") -> List:
        return self.load_excel(file_name, mode)

    def load_unstructured(self, file_name: str) -> List:
        """
        Load data using unstructured.
        """
        elements = partition(filename=file_name, strategy='hi_res')
        page_numbers = set([el.metadata.page_number for el in elements])

        excel_file = load_workbook(file_name, data_only=True)
        sheets = excel_file.worksheets

        visible_sheets = [i.sheet_state for i in sheets]
        visible_pages = [a for a, b in zip(page_numbers, visible_sheets) if b == "visible"]

        visible_elements = [el for el in elements if el.metadata.page_number in visible_pages]

        return visible_elements

    def save_elements_json(self, elements, filename, json_path):
        convert_to_dict(elements)
        output = json_path + "/" + os.path.splitext(filename)[0] + ".json"
        elements_to_json(elements, filename=output)
        elements = elements_from_json(filename=output)

    def load_elements_json(self, filename):
        elements = elements_from_json(filename=filename)
        return elements


if __name__ == '__main__':

    file_path = os.path.dirname(os.path.abspath(__file__))
    filename = "2_15330-HSC901191_1Uen.D"
    input_file = os.path.join(file_path, filename + ".xlsx")
    loader = ExcelDataLoader()

    elements = loader.load_unstructured(input_file)
    loader.save_elements_json(elements, filename)
    loader.load_elements_json(filename + ".json")
