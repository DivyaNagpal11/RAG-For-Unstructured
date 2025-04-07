from typing import List
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_json, elements_from_json
from unstructured.staging.base import convert_to_dict
import warnings
warnings.filterwarnings("ignore")


class HtmlDataLoader:

    def load_html(self, file_name: str, mode="single") -> List:
        loader = UnstructuredHTMLLoader(file_name, mode=mode)
        data = loader.load()
        return data

    def load(self, file_name: str, mode="single") -> List:
        return self.load_html(file_name, mode)

    def load_unstructured(self, file_name: str) -> List:
        """
        Load data using unstructured.
        """
        elements = partition(filename=file_name, strategy='hi_res',)
        return elements

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
    # filename = "2_1543-HSC901191_1Uen.F"
    filename = "tester_html"
    input_file = os.path.join(file_path, filename + ".html")
    loader = HtmlDataLoader()
    elements = loader.load_unstructured(input_file)
    # print(elements)
    loader.save_elements(elements, filename)
    # loader.load_elements_json(filename + ".json")
