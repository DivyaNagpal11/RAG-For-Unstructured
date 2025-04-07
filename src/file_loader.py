import os
from excel_loader import ExcelDataLoader
from html_loader import HtmlDataLoader
from pdf_loader import CpiPdfLoader
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
load_dotenv()


class FileLoader:

    def load_file(self, filename, path, json_path):
        """Loads a file based on its extension with priority order."""
        input_file = os.path.join(path, filename)
        print(filename)

        if filename.endswith(".xlsx"):
            loader = ExcelDataLoader()
            elements = loader.load_unstructured(input_file)
            loader.save_elements_json(elements, filename, json_path)

        elif filename.endswith(".html"):
            loader = HtmlDataLoader()
            elements = loader.load_unstructured(input_file)
            loader.save_elements_json(elements, filename, json_path)

        elif filename.endswith(".pdf"):
            loader = CpiPdfLoader()
            elements = loader.load(input_file, loader_name="Unstructured")
            loader.save_elements_json(elements, filename, json_path)

        return

    def load_folder(self, folder_path, extension, json_path):
        """
        Loads all files in a folder with priority order for extensions.
        """
        unique_files = os.listdir(folder_path)
        filtered_files = [i for i in unique_files if os.path.splitext(i)[-1].lower() in extension]

        filename_dict = {}
        duplicate_files = []

        for filename in filtered_files:
            base_filename, _ = os.path.splitext(filename)
            if base_filename.endswith("._"):
                base_filename = base_filename[:-2]
            if base_filename in filename_dict:
                duplicate_files.append(filename)
                duplicate_files.append(filename_dict[base_filename])
            else:
                filename_dict[base_filename] = filename

        acceptable_files = list(set(filtered_files) - set(duplicate_files))

        duplicate_basefile = []
        for i in duplicate_files:
            base_filename, _ = os.path.splitext(i)
            if base_filename.endswith("._"):
                base_filename = base_filename[:-2]
            duplicate_basefile.append(base_filename)

        for i in set(duplicate_basefile):
            if i + ".xlsx" in filtered_files:
                acceptable_files.append(i + ".xlsx")
            elif i + ".html" in filtered_files:
                acceptable_files.append(i + ".html")
            elif i + "._.pdf" in filtered_files:
                acceptable_files.append(i + "._.pdf")
            elif i + ".pdf" in filtered_files:
                acceptable_files.append(i + ".pdf")

        for filename in tqdm(acceptable_files):
            self.load_file(filename, folder_path, json_path)

        return


if __name__ == '__main__':
    folder_path = os.getenv('CPI_DATA_PATH')
    json_path = os.getenv('CPI_JSON_PATH')
    extension = [".html", ".xlsx", ".pdf"]
    file_loaders = FileLoader()
    file_loaders.load_folder(folder_path, extension, json_path)
