import warnings
from dotenv import load_dotenv
from os import environ as env
from typing import List
from langchain_community.vectorstores import Chroma
from sentence_transformers_embedding import SentenceTransformerEmbeddings
from langchain_core.documents import Document
import time
from tqdm import tqdm
from ragas.testset.generator import TestsetGenerator
import pandas as pd
from data_chat import HFCustomDataChat

# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()

HF_API_URL = env.get('HF_API_URL', "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2")
HF_API_KEY = env.get('HF_API_KEY', "")
STREAM_MODE = False


class Evaluate():
    
    def generate_synthetic_test_set(self, chat, documents: List[Document], num_tests):
        """
        Generate synthetic test set using RAGAS.
        """
        
        generator = TestsetGenerator.from_langchain(
            chat.llm,
            chat.llm,
            chat.embedding
        )
        
        # # Change resulting question type distribution
        # distributions = {
        #     simple: 0.5,
        #     multi_context: 0.4,
        #     reasoning: 0.1
        # }
        
        testset = generator.generate_with_langchain_docs(documents, num_tests, is_async=False)
        return testset
    
    def convert_to_langchain_documents(self, results):
        documents = []
        for result in results["documents"]:
            documents.append(Document(page_content=result))
        return documents


def main():
    chat = HFCustomDataChat(stream=STREAM_MODE)
    eval = Evaluate()
    
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        persist_directory="docs/chroma_db/",
        embedding_function=embedding
    )
    
    documents = vector_store.get()
    documents = eval.convert_to_langchain_documents(documents)
    testing = pd.DataFrame()
    
    for i in tqdm(range(0, len(documents[:105]))):
        testset = eval.generate_synthetic_test_set(chat, documents[i:i + 2], num_tests=1)
        print("Got here")
        testset = testset.to_pandas()
        testing = pd.concat([testing, testset])
        testing.to_csv("test_set35.csv")
        i = i + 2
        time.sleep(3)


if __name__ == '__main__':
    main()