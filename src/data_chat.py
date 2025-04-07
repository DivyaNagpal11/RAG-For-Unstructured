import os
import warnings
from dotenv import load_dotenv
from os import environ as env
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers_embedding import SentenceTransformerEmbeddings
from ollama_chat_llm import OllamaChatLLM
from ollama_chat_llm_stream import OllamaChatLLMStream
from langchain_core.documents import Document
from langchain.vectorstores import utils as chromautils
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from tqdm import tqdm
from unstructured.staging.base import elements_from_json


# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()

OLLAMA_MODEL = env.get('OLLAMA_MODEL', 'mistral')
STREAM_MODE = False


def stream_call_back(response):
    """
    Callback function to display stream.
    """
    full_response = ""
    for chunk in response:
        full_response += chunk
    return full_response


class CustomDataChat():
    """
    Use Ollama models to chat with custom data.
    """
    def __init__(self, stream: bool = True) -> None:
        # Initialize the embedding model - using sentence-transformers
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.stream = stream
        if stream:
            self.llm = OllamaChatLLMStream(access={
                "model": OLLAMA_MODEL,
                "callback": stream_call_back
            })
        else:
            self.llm = OllamaChatLLM(access={
                "model": OLLAMA_MODEL
            })
        
        self.persist_folder = "docs/chroma_db/"
        self.store = Chroma(persist_directory=self.persist_folder,
                          embedding_function=self.embedding)
        # Uncomment this line to save the vector store
        # self.store.persist()
        print(self.store.__len__())

    def load_elements_json(self, filename):
        elements = elements_from_json(filename=filename)
        return elements

    def split(self, text: str) -> List[Document]:
        """
        Split text using recursive character text splitter.
        """
        # Initialize the recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split the text into chunks
        chunks = text_splitter.create_documents([text])
        return chunks

    def vector_store(self,
                     documents: List[Document]) -> Chroma:
        """
        Create vector store.
        """
        docs = chromautils.filter_complex_metadata(documents)
        print("Creating Vector Store")

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.persist_folder
        )
        vectordb.persist()
        return docs

    def update_vector_store(self,
                            documents: List[Document]) -> Chroma:
        """
        Update vector store.
        """
        docs = chromautils.filter_complex_metadata(documents)

        print("Adding to vector store")
        self.store.add_documents(documents=docs)
        print("Docs count", self.store.__len__())
        return docs

    def ingest(self, file_name: str) -> None:
        """
        Load elements, split, then save to vector store.
        """
        # Initialize documents list
        all_documents = []
        
        # Process all JSON files in the directory
        for file in tqdm(sorted(glob.glob(f"{file_name}/*.json"))):
            print(file)
            # Load elements from JSON
            elements = self.load_elements_json(file)
            
            # Process each element
            for element in elements:
                text = element.text
                metadata = element.metadata.to_dict()
                
                # Split the text using recursive text splitter
                chunks = self.split(text)
                
                # Add metadata to each chunk
                for chunk in chunks:
                    chunk.metadata.update(metadata)
                    all_documents.append(chunk)

        # Create or update vector store in batches
        print("Total documents:", len(all_documents))
        for i in range(0, len(all_documents), 250):
            batch = all_documents[i:i + 250]
            print("Batch size:", len(batch))
            if i == 0:
                docs = self.vector_store(batch)
            else:
                time.sleep(2)
                docs = self.update_vector_store(batch)
        return docs

    def query_answer(self,
                     query: str,
                     docs: List) -> str:
        """
        Query LLM to get answer.
        """
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system",
                  """
                  Use the following pieces of context to answer the user's question. \
                If you don't know the answer, just say that you don't know, \
                don't try to make up an answer:
                """),
                 ("user",
                    """
                Context:

                {context}

                Question:

                {question}
                """)])
            chain = create_stuff_documents_chain(self.llm, prompt)
            print(f"prompt: {prompt}")
            response = chain.invoke({"context": docs, "question": query})
            return (response)
        except Exception as ex:
            print(ex)
            return ("Sorry, I don't know.")


def main():
    import glob
    chat = CustomDataChat(stream=STREAM_MODE)
    file_path = os.getenv('JSON_PATH')

    # Comment this line if you are loading a vector store
    chat.ingest(file_path)

    # Load existing Chroma vector store
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory="docs/chroma_db/",
        embedding_function=embedding
    )

    print(vector_store.__len__())


if __name__ == '__main__':
    main()