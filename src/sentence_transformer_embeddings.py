from pydantic import BaseModel
from langchain.embeddings.base import Embeddings
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(BaseModel, Embeddings):
    model_name: str = "all-MiniLM-L6-v2"
    model: SentenceTransformer = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SentenceTransformer(self.model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query text."""
        embedding = self.model.encode(text)
        return embedding.tolist()


if __name__ == "__main__":
    from dotenv import load_dotenv

    # load the Environment Variables
    load_dotenv()

    # Example usage
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embed_text = emb.embed_query("my name is Divya")
    print(len(embed_text))