from langchain_community.vectorstores import Chroma
from sentence_transformers_embedding import SentenceTransformerEmbeddings
import warnings
from dotenv import load_dotenv
from os import environ as env
from data_chat import HFCustomDataChat

# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()

HF_API_URL = env.get('HF_API_URL', "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2")
HF_API_KEY = env.get('HF_API_KEY', "")
STREAM_MODE = True

# Define embedding function
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load existing Chroma vector store
persist_directory = "docs/chroma_db/"

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vector_store.__len__())

# Now you can query the loaded vector store
chat = HFCustomDataChat(stream=STREAM_MODE)
query = "What are the test specifications for Compliant Device Test?"
docs = vector_store.similarity_search(query=query, k=5)
response = chat.query_answer(query, docs)
print("response:", response)