from datasets import Dataset
from ragas.metrics import context_recall, context_precision
from ragas import evaluate
import pandas as pd
from sentence_transformers_embedding import SentenceTransformerEmbeddings
from data_chat import HFCustomDataChat
from ragas.run_config import RunConfig
from dotenv import load_dotenv
import warnings
from os import environ as env

# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()

HF_API_URL = env.get('HF_API_URL', "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2")
HF_API_KEY = env.get('HF_API_KEY', "")
STREAM_MODE = False

test_set = pd.read_csv("test_set35.csv").drop(columns=["Unnamed: 0"])
test_set['contexts'] = test_set['contexts'].apply(lambda x: eval(x) if isinstance(x, str) else [])
print(test_set.shape)
subset = test_set.loc[0:24, :]  # Filter the dataset to process in batches of size 25

data_dict = {'question': subset['question'].tolist(),
             'ground_truth': subset['ground_truth'].tolist(),
             'contexts': subset['contexts'].tolist(),
             'evolution_type': subset['evolution_type'].tolist(),
             'episode_done': subset['episode_done'].tolist()}

dataset = Dataset.from_dict(data_dict)
chat = HFCustomDataChat(stream=STREAM_MODE)

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print(dataset.features["contexts"].dtype)
score = evaluate(dataset,
                 metrics=[context_recall, context_precision],
                 llm=chat.llm,
                 embeddings=embedding,
                 run_config=RunConfig(max_workers=1, max_retries=1, max_wait=60))
score.to_pandas().to_csv("metrics_rec_prec_0_24.csv")