import gradio as gr
import download_report as dr
from data_chat import CustomDataChat
import warnings
from dotenv import load_dotenv
from os import environ as env
from langchain_community.vectorstores import Chroma
from sentence_transformers_embedding import SentenceTransformerEmbeddings

# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()

OLLAMA_MODEL = env.get('OLLAMA_MODEL', 'mistral')
STREAM_MODE = True


def generate_report(topic, model_choice, model_name, system_prompt):
    """
    This function is triggered when the user hits submit button.
    """
    # Generate the report based on the topic
    chat = CustomDataChat(stream=STREAM_MODE)
    
    # Define embedding function
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load existing Chroma vector store
    vector_store = Chroma(
        persist_directory="docs/chroma_db/",
        embedding_function=embedding
    )
    
    print(vector_store.__len__())
    
    docs = vector_store.similarity_search(query=topic, k=5)
    response = chat.query_answer(topic, docs)
    print("response:", response)
    return response


def download_file(text, format):
    """
    This function is triggered when the user hits download button.
    """
    dr.DownloadReport().save_text_as_file(text, format)
    return


if __name__ == "__main__":
    
    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """
    
    with gr.Blocks(title="Multimodal RAG", css=css) as demo:
        
        gr.Markdown("""# Wikipedia Researcher Tool""")
        
        with gr.Row():
            with gr.Column():
                fn_submit = generate_report
                topic = gr.Textbox(label="Topic", placeholder="Enter your topic here")
                model_choice = gr.Radio(choices=["Ollama"], label="Model", value="Ollama")
                model_name = gr.Textbox(label="Model Name", placeholder="mistral", value="mistral")
                system_prompt = gr.Textbox(label="System Prompts (Optional)")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                fn_download = download_file
                output = gr.Textbox(label="Output", lines=9.3)
                download_format = gr.Dropdown(
                    choices=["pdf", "docx", "html", "txt"],
                    label="Download Format")
                download_btn = gr.Button("Download")
        
        submit_btn.click(fn=fn_submit,
                         inputs=[topic, model_choice, model_name, system_prompt],
                         outputs=output, api_name=False)
        download_btn.click(fn=fn_download, inputs=[output, download_format], api_name=False)
    
    demo.launch()
