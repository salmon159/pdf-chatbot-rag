from huggingface_hub import hf_hub_download

from llama_cpp import Llama

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import *
from prompts import SYSTEM_PROMPT

retriever = None
llm = None

def load_components():

    global retriever
    global llm

    # Load vector database
    if retriever is None:

        print("Loading embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME
        )

        print("Loading FAISS index...")

        vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vector_db.as_retriever(
            search_kwargs={"k": 2}
        )

        print("Retriever loaded.")

    # Load GGUF model
    if llm is None:

        print("Downloading GGUF model...")

        model_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME
        )

        print("Loading Mistral GGUF model...")

        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

        print("LLM loaded.")

def ask_question(query):

    load_components()

    docs = retriever.invoke(query)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
[INST]

{SYSTEM_PROMPT}

###Context:
{context}

###Question:
{query}

[/INST]
"""

    try:

        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.1
        )

        answer = output["choices"][0]["text"]

        return answer.strip()

    except Exception as e:

        return f"Error: {str(e)}"
