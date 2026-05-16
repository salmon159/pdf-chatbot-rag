import requests
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import *

HF_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

retriever = None

def load_retriever():

    global retriever

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

    return retriever

def query_llm(prompt):

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 64
        }
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=20
        )

        print("STATUS:", response.status_code)

        if response.status_code != 200:

            return f"HF API Error: {response.status_code}"

        result = response.json()

        print(result)

        if isinstance(result, list):

            return result[0].get(
                "generated_text",
                "No response generated."
            )

        if isinstance(result, dict):

            if "error" in result:

                return result["error"]

        return str(result)

    except Exception as e:

        return f"ERROR: {str(e)}"

def ask_question(query):

    retriever_instance = load_retriever()

    docs = retriever_instance.invoke(query)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

    return query_llm(prompt)
