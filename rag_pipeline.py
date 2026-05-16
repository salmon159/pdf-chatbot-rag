import requests
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import *
from prompts import SYSTEM_PROMPT

HF_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)

# Load vector DB
vector_db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)

def query_llm(prompt):

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 128,
            "temperature": 0.1
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    result = response.json()

    print(result)

    # Handle model loading / API errors
    if isinstance(result, dict):

        if "error" in result:

            return f"Model Error: {result['error']}"

        if "estimated_time" in result:

            return "Model is loading. Please try again in a few seconds."

    # Handle successful response
    if isinstance(result, list):

        if len(result) > 0:

            if "generated_text" in result[0]:

                return result[0]["generated_text"]

    return "I couldn't generate a response."

def ask_question(query):

    docs = retriever.get_relevant_documents(query)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
{SYSTEM_PROMPT}

###Context:
{context}

###Question:
{query}

Answer:
"""

    answer = query_llm(prompt)

    return answer
