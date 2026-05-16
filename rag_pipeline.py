from transformers import pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from config import *

from prompts import SYSTEM_PROMPT

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)

# Load vector database
vector_db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# Load LLM
pipe = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    max_new_tokens=256
)

def ask_question(query):

    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(
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

    result = pipe(prompt)

    answer = result[0]["generated_text"]

    return answer
