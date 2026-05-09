
from transformers import pipeline

from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from config import *

from prompts import SYSTEM_PROMPT

template = """
{system_prompt}

###Context:
{context}

###Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    partial_variables={
        "system_prompt": SYSTEM_PROMPT
    }
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": PROMPT
    }
)

def ask_question(query):

    response = qa_chain.run(query)

    return response
