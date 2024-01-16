from fastapi import FastAPI
from pydantic import BaseModel

import os
import openai
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

openai.api_key  = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
persist_directory = "../db/vectorstore"
embedding = OpenAIEmbeddings()

app = FastAPI()

class Item(BaseModel):
    question: str
    k: int = 3

# vector store
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)

# Prompt
# Building Prompt
template = """You are a legal assistance chatbot for Indian Laws,
use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use 5 sentences maximum. Keep the answer as concise as possible.
Always say "seek expert Legal Advice for more information and assistance." at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

@app.get("/search")
def vector_search(item: Item):
    docs = vectordb.similarity_search(item.question, k=3)
    res = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return res

@app.get("/chat")
def generate_response(item: Item):
    res = qa_chain(item.question)
    return res