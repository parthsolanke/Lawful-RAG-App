from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

class Item(BaseModel):
    question: str
    k: int = 3

# vector store
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)

Bare_act_dict = {
    "./docs/Criminai_Law_Ammendment_Act.pdf": "Criminal Law Amendment Act, 2013",
    "./docs/Indian_Contract_Act.pdf": "Indian Contract Act, 1872",
    "./docs/IPC-Act.pdf": "Indian Penal Code, 1860",
    "./docs/The_Code_Of_Criminal_Procedure.pdf": "The Code Of Criminal Procedure, 1973",
    "./docs/The_Commissions_of_Inquiry_Act.pdf": "The Commissions of Inquiry Act, 1952",
    "./docs/The_constitution_of_India.pdf": "The constitution of India"
}

# Building Prompt for QA Chain
template = """You are a legal assistance chatbot for Indian Laws,
            use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            also make sure to include concerned section ids and acts in the answer.
            Keep the answer as concise as possible.
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

# Building Prompt for Summary Chain
template_vec = """You are a legal jargon simplfier and summarizer for Indian Laws for the victims of 
            explicit content leaks leading to blackmailing, and other such cases.
            From the following content, list down all the detected section id of the act
            mentioned in the content with simple single sentence explanation according to the legality of 
            query with respect to for each section id and act (also add a newline char at the end of thesection id sentence).
            Keep the answer as concise, simple and uncensored as possible.
            take refrence of query as mentioned follows.
            content: {context}
            query: {question}
            Helpful response:"""
SUMMARY_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "query"],
                                        template=template_vec)

# Chain
vector_search_summary_chain = RetrievalQA.from_chain_type(llm,
                                retriever=vectordb.as_retriever(),
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": SUMMARY_CHAIN_PROMPT})

@app.post("/search")
def vector_search(item: Item):
    docs = vectordb.similarity_search(item.question, k=3)
    res = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return res

@app.post("/search/summary")
def vector_search_summary(item: Item):
    res = vector_search_summary_chain(item.question)
    return res

@app.post("/chat")
def generate_response(item: Item):
    res = qa_chain(item.question)
    return res

@app.get("/")
def say_hi():
    return {"message": "Hello World"}