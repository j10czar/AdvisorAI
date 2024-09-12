import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter 


#initalize the model and process UF URLs for advising data

def process_input(prompt):
    #call the local model (must set this up before running locally)

    urls = """https://catalog.ufl.edu/UGRD/academic-advising/"""
    model_local = Ollama(model='mistral')

    url_list = urls.split("\n")

    #Uses webbase loader to load the data from each URL
    docs = [WebBaseLoader(url).load() for url in url_list]
    data_list = [item for sublist in docs for item in sublist]


    #chunk documents
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    data_split = text_splitter.split_documents(data_list)


    #convert chunks into enbeddings to put in vector database

    vectorstore = Chroma.from_documents(
        documents=data_split,
        collection_name = 'rag_chroma',
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )

    #convert vectorstore to retriever
    retriever = vectorstore.as_retriever()

    #give the model instructions
    after_rag_template = """You can make your reponses in markdown. answer the question based only on the following text,
     you are an academic advisor for the University of florida,
      you will assist students using the information that is provided to you
       Only If the student asks for information relative to building a schedule, you will take the data from the websites provided and build a schedule for the student using the courses the student provided to you when you ask.
        If the student asks for info that is not relavant to academic advising you will ask that the conversation is kept to advising."""


    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(prompt)






