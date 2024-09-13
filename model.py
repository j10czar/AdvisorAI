# import streamlit as st
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.text_splitter import CharacterTextSplitter 


# #initalize the model and process UF URLs for advising data

# def process_input(prompt):
#     #call the local model (must set this up before running locally)

#     urls = """https://catalog.ufl.edu/UGRD/academic-advising/"""
#     model_local = Ollama(model='mistral')

#     url_list = urls.split("\n")

#     #Uses webbase loader to load the data from each URL
#     docs = [WebBaseLoader(url).load() for url in url_list]
#     data_list = [item for sublist in docs for item in sublist]


#     #chunk documents
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
#     data_split = text_splitter.split_documents(data_list)


#     #convert chunks into enbeddings to put in vector database

#     vectorstore = Chroma.from_documents(
#         documents=data_split,
#         collection_name = 'rag_chroma',
#         embedding=OllamaEmbeddings(model='nomic-embed-text'),
#     )

#     #convert vectorstore to retriever
#     retriever = vectorstore.as_retriever()

#     #give the model instructions
#     after_rag_template = """You can make your reponses in markdown. 
#      you are an academic advisor for the University of florida receiving prompts from a student,
#       you will assist students using the information that is provided to you
#         answer the question based only on the following text found from the UF website,"""
  
#     after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
#     after_rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | after_rag_prompt
#         | model_local
#         | StrOutputParser()
#     )

#     return after_rag_chain.invoke(prompt)






import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter 

def process_input(prompt):
    # Initialize the model
    model_local = Ollama(model='mistral')

    # URL containing UF advising data
    urls = """https://catalog.ufl.edu/UGRD/academic-advising/"""

    url_list = urls.split("\n")

    # Load data from each URL
    docs = [WebBaseLoader(url).load() for url in url_list]
    data_list = [item for sublist in docs for item in sublist]

    # Chunk documents
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    data_split = text_splitter.split_documents(data_list)

    # Convert chunks into embeddings to put in vector database
    vectorstore = Chroma.from_documents(
        documents=data_split,
        collection_name='rag_chroma',
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )

    # Convert vectorstore to retriever
    retriever = vectorstore.as_retriever()

    # Retrieve relevant documents based on the user's question
    context_docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Define the prompt template with placeholders
    after_rag_template = """
You can make your responses in markdown.

You are an academic advisor for the University of Florida receiving prompts from a student. You will assist students using the information that is provided to you.

Answer the question based only on the following text from the UF website:

{context}

Question: {question}

Answer:
"""

    # Format the prompt using Python's string formatting
    prompt_input = after_rag_template.format(context=context, question=prompt)

    # Call the model with the formatted string
    response = model_local(prompt_input)

    # Parse the output
    answer = StrOutputParser().parse(response)

    return answer