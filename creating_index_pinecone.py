from google.colab import drive
import streamlit as st
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def create_index_store(index_name):
  pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
      cloud="aws",
      region="us-east-1"
    )
  )

def construct_index(directory_path, index_name):
    directory_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
    documents = directory_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    model_name = "avsolatorio/GIST-small-Embedding-v0"
    model_kwargs={"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
    docsearch.add_documents(documents=chunks)



drive.mount('/content/drive')
#path = '/content/drive/MyDrive/TFM/data/Teóricos'
#path = '/content/drive/MyDrive/TFM/data/Prácticos'
#path = '/content/drive/MyDrive/TFM/data/Teóricos + Prácticos'
path = '/content/drive/MyDrive/TFM/data/Teóricos + Prácticos filtrados'

pinecone_api_key = st.secrets["langchain_api_secret"]
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'methoontochat'

if index_name not in pc.list_indexes().names():
    create_index_store(index_name)

construct_index(path, index_name)
