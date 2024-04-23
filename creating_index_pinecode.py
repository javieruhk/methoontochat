from pinecone import Pinecone, PodSpec
from langchain.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.embeddings import HuggingFaceEmbeddings


from google.colab import drive
drive.mount('/content/drive')
PATH = '/content/drive/MyDrive/TFM/data'

PINECONE_API_KEY = "3a8a5a1b-bb75-4ace-8f34-75791988c1a0"

index_name = 'ontology-copilot'

pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index_store(index_name):
        pc.create_index(
            name=index_name,
            metric='cosine',
            dimension=384,  # 1536 dim of text-embedding-ada-002
            spec=PodSpec(
                environment="gcp-starter"
            )
        )

from langchain_community.document_loaders import PDFMinerLoader

def construct_index(directory_path, index_name):
    #indexo = pc.Index(index_name)

    directory_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PDFMinerLoader)
    documents = directory_loader.load()

    """
    for file in os.listdir(directory_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
    """

    # Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    # Embeddings

    #model_name = "sentence-transformers/all-mpnet-base-v2" #768
    #model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "avsolatorio/GIST-small-Embedding-v0"
    model_kwargs={"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    print(embeddings)
    print(chunks)

    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    docsearch.add_documents(documents=chunks)
    #docsearch.add_documents(documents=[chunks[0]])

if index_name not in pc.list_indexes().names():
    create_index_store(index_name)

construct_index(PATH, index_name)
