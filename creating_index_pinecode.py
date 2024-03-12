# Relevant resources: 
# Pinecode (Pinecode API for creating vector store, Langchain API for retreiving): 
#   https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/05-langchain-retrieval-augmentation-fast.ipynb
#   https://docs.pinecone.io/docs/langchain
# Alphasec: 
#   https://alphasec.io/langchain-decoded-the-muggles-guide-to-langchain/
#   https://alphasec.io/generative-question-answering-with-langchain-and-pinecone/
#   https://alphasec.io/langchain-decoded-part-4-indexes/, 
# Streamlit, Pinecode, Langchain: https://pub.towardsai.net/deploying-a-langchain-large-language-model-llm-with-streamlit-pinecone-190cd2577ae2

import sys
import os

from pinecone import Pinecone, PodSpec
from langchain.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = "3a8a5a1b-bb75-4ace-8f34-75791988c1a0"

index_name = 'langchain-business-advisor-streamlit'

pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index_store(index_name):
        # we create a new index
        pc.create_index(
            name=index_name,
            metric='cosine',
            dimension=4096,  # 1536 dim of text-embedding-ada-002
            spec=PodSpec(
                environment="gcp-starter"
            )
        )

def construct_index(directory_path, index_name):
    #indexo = pc.Index(index_name)

    directory_loader = DirectoryLoader(directory_path, glob="**/*.txt")
    documents = directory_loader.load()
    # Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    # Embeddings
    embeddings = OllamaEmbeddings(model="llama2", temperature=0.0)#base_url='http://localhost:11434'
    print(embeddings)
    print(chunks)

    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    docsearch.add_documents(documents=chunks)
    #docsearch.add_documents(documents=[chunks[0]])

    #solución parecida a la anterior que tampoco funciona
    #docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    #docsearch.from_documents(documents=chunks, embedding=embeddings)
    
    #original que debería funcionar
    #from langchain_community.vectorstores import Pinecone as PineconeVector
    #docsearch = PineconeVector.from_documents(chunks, embeddings, index_name=index_name)
    
    #con Chroma en vez de Pinecone
    #from langchain_community.vectorstores import Chroma
    #docsearch = Chroma.from_documents(documents=chunks, embedding=embeddings)


    """
    chunks_and_embeddings = []
    for chunk in chunks:
        embedding = embeddings.encode(chunk)
        # Aquí puedes definir los metadatos que desees
        metadata = {"text": chunk}
        chunks_and_embeddings.append((chunk, embedding, metadata))
    upsert_response = pc.upsert(
        vectors=chunks_and_embeddings,
        namespace=index_name
    )
    """

    """
    from uuid import uuid4

    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'wiki-id': str(record['id']),
            'source': record['url'],
            'title': record['title']
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['text'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
    """

    """
    from tqdm.auto import tqdm

    batch_size = 100  # how many embeddings we create and insert at once

    for index in indexes:
        print(f"Indexing for {index['name']}")
        for i in tqdm(range(0, len(data), batch_size)):
            passed = False
            # find end of batch
            i_end = min(len(data), i+batch_size)
            # create batch
            batch = data[i:i_end]
            embeds = embed(batch["text"], name=index["name"])
            to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
            # upsert to Pinecone
            index["index"].upsert(vectors=to_upsert)
    """
   
#pc.delete_index(index_name)
#create_index_store(index_name)
    
if index_name not in pc.list_indexes().names():
    create_index_store(index_name)

construct_index('./data', index_name)

#https://api.python.langchain.com/en/latest/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html#langchain_pinecone.vectorstores.PineconeVectorStore.add_documents

#https://docs.pinecone.io/v1/docs/langchain