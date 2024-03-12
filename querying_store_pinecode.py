from pinecone import Pinecone
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA


PINECONE_API_KEY = "3a8a5a1b-bb75-4ace-8f34-75791988c1a0"

index_name = 'langchain-business-advisor-streamlit'

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = OllamaEmbeddings(model="llama2", temperature=0.0)#base_url='http://localhost:11434'
vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

llm = Ollama(
    model="llama2", 
    temperature=0.0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

retriever = vector_db.as_retriever()

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="refine", #  With "stuff" was not able to find anything. 
    retriever=retriever, 
    verbose=True
)

## Interesting note: it seems that the procedure in [Deploying a Langchain Large Language Model (LLM) with Streamlit & Pinecone] works better.
# docsearch = Pinecone.from_existing_index(os.environ['PINECONE_INDEX_NAME'], embeddings)
# query = "write me langchain code to build my hugging face model"
# docs = docsearch.similarity_search(query)
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.llms import OpenAI
# model = OpenAI(model_name="text-davinci-003")
# sources_chain = load_qa_with_sources_chain(model, chain_type="refine")
# result = sources_chain.run(input_documents=docs, question=query)
# print(result)

query =  "What is Jeff?"
#response = qa_stuff.run(query)
response = qa_stuff.invoke(query)
print(response)