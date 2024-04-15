from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from pinecone import Pinecone
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import os
from uuid import uuid4
uid = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ontology-copilot-"+str(uid)
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_secret"]

from langsmith import Client
client = Client()

def load_model():
	repo_id="tiiuae/falcon-7b-instruct"
	huggingfacehub_api_token = st.secrets["hugging_face_api_secret"]
	
	llm = HuggingFaceEndpoint(
		repo_id=repo_id, 
		huggingfacehub_api_token=huggingfacehub_api_token, 
		temperature= 0.5, 
		max_new_tokens=250
		)
	return llm

def get_vector_db():
	Pinecone.api_key = st.secrets["pinecode_api_secret"]
	index_name = 'ontology-copilot'
	pc = Pinecone(api_key=Pinecone.api_key)

	embeddings = OllamaEmbeddings(model="llama2:7b", temperature=0.0)
	vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=Pinecone.api_key)

	return vector_db

#qa_chatbot = RetrievalQA.from_chain_type(
#    llm=llm, 
#    chain_type="refine", 
#    retriever=vector_db.as_retriever(), 
#    verbose=True
#)

def create_chain():
	vector_db = get_vector_db()
	memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)
	
	chatbot_initial_prompt = """ 
        You are Ontology copilot, a tool designed to assist in the ontology design process. 
        
        Your main role is to specify the key requirements of an ontology, verify compliance with best practices and standards in the field, perform preliminary testing to ensure quality, and work in conjunction with external tools such as OOPS! and FOOPS! to evaluate, validate and provide recommendations on the ontologies generated. 
        
        Your purpose is to act as an assistant to ontology designers, providing guidance and support in the creation process, and never being the one to generate the ontology yourself.
		
		{context}

		{question}
    """
	
	combine_docs_prompt = PromptTemplate(template=chatbot_initial_prompt, input_variables=["context", "question"])

	chain = ConversationalRetrievalChain.from_llm(
		llm=st.session_state.llm,
		retriever=vector_db.as_retriever(),
		memory = memory,
		combine_docs_chain_kwargs={"prompt": combine_docs_prompt}
		)
	
	return chain

def main():
	st.set_page_config(page_title="Ontology copilot")
	st.title("Ontology copilot")

	if "llm" not in st.session_state:
		st.session_state.llm = None
	if "chain" not in st.session_state:
		st.session_state.chain = None		
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []


	st.session_state.llm = load_model()
	st.session_state.chain = create_chain()
	
	chat_input = st.chat_input("Hello, I am an ontology design assistant. What can I do for you?")
	colored_header(label='', description='', color_name='blue-90')
	
	with st.sidebar:
		st.sidebar.image("resources/OEG.png", use_column_width=True)
		st.markdown('''
		## What is Ontology copilot?

		Ontology copilot is a tool to help in the ontology design process. Some of the actions it can perform are:
	
		- Specification of ontology requirements.
	
		- Best practices and standards to be met by the ontology.
	
		- Preliminary testing of the generated ontology in order to check if the mentioned good practices and standards are met.
	
		- Use external tools such as [OOPS!](https://oops.linkeddata.es/) and [FOOPS!](https://foops.linkeddata.es/FAIR_validator.html) in order to evaluate and validate ontologies, detecting possible errors and bad practices, as well as receiving recommendations.

		As a clarification, this tool does not generate ontologies by itself, but is intended as an assistant in the ontology generation process.
		''')
		add_vertical_space(5)
		st.sidebar.image("resources/UPM.png", use_column_width=True)
		st.write('''
		## Author:
		- Javier Gómez de Agüero Muñoz
		## Tutors:
		- Elena Montiel Ponsoda 
		- Carlos Ruiz Moreno
		''')
		
	#ocultar menú de 3 puntos
	#hide_streamlit_style = """
	#            <style>
	#            #MainMenu {visibility: hidden;}
	#            footer {visibility: hidden;}
	#            footer:after {content:'Made by Jeff team (crm, 2023)';visibility: visible;}
	#            </style>
	#            """
	#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
		
	num_message = 0
	for num_message, message_text in enumerate(st.session_state.chat_history):
		if isinstance(message, AIMessage):
			message(message_text.content, is_user=False, key=str(num_message) + '_ai')
		elif isinstance(message, HumanMessage):	
			message(message_text.content, is_user=True, key=str(num_message) + '_user')

	
	if chat_input is not None and chat_input != "":
		num_message+=1
		message(chat_input, is_user=True, key=str(num_message) + '_user')
		
		with st.spinner("Wait until I get the response..."):
			#CHAIN
			response = st.session_state.chain.invoke(chat_input)

			st.session_state.chat_history = response["chat_history"]
			st.session_state.chat_history.append(HumanMessage(content=response["question"]))
			st.session_state.chat_history.append(AIMessage(content=response["answer"]))
			
			num_message+=1
			message(response["answer"], is_user=False, key=str(num_message) + '_ai')
			

			#LLM
			#response = st.session_state.llm.invoke(chat_input)
			#num_message+=1
			#message(response, is_user=False, key=str(num_message) + '_ai')

if __name__ == "__main__":
	main()