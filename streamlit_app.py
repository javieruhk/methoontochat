import os
from uuid import uuid4
import streamlit as st
from langsmith import Client
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_chat import message


uid = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "methoontochat-theoretical+practical-filtered"+str(uid)
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_secret"]

client = Client()

@st.cache_resource(show_spinner="Loading LLM...")
def load_model():
	repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
	huggingfacehub_api_token = st.secrets["hugging_face_api_secret"]
	
	llm = HuggingFaceEndpoint(
		repo_id=repo_id, 
		huggingfacehub_api_token=huggingfacehub_api_token, 
		temperature= 0.5, 
		max_new_tokens=300
		)
	return llm

def get_vector_db():
	index_name = 'methoontochat'

	model_name = "avsolatorio/GIST-small-Embedding-v0"
	embeddings = HuggingFaceEmbeddings(model_name=model_name)

	pinecone_key = st.secrets["pinecode_api_secret"]

	vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_key)

	return vector_db

def create_chain():
	vector_db = get_vector_db()
	memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True, output_key="answer")
	
	chatbot_initial_prompt = """
		You are talking with a human and it is important to respond to his questions in a helpful way, while being safe and concise (less than 150 words). If a question lacks coherence, clarify why rather than providing an inaccurate response. If you're unsure of an answer, it is best not to provide false information.
	
		You are MethoOntoChat, a tool designed to assist in the ontology design process. 
	
		Your main role is to assist in the creation of ontologies, providing information and resolving conceptual doubts about the methodology to be followed. With this you should facilitate the understanding of certain concepts of ontologies and thus accelerate the completion of tasks such as the verification of best practices and standards in the field, and the specification of requirements by providing the necessary context and information about the procedures followed by the ontology generation methodology chosen. 
	
		Your purpose is to act as an assistant to ontology designers, providing guidance and support in the creation process, and never being the one to generate the ontology yourself.
		
		Remember to answer the human, as well as you can, only once per question without creating a conversation on your own. Answer concisely and always summarize it in less than 150 words.
		
		{context}
	
		{question}
	"""

	combine_docs_prompt = PromptTemplate(template=chatbot_initial_prompt, input_variables=["context", "question"])

	chain = ConversationalRetrievalChain.from_llm(
		llm=st.session_state.llm,
		retriever=vector_db.as_retriever(), #poner dentro del paréntesis search_kwargs={"k":5} para indicar el número de partes de documentos a extraer
		memory = memory,
		combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
		return_source_documents=True, #para que devuelva los documentos de los que extrae la información 
		)
	
	return chain

def create_user_interface():
	st.set_page_config(page_title="MethoOntoChat")
	st.title("MethoOntoChat")
	colored_header(label='', description='', color_name='blue-90')
		
	with st.sidebar:
		st.sidebar.image("resources/OEG.png", use_column_width=True)
		st.markdown('''
		## What is MethoOntoChat?

		MethoOntoChat is a tool to help in the ontology design process. Some of the actions it can perform are:
	
		- Assisting in the specification of requirements by providing the necessary context and information about the procedures followed by the chosen ontology generation methodology.

		- Provide recommendations to verify compliance with best practices and standards.
	
		As a clarification, this tool does not generate ontologies by itself, but is intended as an assistant in the ontology generation process.
		''')
		add_vertical_space(5)
		st.sidebar.image("resources/UPM.png", use_column_width=True)
		st.write('''
		## Author:
		- Javier Gómez de Agüero Muñoz
		## Tutors:
		- Elena Montiel Ponsoda 
		- María Poveda Villalón 
		''')

def initialize_state():
	if "llm" not in st.session_state:
		st.session_state.llm = None
	if "chain" not in st.session_state:
		st.session_state.chain = None		
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []
		
	st.session_state.llm = load_model()
	if not st.session_state.chain:
		with st.spinner("Loading chain..."):
			st.session_state.chain = create_chain()

def process_messages():
	chat_input = st.chat_input("Hello, I am an ontology developing assistant. What can I do for you?")

	num_message = 0
	for num_message, message_text in enumerate(st.session_state.chat_history):
		if isinstance(message_text, AIMessage):
			message(message_text.content, is_user=False, key=str(num_message) + '_ai')
		elif isinstance(message_text, HumanMessage):	
			message(message_text.content, is_user=True, key=str(num_message) + '_user')

	if chat_input is not None and chat_input != "":
		num_message+=1
		message(chat_input, is_user=True, key=str(num_message) + '_user')
		
		with st.spinner("Wait until I get the response..."):
			response = st.session_state.chain.invoke(chat_input)

			answer = response["answer"].strip()
			answer_processed = ' '.join(answer.split())

			st.session_state.chat_history.append(HumanMessage(content=response["question"]))
			st.session_state.chat_history.append(AIMessage(content=answer_processed))
			
			num_message+=1

			message(answer_processed, is_user=False, key=str(num_message) + '_ai')

def main():
	create_user_interface()

	initialize_state()
	
	process_messages()
	

if __name__ == "__main__":
	main()
