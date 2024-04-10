import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.chains import RetrievalQA
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain import globals

globals.set_verbose(True)

# Setting some variables 
Pinecone.api_key = st.secrets["pinecode_api_secret"]

# SETTING SOME GLOBAL OBJECTS

llm = Ollama(
    model="llama2:7b", 
    temperature=0.0,
    #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
from langchain_community.llms import LlamaCpp
from langchain_experimental.chat_models import Llama2Chat
#añadir tags de caché streamlit

#model = LlamaCpp(
#    model_path = "llama-2-7b-chat.Q4_K_M.gguf",
#    n_ctx = 2048,#max entrada cambiable
#    max_tokens = 256,#max respuesta
#    temperature=0.75,
#    n_gpu_layers = -1,
#    n_batch = 512
#)

#llm = Llama2Chat(llm=model)#wrapper para prompt

index_name = 'ontology-copilot'
pc = Pinecone(api_key=Pinecone.api_key)

embeddings = OllamaEmbeddings(model="llama2:7b", temperature=0.0)
vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=Pinecone.api_key)

# A notable feature of the Retrieval Plugin is its capacity to provide ChatGPT with memory. By utilizing the plugin's upsert endpoint, ChatGPT can save snippets from the conversation to the vector database for later reference (only when prompted to do so by the user). This functionality contributes to a more context-aware chat experience by allowing ChatGPT to remember and retrieve information from previous conversations. Learn how to configure the Retrieval Plugin with memory here.

qa_chatbot = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="refine", 
    retriever=vector_db.as_retriever(), 
    verbose=True
)

st.set_page_config(page_title="Ontology copilot")
st.title("Ontology copilot")
#st.subheader("An ontology design assistant")

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

# initialize message history
initialSystemMessage = (
    """ 
        You are Ontology copilot, a tool designed to assist in the ontology design process. 
        
        Your main role is to specify the key requirements of an ontology, verify compliance with best practices and standards in the field, perform preliminary testing to ensure quality, and work in conjunction with external tools such as OOPS! and FOOPS! to evaluate, validate and provide recommendations on the ontologies generated. 
        
        Your purpose is to act as an assistant to ontology designers, providing guidance and support in the creation process, and never being the one to generate the ontology yourself.
    """
)

initial_messages = [
        SystemMessage(content=initialSystemMessage),
        AIMessage(content="Hello, I am an ontology design assistant. What can I do for you?")
    ]

#ver cómo comprobar si se ha escrito algun mensaje
if "messages" not in st.session_state:
    st.session_state.messages = initial_messages
else:
    st.session_state.messages = initial_messages[:]

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-90')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_text()

def generate_response(prompts):
    # Before the QA Retrieval pluging - 
    prompt = " ".join([msg.content for msg in prompts])
    response = llm.invoke(prompt)
    # However, it has Memory Feature - https://github.com/openai/chatgpt-retrieval-plugin#memory-feature
    #response = qa_chatbot.invoke(prompt)["result"]
    # approach step 1) obtener la info de los documentos; 2) usarla para generar el prompt de entrada al modelo
    ## Prompt construction / retrieval: 
    ## Prompt execution / inference:
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        ## Constructing response
        print("yeyeye")
        print(st.session_state.messages)
        print("yoyoy")
        response = generate_response(st.session_state.messages)
        #print(response)
        #print(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response))
        
    # display message history
    if st.session_state["messages"]:
        messages = st.session_state.get('messages', [])

        for i, msg in enumerate(messages[1:]):
            if type (msg) == AIMessage: ## % 2 == 0:
                message(msg.content, is_user=False, key=str(i) + '_ai')
            else:
                message(msg.content, is_user=True, key=str(i) + '_user')