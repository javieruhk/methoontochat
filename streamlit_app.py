import openai 

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

# Setting some variables 
Pinecone.api_key = st.secrets["pinecode_api_secret"]

##
# SETTING SOME GLOBAL OBJECTS

llm = Ollama(
    model="llama2", 
    temperature=0.0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

index_name = 'langchain-business-advisor-streamlit'
pc = Pinecone(api_key=Pinecone.api_key)

embeddings = OllamaEmbeddings(model="llama2", temperature=0.0)
vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=Pinecone.api_key)

# A notable feature of the Retrieval Plugin is its capacity to provide ChatGPT with memory. By utilizing the plugin's upsert endpoint, ChatGPT can save snippets from the conversation to the vector database for later reference (only when prompted to do so by the user). This functionality contributes to a more context-aware chat experience by allowing ChatGPT to remember and retrieve information from previous conversations. Learn how to configure the Retrieval Plugin with memory here.

qa_chatbot = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="refine", 
    retriever=vector_db.as_retriever(), 
    verbose=True
)
##

st.set_page_config(page_title="Jeff Advisor for Mr Jeff - An LLM-powered app by Jeff")
st.title("Jeff Advisor for Mr Jeff")
st.subheader("An LLM-powered app by Jeff")

with st.sidebar:
    st.sidebar.image("resources/logo_mrjeff.png", use_column_width=True)
    st.markdown('''
    ## About

    Jeff is a technology-driven company that offers a wide range of services through its various brands, including Mr Jeff for laundry and dry-cleaning. 
    
    Jeff operates through a franchise model, empowering entrepreneurs to run their own stores while benefiting from the support, expertise, and technology of the company.

    This Jeff Advisor for Mr Jeff, an smart guidance that gives entrepreneurs actionable insights to help them run their businesses. It is is a modified AI-powered chatbot built using:
    - All the know-how of our training at **Jeff Academy**
    - All the guidelines provided by the **Jeff Operation Handbook**
    - All the metrics about performance from our partners in the **Jeff Platform**
    
    ''')
    add_vertical_space(5)
    st.sidebar.image("resources/logo_jeff.png", use_column_width=False)
    st.write('This is a Jeff product (c) 2023')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {content:'Made by Jeff team (crm, 2023)';visibility: visible;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# initialize message history
initialSystemMessage = (
    """ 
        You are Jeff Advisor, the helpful business assistant of the company Mr Jeff that guides and gives entrepreneurs actionable insights to help them run their businesses, simplifying the decision making process, and connecting with our technology, data, know-how and products available. "
    
        You create and adapt different business playbooks for the company's franchises to structure their business strategy with constant feedback on how each action impacts progress towards the goals
    """
)

initial_messages = [
        SystemMessage(content=initialSystemMessage),
        AIMessage(content="Hello, I am Jeff Advisor for Mr Jeff, How may I help you?")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = initial_messages
else:
    st.session_state.messages = initial_messages[:]

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
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
    print("request")
    print(prompts)
    prompt = " ".join([msg.content for msg in prompts])
    #response = llm(prompt)
    # However, it has Memory Feature - https://github.com/openai/chatgpt-retrieval-plugin#memory-feature
    response = qa_chatbot.invoke(prompt)
    print(response)
    # approach step 1) obtener la info de los documentos; 2) usarla para generar el prompt de entrada al modelo
    ## Prompt construction / retrieval: 
    ## Prompt execution / inference:
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        ## Constructing response
        response = generate_response(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response))
        
    # display message history
    if st.session_state["messages"]:
        messages = st.session_state.get('messages', [])
        for i, msg in enumerate(messages[1:]):
            if type (msg) == AIMessage: ## % 2 == 0:
                message(msg.content, is_user=False, key=str(i) + '_ai')
            else:
                message(msg.content, is_user=True, key=str(i) + '_user')
