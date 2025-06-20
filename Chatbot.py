import streamlit as st 
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
import time
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import os

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#key de Google
if "GOOGLE_API_KEY" not in os.environ:
     os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


#Configurar modelo de Gemini
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash-exp",
    temperature = 0
)

# Ruta de database
persist_directory = './FCB_recursive_db/'

# #embeddins
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# vectorstore
vectordb = Chroma(embedding_function = embed_model,
                     persist_directory = persist_directory)

# #Recuperacion de los embeddings
retriever_mmr = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k":20, "fetch_k":60, "lambda_mult": 0.75})
                # search_kwargs = {'k':20})

retriever_similarity = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k":10}) 

ensemble_retriever = EnsembleRetriever(retrievers = [ retriever_mmr , retriever_similarity],
                                       weights = [0.55, 0.45])
# https://github.com/langchain-ai/langchain/issues/31192
_filter = LLMListwiseRerank.from_llm(llm, top_n = 3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor = _filter, base_retriever = ensemble_retriever
)
# # Prompt personalizado

custom_prompt_template = """Eres un asistente especializado de responder preguntas sobre los reglamentos de la FACULTAD DE CIENCIAS BIOLÓGICAS de la Universidad Juárez del Estado de Durango (UJED).
Da una respuesta detallada sobre las preguntas relacionadas con los reglamentos.
Si no sabes la respuesta, di que no puedes responder.

Contexto: {context}
Pregunta: {question}

Solo devuelve la respuesta util a continuacion y nada mas.
Responde siempre en español y de forma sarcastica:
Respuesta útil:
"""

prompt = PromptTemplate(template = custom_prompt_template, 
                        input_variables = ['context', 'question'])

qa_chain = RetrievalQA.from_chain_type(llm = llm, 
                                  chain_type = "stuff", 
                                  retriever = compression_retriever, 
                                  return_source_documents = True,
                                  chain_type_kwargs = {"prompt": prompt})
     
## Cite sources
def make_output(query):
    answer = qa_chain.invoke(query)
    result = answer["result"]
    
    # Obtener fuentes únicas
    unique_sources = set()
    for source in answer["source_documents"]:
        unique_sources.add(source.metadata.get('source', 'Fuente desconocida'))
    
    # Si no hay fuentes, devolver solo el resultado
    if not unique_sources:
        return result
    
    # Formatear las fuentes
    fuentes_formateadas = "\n".join([f"- {fuente}" for fuente in unique_sources])
    
    return f"{result}\n\n**Fuentes:**\n{fuentes_formateadas}"

# Function to modify the output by adding spaces between each word with a delay
def modify_output(input):
    # Iterate over each word in the input string
    for text in input.split(" "):
        # Yield the word with an added space
        yield text + " "
        # Introduce a small delay between each word
        time.sleep(0.05)

# #Para streamlit 
# ###############################################
st.set_page_config(page_title="BioBot",
                   page_icon="🦾")

st.markdown("<h1 style='text-align: center; color: white; font-family:serif;'>🧬BioBot🤖</h1>", unsafe_allow_html=True)

# markdown = """
# La app realiza consultas sobre los Reglamentos de licenciatura de la :orange[*Facultad de Ciencias Biologicas de la UJED*].
# - Puedes preguntar sobre los requisitos, 
#   examenes, derechos y obligaciones de la FCB-UJED.
# - Ejemplo: :orange[*¿Cuáles son las obligaciones de los laboratoristas?*] 
# \n
# *Nota*: El chatbot es algo :red[**PECULIAR**] 
# \n
# :grey-background[*Developed by Bruno Rodriguez*]
# """

#-------- Hide streamlit style ------------    
hide_st_style = '''
                    <style>
                    #Main Menu {visibility:hidden;}
                    footer {visibility:hidden;}
                    header {visibility:hidden;}
                    </style>
    '''
st.markdown(hide_st_style, unsafe_allow_html= True)

st.sidebar.markdown("""<h1>ℹ️ <em>Información</em>
                    <br>
                    V.1.0
                    <h1>""",unsafe_allow_html=True )

# st.sidebar.info(markdown)
st.sidebar.markdown(
    """
    <div style='padding:10px; text-align: justify; border-radius:5px; background-color:#000000; color:#fafafa; font-size: 16px; font-family:serif;'>
    La app realiza consultas sobre los Reglamentos de licenciatura de la <span style="color:orange;"><em>Facultad de Ciencias Biologicas de la UJED</em></span>
        <li>Puedes preguntar sobre los requisitos, examenes, derechos y obligaciones de la FCB-UJED.</li>
        <li>Ejemplo: <span style="color:orange;"><em>¿Cuáles son las obligaciones de los laboratoristas?</em></span></li>
        <br>
        <div style='text-align: center; font-family:serif; font-size: 12px;'><em>Nota: El chatbot es algo <span style='color:red;'>PECULIAR</span></em></div>
        <br>
        <div style='background-color:#fdf302; display:inline-block; padding:4px 8px; border-radius:4px; color:#000000; font-size: 12px; font-family:serif;'><i>Developed by Bruno Rodriguez</i></em></div>
    </div>
    """,
    unsafe_allow_html=True
)

logo = "./LOGO.png"
st.sidebar.image(logo)

# https://idoali.medium.com/building-a-report-chatbot-using-langchain-and-streamlit-7fc444487596
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
if prompt := st.chat_input('¿Cuál es tu pregunta sobre los reglamentos?'):
    #La pregunta que formulaste se quedara mostrada
    with st.chat_message('user', avatar="./Clicker.jpg"):
        st.markdown(prompt)
    
    st.session_state.messages.append({'role':'user','content':prompt})
    
    with st.spinner('Dejame pensar la respuesta...'): #Para crear un spinnner
        #La variable de respuesta
        response = make_output(prompt)

    with st.chat_message('Momos', avatar="🤖"):
         st.write_stream(modify_output(response))

    st.session_state.messages.append({'role':'Momos','content':response})   
