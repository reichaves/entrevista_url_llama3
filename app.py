# -*- coding: utf-8
# Author: Reinaldo Chaves (reichaves@gmail.com)
# This project implements a conversational Retrieval-Augmented Generation (RAG) system
# using Streamlit, LangChain, and large language models to interview content from URLs
# Response generation uses the llama-3.2-90b-text-preview model from Meta
# Text embeddings use the all-MiniLM-L6-v2 model from Hugging Face
# I am grateful for Krish C Naik's classes (https://www.youtube.com/user/krishnaik06)

# Import necessary libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
import os
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain_core.outputs import ChatResult
from langchain_groq import ChatGroq
from pydantic import Field

# Configure Streamlit page settings
st.set_page_config(page_title="RAG Q&A Conversacional", layout="wide", initial_sidebar_state="expanded", page_icon="🤖", menu_items=None)

# Apply dark theme using custom CSS
# This section includes CSS to style various Streamlit components for a dark theme
st.markdown("""
    <style>
    /* Estilo global */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebarNav"] .stMarkdown {
        color: #fafafa !important;
    }
    
    /* Botões */
    .stButton > button {
        color: #4F8BF9 !important;
        background-color: #262730 !important;
        border-radius: 20px !important;
        height: 3em !important;
        width: 200px !important;
    }
    
    /* Inputs de texto */
    .stTextInput > div > div > input {
        color: #fafafa !important;
        background-color: #262730 !important;
    }
    
    /* Rótulos de input */
    .stTextInput > label, [data-baseweb="label"] {
        color: #fafafa !important;
        font-size: 1rem !important;
    }
    
    /* Garantindo visibilidade do texto em todo o app */
    .stApp > header + div, [data-testid="stAppViewContainer"] > div {
        color: #fafafa !important;
    }
    
    /* Forçando cor de texto para elementos específicos */
    div[class*="css"] {
        color: #fafafa !important;
    }
    
    /* Ajuste para elementos de entrada */
    [data-baseweb="base-input"] {
        background-color: #262730 !important;
    }
    [data-baseweb="base-input"] input {
        color: #fafafa !important;
    }
    
    /* Ajuste para o fundo do conteúdo principal */
    [data-testid="stAppViewContainer"] > section[data-testid="stSidebar"] + div {
        background-color: #0e1117 !important;
    }

    /* Forçando cor de fundo escura para todo o corpo da página */
    body {
        background-color: #0e1117 !important;
    }

    /* Ajustando cores para elementos de seleção e opções */
    .stSelectbox, .stMultiSelect {
        color: #fafafa !important;
        background-color: #262730 !important;
    }

    /* Ajustando cores para expansores */
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Ajustando cores para caixas de código */
    .stCodeBlock {
        background-color: #1e1e1e !important;
    }

    /* Ajustando cores para tabelas */
    .stTable {
        color: #fafafa !important;
        background-color: #262730 !important;
    }
    /* Estilo para o título principal */
    .yellow-title {
        color: yellow !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
    }

    /* Estilo para o título da sidebar */
    .orange-title {
        color: orange !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Sidebar with guidelines
st.sidebar.markdown("<h2 class='orange-title'>Orientações</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
* Se encontrar erros de processamento, reinicie com F5.
* Para recomeçar uma nova sessão pressione F5.
* Utilize URLs de sites que não tenham senha ou captcha. Não captura textos dentro de imagens e gráficos.

**Obtenção de chaves de API:**
* Você pode fazer uma conta no Groq Cloud e obter uma chave de API [aqui](https://console.groq.com/login)
* Você pode fazer uma conta no Hugging Face e obter o token de API de modo write Hugging Face [aqui](https://huggingface.co/docs/hub/security-tokens)

**Atenção:** O conteúdo das URLs que você compartilhar com o modelo de IA generativa pode ser usado pelo LLM para treinar o sistema. Portanto, evite compartilhar URLs que contenham:
1. Dados bancários e financeiros
2. Dados de sua própria empresa
3. Informações pessoais
4. Informações de propriedade intelectual
5. Conteúdos autorais

E não use IA para escrever um texto inteiro! O auxílio é melhor para gerar resumos, filtrar informações ou auxiliar a entender contextos - que depois devem ser checados. Inteligência Artificial comete erros (alucinações, viés, baixa qualidade, problemas éticos)!

Este projeto não se responsabiliza pelos conteúdos criados a partir deste site.

**Sobre este app**

Este aplicativo foi desenvolvido por Reinaldo Chaves. Para mais informações, contribuições e feedback, visite o [repositório do projeto no GitHub](https://github.com/reichaves/entrevista_url_llama3).
""")

# Main title and description
st.markdown("<h1 class='yellow-title'>Chatbot com modelos opensource - entrevista URLs ✏️</h1>", unsafe_allow_html=True)
st.write("Insira uma URL e converse com o conteúdo dela - aqui é usado o modelo de LLM llama-3.2-90b-text-preview e a plataforma de embeddings é all-MiniLM-L6-v2")

# Request API keys from the user
groq_api_key = st.text_input("Insira sua chave de API Groq (depois pressione Enter):", type="password")
huggingface_api_token = st.text_input("Insira seu token de API HuggingFace (depois pressione Enter):", type="password")

# Custom wrapper for ChatGroq with rate limiting
class RateLimitedChatGroq(BaseChatModel):
    llm: ChatGroq = Field(default_factory=lambda: ChatGroq())
    
    def __init__(self, groq_api_key: str, model_name: str, temperature: float = 0):
        super().__init__()
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return self.llm._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.error(f"Rate limit reached. Please try again in a few moments. Error: {str(e)}")
            else:
                st.error(f"An error occurred while processing your request: {str(e)}")
            raise e

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self.llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self):
        return "rate_limited_chat_groq"

# Main application logic
if groq_api_key and huggingface_api_token:
    # Set API tokens as environment variables
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Initialize language model and embeddings
    rate_limited_llm = RateLimitedChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-90b-text-preview", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a session ID for chat history management
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize session state for storing chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Get URL input from user    
    url = st.text_input("Insira a URL para análise:")

    if url:
        try:
            # Fetch and process the webpage content
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        
            # Extract text from the webpage
            text = soup.get_text(separator='\n', strip=True)
        
            # Limit the text to a maximum number of characters
            max_chars = 50000
            if len(text) > max_chars:
                text = text[:max_chars]
                st.warning(f"O conteúdo da página da web foi truncado para {max_chars} caracteres devido ao comprimento.")
        
            # Create a Document object
            document = Document(page_content=text, metadata={"source": url})

            # Split the document into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents([document])

            # Create FAISS vector store for efficient similarity search
            vectorstore = FAISS.from_documents(splits, embeddings)

            st.success(f"Processado {len(splits)} pedaços de documentos (chunks) da URL.")

            # Set up the retriever
            retriever = vectorstore.as_retriever()

            # Define the system prompt for contextualizing questions
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            # Create a history-aware retriever
            history_aware_retriever = create_history_aware_retriever(rate_limited_llm, retriever, contextualize_q_prompt)

            # Define the main system prompt for the chatbot
            system_prompt = (
                "Você é um assistente especializado em analisar conteúdo de páginas web. "
                "Sempre coloque no final das respostas: 'Todas as informações devem ser checadas com a(s) fonte(s) original(ais)'"
                "Responda em Português do Brasil a menos que seja pedido outro idioma"
                "Se você não sabe a resposta, diga que não sabe"
                "Siga estas diretrizes:\n\n"
                "1. Explique os passos de forma simples e mantenha as respostas concisas.\n"
                "2. Inclua links para ferramentas, pesquisas e páginas da Web citadas.\n"
                "3. Ao resumir passagens, escreva em nível universitário.\n"
                "4. Divida tópicos em partes menores e fáceis de entender quando relevante.\n"
                "5. Seja claro, breve, ordenado e direto nas respostas.\n"
                "6. Evite opiniões e mantenha-se neutro.\n"
                "7. Se não souber a resposta, admita que não sabe.\n\n"
                "Ao analisar o conteúdo da página web, considere:\n"
                "- O tema principal da página\n"
                "- A estrutura e organização do conteúdo\n"
                "- Informações relevantes e pontos-chave\n"
                "- Qualquer data ou informação temporal relevante\n"
                "- A fonte da informação e sua credibilidade\n\n"
                "Use o seguinte contexto para responder à pergunta: {context}\n\n"
                "Sempre termine as respostas com: 'Todas as informações precisam ser checadas com as fontes das informações'."
            )

            # Create the question-answering prompt template
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            # Create the question-answering chain
            question_answer_chain = create_stuff_documents_chain(rate_limited_llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Function to get or create session history
            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            # Create a conversational RAG chain with message history
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Get user input and process the question
            user_input = st.text_input("Sua pergunta:")
            if user_input:
                with st.spinner("Processando sua pergunta..."):
                    session_history = get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}},
                    )
                st.write("Assistente:", response['answer'])

                # Display chat history
                with st.expander("Ver histórico do chat"):
                    for message in session_history.messages:
                        st.write(f"**{message.type}:** {message.content}")
        # Error handling
        except requests.RequestException as e:
            st.error(f"Erro ao acessar a URL: {str(e)}")
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.error(f"Limite de taxa excedido para o modelo LLM. Tente novamente em alguns instantes. Erro: {str(e)}")
            else:
                st.error(f"Ocorreu um erro inesperado: {str(e)}")
else:
    st.warning("Por favor, insira tanto a chave da API do Groq quanto o token da API do Hugging Face.")
