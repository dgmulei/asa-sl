import streamlit as st
import os
import json
import logging
import time
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
from utils.conversation_manager import ConversationManager, SessionManager, Message, ConversationContext
from utils.document_loader import DocumentLoader, Document, DocumentMetadata
from utils.embeddings_manager import EmbeddingsManager
from utils.query_engine import QueryEngine, QueryResult

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ===== INITIALIZATION FUNCTIONS =====
def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        os.getenv('CHROMA_DB_PATH', './chroma_db'),
        os.getenv('DOCUMENTS_PATH', './data/real_estate_docs'),
        '.streamlit'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def init_session_state():
    """Initialize all session state variables."""
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = ConversationContext(messages=[], system_message_added=False)
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0

# ===== CACHED RESOURCES =====
@st.cache_resource(show_spinner=False)
def get_embeddings_manager() -> EmbeddingsManager:
    model_name = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
    db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    return EmbeddingsManager(model_name=model_name, db_path=db_path)

@st.cache_resource(show_spinner=False)
def get_document_loader() -> DocumentLoader:
    documents_path = os.getenv('DOCUMENTS_PATH', './data/real_estate_docs')
    return DocumentLoader(documents_path)

@st.cache_resource
def initialize_components():
    logger.info("Starting initialization...")
    ensure_directories()
    loader = get_document_loader()
    embeddings_manager = get_embeddings_manager()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found")
    
    query_engine = QueryEngine("real_estate_docs")
    conversation_manager = ConversationManager(
        query_engine=query_engine,
        api_key=str(openai_api_key)
    )
    return loader, embeddings_manager, query_engine, conversation_manager

# ===== HELPER FUNCTIONS =====
def display_chat_messages():
    """Display chat messages excluding system messages."""
    for message in st.session_state.conversation_context.messages:
        if message.role != "system":
            if message.role == "assistant":
                with st.chat_message(message.role, avatar="static/images/asa_logo.png"):
                    st.markdown(message.content)
            else:
                with st.chat_message(message.role):
                    st.markdown(message.content)

def process_user_message(message: str, conversation_manager: ConversationManager):
    """Process user message and get response."""
    response = conversation_manager.get_response(message, st.session_state.conversation_context)
    st.session_state.chat_input_key += 1
    return True

def process_single_document(loader: DocumentLoader, embeddings_manager: EmbeddingsManager, file_path: str) -> None:
    """Process a single document and add to embeddings."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(file_path)
                chunks = loader.chunk_text(content)
                documents = []
                
                for i, (section_title, chunk) in enumerate(chunks):
                    metadata: DocumentMetadata = {
                        'source': str(filename),
                        'chunk_id': i,
                        'file_path': str(file_path),
                        'section_title': str(section_title)
                    }
                    doc = Document(text=chunk, metadata=metadata)
                    documents.append(doc)
                
                embeddings_manager.add_documents(documents)
                logger.info(f"Successfully processed single document: {filename}")
        except Exception as e:
            logger.error(f"Error processing single document {file_path}: {str(e)}")
            raise

# ===== MAIN APPLICATION =====
def main():
    st.set_page_config(
        page_title="Asa - Bainbridge Island Real Estate Advisor",
        page_icon="🏠",
        layout="wide"
    )
    
    init_session_state()
    
    # ===== CSS STYLING =====
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    .stApp header {
        background: linear-gradient(135deg, #2C3E50, #3498DB 120%);
        opacity: 0.95;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }
    
    h1 {
        color: #2C3E50;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: left;
        border-bottom: 3px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        max-width: 90%;
        transition: all 0.3s ease;
    }
    
    .stChatMessage.user {
        background: linear-gradient(135deg, #E8F4FD, #D1E8FA);
        margin-left: auto;
        margin-right: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stChatMessage.assistant {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        margin-left: 1rem;
        margin-right: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Chat Input Styling */
    .stChatInputContainer {
        background: white;
        border-top: 2px solid #E0E0E0;
        padding: 2rem 3rem;
        position: sticky;
        bottom: 0;
        z-index: 100;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
    }
    
    [data-testid="stChatInput"] {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .stChatInputContainer textarea {
        border: 2px solid #E0E0E0;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #3498DB;
        box-shadow: 0 2px 12px rgba(52,152,219,0.15);
    }
    
    /* Reset Button Styling */
    .stButton button {
        background: #f8f9fa;
        color: #2C3E50;
        border: 1px solid #E0E0E0;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        margin: 1rem auto 2rem;
        width: auto;
        min-width: 160px;
        display: block;
    }
    
    .stButton button:hover {
        background: #e9ecef;
        border-color: #ced4da;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    /* Expander Styling */
    [data-testid="stExpander"] {
        background: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        overflow: hidden;
    }
    
    .streamlit-expanderHeader {
        background: #F8F9FA;
        padding: 1.2rem 1.5rem;
        border-bottom: 1px solid #E0E0E0;
        font-weight: 600;
        color: #2C3E50;
    }
    
    .streamlit-expanderContent {
        padding: 1.8rem;
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #3498DB;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2980B9;
        background: #F8F9FA;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div {
        background-color: #3498DB;
    }
    
    /* Document List Styling */
    .document-list {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .document-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #E0E0E0;
        font-size: 0.9rem;
    }
    
    /* Typography */
    .stMarkdown {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #2C3E50;
    }
    
    /* Layout Improvements */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
    }
    
    [data-testid="column"] {
        padding: 0 1.5rem;
    }

    /* Mobile-Only Styles - These won't affect desktop */
    @media screen and (max-width: 768px) {
        /* Target the Streamlit chat input structure */
        .stChatInput {
            min-height: 200px !important;
        }
        
        [data-baseweb="textarea"] {
            min-height: 200px !important;
        }
        
        [data-baseweb="base-input"] {
            min-height: 200px !important;
        }
        
        /* Ensure the textarea itself is tall enough */
        [data-testid="stChatInputTextArea"] {
            min-height: 200px !important;
            height: 200px !important;
            font-size: 16px !important;
        }
        
        /* Add padding to prevent content from being hidden */
        .main .block-container {
            padding-bottom: 220px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        loader, embeddings_manager, query_engine, conversation_manager = initialize_components()
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return

    # ===== LAYOUT STRUCTURE =====
    main_container = st.container()
    with main_container:
        chat_col, info_col = st.columns([7, 3])
        
        with chat_col:
            st.title("Asa - Bainbridge Island Real Estate Advisor")
            display_chat_messages()
            
            key = f"chat_input_{st.session_state.chat_input_key}"
            if prompt := st.chat_input("What would you like to know about Bainbridge Island real estate?", key=key):
                if process_user_message(prompt, conversation_manager):
                    st.rerun()
        
        with info_col:
            if st.button("Reset Conversation", type="secondary"):
                st.session_state.conversation_context = ConversationContext(messages=[], system_message_added=False)
                st.session_state.chat_input_key += 1
                st.rerun()

            with st.expander("About Asa", expanded=True):
                st.write("I'm your professional real estate advisor for Bainbridge Island. I can help with:")
                st.markdown("""
                * Market analysis and trends
                * Property valuations
                * Investment strategies
                * Development opportunities
                * Local regulations and zoning
                """)
            
            with st.expander("Upload Documents", expanded=True):
                st.write("Upload new documents to analyze:")
                uploaded_file = st.file_uploader("Upload a document (.txt)", type="txt", key="file_uploader")
                
                if uploaded_file:
                    try:
                        save_path = os.path.join('./data/real_estate_docs', uploaded_file.name)
                        os.makedirs('./data/real_estate_docs', exist_ok=True)
                        
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        progress_text = "Processing document..."
                        progress_bar = st.progress(0, text=progress_text)
                        
                        try:
                            process_single_document(loader, embeddings_manager, save_path)
                            progress_bar.progress(100, text="Document processed successfully!")
                        finally:
                            time.sleep(1)
                            progress_bar.empty()
                            
                    except Exception as e:
                        st.error(f"Error uploading file: {str(e)}")
            
            with st.expander("Currently Loaded Documents", expanded=False):
                if os.path.exists(embeddings_manager.processed_files_path):
                    with open(embeddings_manager.processed_files_path, 'r') as f:
                        processed_files = json.load(f)
                        if processed_files:
                            st.markdown('<div class="document-list">', unsafe_allow_html=True)
                            for file in processed_files:
                                st.write(f"📄 {file}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.write("No documents currently loaded")

if __name__ == "__main__":
    main()
