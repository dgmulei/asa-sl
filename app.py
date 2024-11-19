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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

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
        
    query_engine = QueryEngine(embeddings_manager.get_collection())
    conversation_manager = ConversationManager(
        query_engine=query_engine,
        api_key=str(openai_api_key)
    )
    return loader, embeddings_manager, query_engine, conversation_manager

def display_chat_messages():
    # Only display user and assistant messages, not system messages
    for message in st.session_state.conversation_context.messages:
        if message.role != "system":
            with st.chat_message(message.role):
                st.markdown(message.content)

def process_user_message(message: str, conversation_manager: ConversationManager):
    # Process the message and get response
    response = conversation_manager.get_response(message, st.session_state.conversation_context)
    # Increment the chat input key to force a new input field
    st.session_state.chat_input_key += 1
    return True

def process_single_document(loader: DocumentLoader, embeddings_manager: EmbeddingsManager, file_path: str) -> None:
    """Process a single document instead of the entire directory."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(file_path)
                
                # Create a document object similar to how DocumentLoader does it
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

def main():
    st.set_page_config(
        page_title="Asa - Bainbridge Island Real Estate Advisor",
        page_icon="🏠",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stChatMessage.user {
            background-color: #f0f2f6;
        }
        .stChatMessage.assistant {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
        }
        .citation {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .stAlert {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    try:
        loader, embeddings_manager, query_engine, conversation_manager = initialize_components()
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return

    st.title("Asa - Bainbridge Island Real Estate Advisor")
    
    # Add a reset button
    if st.sidebar.button("Reset Conversation"):
        st.session_state.conversation_context = ConversationContext(messages=[], system_message_added=False)
        st.session_state.chat_input_key += 1
        st.rerun()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display existing messages
        display_chat_messages()
        
        # Handle new message with unique key
        key = f"chat_input_{st.session_state.chat_input_key}"
        if prompt := st.chat_input("What would you like to know about Bainbridge Island real estate?", key=key):
            # Process the message and get response
            if process_user_message(prompt, conversation_manager):
                st.rerun()
    
    with col2:
        st.write("About Asa")
        st.markdown("""
            I'm your professional real estate advisor for Bainbridge Island. I can help with:
            - Market analysis and trends
            - Property valuations
            - Investment strategies
            - Development opportunities
            - Local regulations and zoning
        """)
        
        st.write("---")
        st.write("Upload new documents to analyze:")
        
        uploaded_file = st.file_uploader("Upload a document (.txt)", type="txt", key="file_uploader")
        
        # Only process if we have a new upload
        if uploaded_file:
            try:
                save_path = os.path.join('./data/real_estate_docs', uploaded_file.name)
                os.makedirs('./data/real_estate_docs', exist_ok=True)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                progress_text = "Processing document..."
                progress_bar = st.progress(0, text=progress_text)
                
                try:
                    # Process only the newly uploaded file
                    process_single_document(loader, embeddings_manager, save_path)
                    progress_bar.progress(100, text="Document processed successfully!")
                finally:
                    time.sleep(1)
                    progress_bar.empty()
                    
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
        
        st.write("---")
        st.write("Currently loaded documents:")
        
        if os.path.exists(embeddings_manager.processed_files_path):
            with open(embeddings_manager.processed_files_path, 'r') as f:
                processed_files = json.load(f)
                if processed_files:
                    for file in processed_files:
                        st.write(f"- {file}")
                else:
                    st.write("No documents currently loaded")

if __name__ == "__main__":
    main()
