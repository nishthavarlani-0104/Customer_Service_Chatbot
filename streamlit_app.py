"""
Streamlit Web UI for Customer Support Chatbot
Provides a user-friendly web interface for the RAG-based chatbot
"""
import streamlit as st
import time
from pathlib import Path
import config
from document_processor import process_document
from vector_store_setup import get_vector_store, store_documents, collection_exists, get_qdrant_client, create_collection_if_not_exists
from customer_support_chat import create_rag_chain


# Page configuration
st.set_page_config(
    page_title="SolarNova Dynamics - Customer Support",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)



def setup_vector_store():
    """
    Check and setup vector store if needed
    """
    try:
        client = get_qdrant_client()
        collection_exists_flag = collection_exists(client, config.QDRANT_COLLECTION_NAME)
        
        if collection_exists_flag:
            collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
            if collection_info.points_count > 0:
                return True
        else:
            create_collection_if_not_exists(client, config.QDRANT_COLLECTION_NAME)
        
        # Need to process documents
        with st.spinner("Loading knowledge base..."):
            pdf_path = Path(config.PDF_PATH)
            if not pdf_path.exists():
                st.error(f"Knowledge base file not found. Please contact support.")
                return False
            
            chunks = process_document()
            store_documents(chunks)
            return True
    except Exception as e:
       
        return False


@st.cache_resource
def initialize_chatbot():
    """
    Initialize chatbot with caching to avoid re-initialization
    """
    try:
        vector_store = get_vector_store()
        chatbot = create_rag_chain(vector_store=vector_store)
        return chatbot
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None

def main():
    """
    Main Streamlit application
    """
    # Header
    st.markdown('<div class="main-header">💬 SolarNova Dynamics Customer Support</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - simplified for end users
    with st.sidebar:
        st.header("💬 Chat Options")
        
        if st.button("🗑️ Clear Chat History"):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            # Generate new session ID when clearing chat
            st.session_state.session_id = f"session_{int(time.time())}"
            st.rerun()
        
        st.markdown("---")
        st.markdown("**How can I help?**")
        st.markdown("Ask me anything about SolarNova Dynamics products, services, or support.")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    
    
    setup_vector_store()
     
    
    # Initialize chatbot
    current_session_id = st.session_state.get('session_id')
    
    chatbot = initialize_chatbot()
    
    if chatbot is None:
        st.error("⚠️ Unable to connect to the support system. Please try refreshing the page.")
        st.stop()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
    
    # Chat input
    if prompt := st.chat_input("Ask a question about SolarNova Dynamics..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_config = {"configurable": {"session_id": current_session_id}}
                    response = chatbot.invoke(
                        {"input": prompt},
                        config=chat_config
                    )
                    
                    # Display response
                    st.write(response)
                    
                    # Add bot response to chat history
                    bot_message = {"role": "assistant", "content": response}
                    
                    st.session_state.messages.append(bot_message)
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
