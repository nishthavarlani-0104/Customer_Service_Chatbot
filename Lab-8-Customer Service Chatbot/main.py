import sys
from pathlib import Path
import config
from document_processor import process_document
from vector_store_setup import store_documents, collection_exists, get_qdrant_client, create_collection_if_not_exists
from customer_support_chat import create_rag_chain
import uuid

def setup_vector_store():
    """
    Setup vector store: check if it exists, if not process and store documents
    """
    print("=" * 60)
    print("Customer Support Chatbot - Vector Store Setup")
    print("=" * 60)
    
    # Check if PDF exists
    pdf_path = Path(config.PDF_PATH)
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please ensure the PDF file is in the current directory.")
        sys.exit(1)
    
    # Check if collection exists and has documents
    client = get_qdrant_client()
    collection_exists_flag = collection_exists(client, config.QDRANT_COLLECTION_NAME)
    
    if collection_exists_flag:
        
        collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
        if collection_info.points_count > 0:
            print(f"Vector store already exists with {collection_info.points_count} documents")
            print("Skipping document processing...")
            return
    else:
        create_collection_if_not_exists(client, config.QDRANT_COLLECTION_NAME)
    
    # Process and store documents
    print("Processing PDF document...")
    chunks = process_document()
    
    print("\nStoring documents in Qdrant Cloud...")
    store_documents(chunks)
    print("Vector store setup complete!\n")


def main():
    print("=" * 60)
    print("SolarNova Dynamics - Customer Support Chatbot")
    print("=" * 60)
    print()
    session_id = uuid.uuid4()
    session_id = str(session_id)
    # Setup vector store
    try:
        setup_vector_store()
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        print("Please check your Qdrant Cloud credentials in .env file")
        sys.exit(1)
    
    # Initialize chatbot
    print("Initializing chatbot...")
    try:
        chatbot = create_rag_chain()  #RAG chain with memory
        chat_config = {"configurable": {"session_id": session_id}}
        print("Chatbot ready!\n")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)
    
    # Interactive chat loop
    print("=" * 60)
    print("Chat Interface")
    print("Type your questions about SolarNova Dynamics")
    print("Commands: 'quit' or 'exit' to end, 'new' for new session")
    print("=" * 60)
    print()
        
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using SolarNova Dynamics Customer Support!")
                break
            
            if not user_input:
                continue
            
            # Get response from chatbot
            print("Bot: ", end="", flush=True)
            response = chatbot.invoke(
                {"input": user_input},
                config=chat_config
            )
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nThank you for using SolarNova Dynamics Customer Support!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()