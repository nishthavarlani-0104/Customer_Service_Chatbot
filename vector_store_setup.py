from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import config
from document_processor import process_document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_qdrant_client():
    """
    Create and return Qdrant Cloud client
    
    Returns:
        QdrantClient instance connected to Qdrant Cloud
    """
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
       
    )
    return client


def get_embeddings():
    """
    Initialize embeddings model
    
    Returns:
        HuggingFaceEmbeddings instance

    """
    if config.EMBEDDING_MODEL_TYPE == "HUGGING_FACE":
        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    elif config.EMBEDDING_MODEL_TYPE == "OPENAI":
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    Check if a collection exists in Qdrant
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to check
        
    Returns:
        True if collection exists, False otherwise
    """
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        return collection_name in collection_names
    except UnexpectedResponse as e:
        if e.status_code == 404:
            raise ConnectionError(
                "Qdrant cluster returned 404. The cluster URL may be wrong or the cluster "
                "was deleted. Update QDRANT_URL and QDRANT_API_KEY in your .env file from "
                "the Qdrant Cloud dashboard."
            ) from e
        raise
    except Exception as e:
        print(f"Error checking collection existence: {e}")
        return False


def create_collection_if_not_exists(client: QdrantClient, collection_name: str):
    """
    Create Qdrant collection if it doesn't exist
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
    """
    
    print(f"Creating collection '{collection_name}' in Qdrant Cloud...")
    try:
        if not collection_exists(client, collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE
                ),
            )
            print(f"Collection '{collection_name}' created successfully")
    except UnexpectedResponse as e:
        if "already exists" in str(e).lower():
            print(f"Collection '{collection_name}' already exists")
        else:
            raise


def get_vector_store(collection_name: str = None):
    """
    Get or create Qdrant vector store
    
    Args:
        collection_name: Name of the collection (default from config)
        
    Returns:
        Qdrant vector store instance
    """
    collection_name = collection_name or config.QDRANT_COLLECTION_NAME
    client = get_qdrant_client()
    embeddings = get_embeddings()

    create_collection_if_not_exists(client, collection_name)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )
    
    return vector_store


def store_documents(documents: list[Document], collection_name: str = None):
    """
    Store documents in Qdrant vector store
    
    Args:
        documents: List of Document objects to store
        collection_name: Name of the collection (default from config)
        
    Returns:
        Qdrant vector store instance
    """
    collection_name = collection_name or config.QDRANT_COLLECTION_NAME
    
    # Get vector store
    vector_store = get_vector_store(collection_name)
    
    # Store documents
    print(f"Storing {len(documents)} documents in Qdrant Cloud...")
    vector_store.add_documents(documents)
    print(f"Successfully stored {len(documents)} documents in collection '{collection_name}'")
    
    return vector_store


if __name__ == "__main__":
    chunks = process_document()
    store_documents(chunks)