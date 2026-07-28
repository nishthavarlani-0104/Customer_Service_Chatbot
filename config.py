"""
Configuration file for Customer Support Chatbot with RAG
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Qdrant Cloud Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "solarnova_customer_support"


# OpenAI Embeddings Configuration
EMBEDDING_MODEL_TYPE="HUGGING_FACE"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# LLM Configuration
LLM_MODEL = "openai/gpt-oss-120b"
LLM_PROVIDER = "openai"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY= os.getenv("GROQ_API_KEY")

# Document Processing Configuration
PDF_PATH = "data/SolarNova Dynamics - Business Document.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
RETRIEVAL_K = 3  # Number of chunks to retrieve

# Chat History Configuration
CHAT_DB_PATH = "customer_support_conversations.db"

