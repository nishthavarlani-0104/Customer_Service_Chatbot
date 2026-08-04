from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import Qdrant

#RAG -- CHain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import config
from vector_store_setup import get_vector_store


def get_sql_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get SQL-based chat history for a session
    
    Args:
        session_id: Unique identifier for the conversation session
        
    Returns:
        SQLChatMessageHistory instance
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{config.CHAT_DB_PATH}"
    )


def create_rag_chain(vector_store: Qdrant = None):
    """
    Create RAG chain using LangChain's built-in create_stuff_documents_chain
    
    This uses LangChain's standardized function for combining documents,
    which handles document formatting automatically. The chain retrieves
    relevant documents and combines them with the prompt and chat history.
    
    Args:
        vector_store: Qdrant vector store instance (if None, will load from config)
        
    Returns:
        RAG chain ready to be used with message history
    """
    # Load vector store if not provided
    if vector_store is None:
        vector_store = get_vector_store()
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_K}  #k -- no of chunks to retrieve
    )

   
    #ollama
    if not config.LLM_BASE_URL:
        llm = init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER)
    else:
        llm = init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, base_url=config.LLM_BASE_URL, api_key = config.LLM_API_KEY)
   
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support assistant for SolarNova Dynamics. 
Your role is to answer questions about SolarNova Dynamics products, services, and business information based on the provided context from the company's business document.

Guidelines:
- Answer questions based ONLY on the provided context from the document
- If the information is not in the context, politely say you don't have that information
- Be friendly, professional, and helpful
- Provide clear and concise answers
- If asked about something not in the document, suggest contacting customer support directly

Use the following context to answer the user's question:
{context}"""),#context -- relevvant docs
        MessagesPlaceholder(variable_name="chat_history"),#chat_history -- previous conversation history
        ("human", "{input}")#input -- user question
    ])
    
  
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )
    
    
    def get_context(x):
        # print("Relevant Chunks:")
        context = retriever.invoke(x.get("input", "")) #Retrieve relevant chunks
        # for chunk in context:
        #     print("Chunk Content: ", chunk.page_content)
        #     print("--------------------------------")
        #     print("Chunk Metadata: ", chunk.metadata)
        #     print("--------------------------------")
        # print("--------------------------------")
        return context

    rag_chain = (
        RunnablePassthrough.assign(context=get_context) | combine_docs_chain
    )

    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_sql_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return chain_with_history
