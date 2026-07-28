from langchain_community.document_loaders import PyPDFLoader # Extract the data from the pdf file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import config



def load_pdf(pdf_path: str) -> list[Document]:
    """
    Load PDF document and extract text with metadata
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects with page content and metadata
    """
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load() #Read the text from the PDF file and give result as a list of Document objects
    print(f"Loaded {len(documents)} pages from PDF")
    return documents


def split_documents(documents: list[Document], chunk_size: int = None, chunk_overlap: int = None) -> list[Document]:
    """
    Split documents into smaller chunks for embedding
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
        
    Returns:
        List of split Document objects with metadata
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    print(f"Splitting documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split documents
    split_docs = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, doc in enumerate(split_docs):
        if "chunk_index" not in doc.metadata:
            doc.metadata["chunk_index"] = i
        if "source" not in doc.metadata:
            doc.metadata["source"] = Path(documents[0].metadata.get("source", "unknown")).name
    
    print(f"Split into {len(split_docs)} chunks")
    return split_docs

def process_document(pdf_path: str = None) -> list[Document]: #given the pdf, return the chunks
    """
    Complete document processing pipeline: load and split
    
    Args:
        pdf_path: Path to PDF file (default from config)
        
    Returns:
        List of processed Document chunks ready for embedding
    """
    pdf_path = pdf_path or config.PDF_PATH
    
    # Load PDF
    documents = load_pdf(pdf_path)
    
    # Split into chunks
    chunks = split_documents(documents)
    
    return chunks


if __name__ == "__main__":
    pdf_path = config.PDF_PATH
    documents = load_pdf(pdf_path)
    # print(f"documents : {documents}")
    # for each_doc in documents:
    #     print(each_doc)
    docs = split_documents(documents)
    # for each_chunck in docs:
    #     print(each_chunck)
    #     print("======================")