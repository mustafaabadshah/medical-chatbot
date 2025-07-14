import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    """Load PDF files from the specified directory."""
    try:
        loader = PyPDFDirectoryLoader(data_path)
        documents = loader.load()
        if not documents:
            raise ValueError("No PDF documents found in the specified directory.")
        print(f"Loaded {len(documents)} PDF pages.")
        return documents
    except Exception as e:
        print(f"Error loading PDFs: {str(e)}")
        raise

def create_chunks(extracted_data):
    """Split documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(extracted_data)
        print(f"Created {len(text_chunks)} text chunks.")
        return text_chunks
    except Exception as e:
        print(f"Error creating chunks: {str(e)}")
        raise

def get_embedding_model():
    """Initialize the embedding model."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        raise

def create_vector_store():
    """Create and save FAISS vector store from PDF documents."""
    try:
        # Step 1: Load PDFs
        documents = load_pdf_files(DATA_PATH)
        
        # Step 2: Create chunks
        text_chunks = create_chunks(documents)
        
        # Step 3: Get embedding model
        embedding_model = get_embedding_model()
        
        # Step 4: Create and save FAISS index
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print(f"FAISS vector store saved successfully at {DB_FAISS_PATH}")
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_store()