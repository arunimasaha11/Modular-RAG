import os
from pinecone import ServerlessSpec, Pinecone 
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ensure_index_exists():
    """Create the Pinecone index if it doesn't already exist. Idempotent — safe to call from anywhere."""
    indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new Index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully.")


# Retriever Function
def get_retriever():
    """Initialize and return the Pinecone vector store retriever."""
    ensure_index_exists()
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings) 
    return vectorstore.as_retriever()
        

# Uploading Document to Vector Store
def add_document(text_content: str):
    """
    Add a single document to the Pinecone vector store. Split the text into chunks before embeddings and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty. Please provide valid text content.")

    ensure_index_exists()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    
    # Create LangChain document objects from the Raw Text
    documents = text_splitter.create_documents([text_content])
    
    print("Splitting the document into chunks and adding to the vector store...")
    
    # Vector Store Instance to add documents
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    # Adding Documents to the Vector Store
    vectorstore.add_documents(documents)
    print(f"Document added successfully to the vector store '{PINECONE_INDEX_NAME}'. Total chunks added: {len(documents)}")