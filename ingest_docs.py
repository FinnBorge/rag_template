"""
Example script to ingest documents into the RAG Bench system.
"""
import asyncio
from langchain.schema import Document as LangchainDocument
from rag_bench.settings.settings import Settings
from rag_bench.settings.settings_loader import load_settings
from rag_bench.components.vector_store_component import (
    PGVectorStoreComponent, 
    MockVectorStoreComponent,
    QdrantVectorStoreComponent
)
from rag_bench.components.embedding_component import get_embedding_component

async def ingest_sample_documents():
    # Load settings
    settings_dict = load_settings("settings.yaml")
    settings = Settings.model_validate(settings_dict)
    
    # Create vector store component based on settings
    vector_store_mode = settings.vectorstore.mode
    if vector_store_mode == "pgvector":
        vector_store = PGVectorStoreComponent(settings)
    elif vector_store_mode == "mock":
        vector_store = MockVectorStoreComponent(settings)
    elif vector_store_mode == "qdrant":
        vector_store = QdrantVectorStoreComponent(settings)
    else:
        raise ValueError(f"Unknown vector store mode: {vector_store_mode}")
    
    # Get the embedding component
    embedding_component = get_embedding_component(settings)
    
    # Create sample documents
    documents = [
        LangchainDocument(
            page_content="RAG (Retrieval Augmented Generation) is a technique that enhances LLM outputs with external knowledge.",
            metadata={"source": "sample", "title": "RAG Introduction"}
        ),
        LangchainDocument(
            page_content="Vector databases store and retrieve embeddings efficiently, enabling semantic search.",
            metadata={"source": "sample", "title": "Vector Databases"}
        ),
        LangchainDocument(
            page_content="Embeddings convert text into numerical vectors that capture semantic meaning.",
            metadata={"source": "sample", "title": "Embeddings"}
        ),
        LangchainDocument(
            page_content="Large Language Models (LLMs) are neural networks trained on vast text corpora to generate human-like text.",
            metadata={"source": "sample", "title": "LLMs"}
        ),
        LangchainDocument(
            page_content="Local LLMs can run on consumer hardware without requiring API calls to cloud services.",
            metadata={"source": "sample", "title": "Local LLMs"}
        ),
    ]
    
    # Add documents to vector store
    await vector_store.aadd_documents(documents, embedding_component)
    print(f"Successfully ingested {len(documents)} documents into {vector_store_mode}")

if __name__ == "__main__":
    asyncio.run(ingest_sample_documents())