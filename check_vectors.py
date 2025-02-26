#\!/usr/bin/env python3
"""
Script to check if documents are properly stored in the vector database.
"""
import asyncio
import sys
from langchain.schema import Document as LangchainDocument
from rag_bench.settings.settings import Settings
from rag_bench.settings.settings_loader import load_settings
from rag_bench.components.vector_store_component import PGVectorStoreComponent

async def check_vectorstore():
    # Load settings
    settings_dict = load_settings("settings.yaml")
    settings = Settings.model_validate(settings_dict)
    
    # Create vector store component
    vector_store = PGVectorStoreComponent(settings)
    
    # Query the vector store directly
    print("Checking database...")
    
    try:
        # Use a simple query to retrieve documents
        docs = await vector_store.asimilarity_search_with_scores("RAG", k=5)
        
        if not docs:
            print("No documents found in the vector store. Database may be empty.")
            sys.exit(1)
        
        print(f"Found {len(docs)} documents in the vector store:")
        for i, (doc, score) in enumerate(docs):
            print(f"\nDocument {i+1} (score: {score:.4f}):")
            print(f"Content: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
        
        # Try to get the count of documents
        # This requires direct SQL access
        try:
            from sqlalchemy import text
            conn = vector_store.engine.connect()
            result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding"))
            count = result.fetchone()[0]
            print(f"\nTotal documents in database: {count}")
        except Exception as e:
            print(f"Couldn't get document count: {e}")
        
    except Exception as e:
        print(f"Error checking vector store: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(check_vectorstore())
