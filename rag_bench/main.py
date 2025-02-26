"""
Main entry point for the RAG benchmarking application.
"""
import logging
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector, inject, singleton

from rag_bench.core.types import EmbeddingComponent, LLMComponent, VectorStoreComponent
from rag_bench.core.document_processors import (
    ThresholdFilter, 
    SemanticReranker, 
    LLMReranker,
    DiversityReranker,
    ProcessingPipeline
)
from rag_bench.core.engine import RAGEngine, SourcedAnswer
from rag_bench.core.query_enhancers import (
    HyponymExpansionEnhancer,
    LLMQueryExpansionEnhancer,
    HybridQueryEnhancer,
    StopWordRemovalEnhancer
)
from rag_bench.core.types import QueryEnhancer, DocumentPostProcessor
from rag_bench.dependency_injection import configure_injection
from rag_bench.routers.api_v1.chat.chat_router import router as chat_router
from rag_bench.settings.settings import Settings
from rag_bench.settings.settings_loader import load_settings

logger = logging.getLogger(__name__)

# Load settings from configuration file
settings_path = os.environ.get("SETTINGS_PATH", "settings.yaml")
settings_dict = load_settings(settings_path)
settings = Settings.model_validate(settings_dict)

# Set up dependency injection
injector = Injector([
    lambda binder: binder.bind(Settings, to=settings, scope=singleton),
    configure_injection
])

app = FastAPI(
    title="RAG Benchmarking System",
    description="A system for benchmarking and evaluating RAG pipelines",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "RAG Benchmark system is running"}


@app.get("/health")
async def health():
    """Health check endpoint with component status."""
    return {
        "status": "ok",
        "components": {
            "llm": True,
            "embedding": True,
            "vectorstore": True,
        }
    }


# Create a simple endpoint to test the RAG engine
def get_rag_engine():
    """Get the RAG engine from the injector."""
    return injector.get(RAGEngine)

@app.get("/api/v1/query")
async def query(
    q: str,
    conversation_id: Optional[str] = None,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Query endpoint for testing the RAG engine.
    
    Args:
        q: The query string
        conversation_id: Optional conversation ID for context
        
    Returns:
        Answer with sources
    """
    logger.info(f"Received query: {q}")
    
    answer = await rag_engine.generate_answer(q, conversation_id)
    
    return answer


# Define a function to set up query enhancers
@singleton
@inject
def setup_query_enhancers(
    settings: Settings,
    llm_component: LLMComponent
) -> List[QueryEnhancer]:
    """Set up query enhancers based on settings."""
    enhancers = []
    
    # Example hyponym map - in a real implementation, this would be loaded from a data file
    hyponym_map = {
        "medication": ["drug", "pill", "capsule", "tablet", "prescription"],
        "doctor": ["physician", "specialist", "clinician", "surgeon", "practitioner"],
        "symptoms": ["signs", "indications", "manifestations"],
    }
    
    # Add stop word removal enhancer
    enhancers.append(StopWordRemovalEnhancer())
    
    # Add hyponym expansion enhancer
    enhancers.append(HyponymExpansionEnhancer(hyponym_map))
    
    # Add LLM query expansion enhancer if configured
    if settings.llm.mode != "mock":
        enhancers.append(LLMQueryExpansionEnhancer(llm_component))
    
    return enhancers


# Define a function to set up document post-processors
@singleton
@inject
def setup_document_processors(
    settings: Settings,
    embedding_component: EmbeddingComponent,
    llm_component: LLMComponent
) -> List[DocumentPostProcessor]:
    """Set up document post-processors based on settings."""
    processors = []
    
    # Add threshold filter
    processors.append(ThresholdFilter(threshold=settings.rag.similarity_threshold))
    
    # Add semantic reranker
    processors.append(SemanticReranker(embedding_component))
    
    # Add LLM reranker if configured
    if settings.llm.mode != "mock":
        processors.append(LLMReranker(llm_component))
    
    # Add diversity reranker
    processors.append(DiversityReranker(embedding_component, diversity_weight=0.3))
    
    return processors


# Keep track of the query enhancers and document processors we've configured
try:
    # Try to get the LLM component
    llm_component = injector.get(LLMComponent)
    embedding_component = injector.get(EmbeddingComponent)
    
    query_enhancers = setup_query_enhancers(settings, llm_component)
    document_processors = setup_document_processors(settings, embedding_component, llm_component)
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    logger.error("\nTROUBLESHOOTING STEPS:")
    logger.error("1. Check that the model file exists in the models directory")
    logger.error("2. Run 'poetry run python initialize_models.py' to download the model")
    logger.error("3. Or set 'llm.mode: mock' in settings.yaml for a mock implementation")
    logger.error("4. Run 'poetry run python setup_all.py' for complete setup\n")
    
    logger.warning("Using mock components as fallback...")
    
    # Switch to mock mode in settings
    settings.llm.mode = "mock"
    settings.embedding.mode = "mock"
    
    # Reinitialize injector with mock components
    from rag_bench.dependency_injection import AppModule
    mock_injector = Injector([AppModule()])
    
    # Get mock components
    llm_component = mock_injector.get(LLMComponent)
    embedding_component = mock_injector.get(EmbeddingComponent)
    
    query_enhancers = setup_query_enhancers(settings, llm_component)
    document_processors = setup_document_processors(settings, embedding_component, llm_component)


def run_server():
    """Run the application server."""
    uvicorn.run(
        "rag_bench.main:app",
        host="0.0.0.0",
        port=settings.server.port,
        reload=settings.server.env_name == "development"
    )


if __name__ == "__main__":
    run_server()