"""
Main entry point for the RAG benchmarking application.
"""
import logging
import os
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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


def create_settings() -> Settings:
    """Load and validate settings from configuration file."""
    settings_path = os.environ.get("SETTINGS_PATH", "settings.yaml")
    settings_dict = load_settings(settings_path)
    return Settings.model_validate(settings_dict)


def create_injector(settings: Settings) -> Injector:
    """Create dependency injection container."""
    return Injector([
        lambda binder: binder.bind(Settings, to=settings, scope=singleton),
        configure_injection
    ])


def get_cors_origins(settings: Settings) -> List[str]:
    """Get CORS origins based on environment."""
    if settings.server.env_name == "production":
        # In production, this should be configured via settings
        return ["https://your-production-domain.com"]
    # Development mode - allow all origins
    return ["*"]


# Load settings and create injector
settings = create_settings()
injector = create_injector(settings)


app = FastAPI(
    title="RAG Benchmarking System",
    description="A system for benchmarking and evaluating RAG pipelines",
    version="0.1.0"
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface HTML page."""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    return HTMLResponse(
        content="<html><body><h1>Chat interface not found</h1>"
        "<p>Please create the static/index.html file.</p></body></html>"
    )


@app.middleware("http")
async def inject_injector(request: Request, call_next):
    """Middleware to inject the DI container into request state."""
    request.state.injector = injector
    response = await call_next(request)
    return response


# Add CORS middleware with environment-aware origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(settings),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(chat_router, prefix="/api/v1")


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


def get_rag_engine():
    """Get the RAG engine from the injector."""
    return injector.get(RAGEngine)


def _build_query_response(
    answer: SourcedAnswer,
    conversation_id: Optional[str],
    total_time_ms: float,
) -> dict:
    """Build a response dict from a SourcedAnswer without mutating it."""
    docs_retrieved = len(answer.sources) if answer.sources else 0
    docs_used = sum(
        1 for src in answer.sources
        if src.get("relevance_score", 0) > 0.7
    ) if answer.sources else 0

    return {
        "answer": answer.answer,
        "sources": answer.sources,
        "conversation_id": conversation_id,
        "metrics": {
            "total_time_ms": total_time_ms,
            "documents_retrieved": docs_retrieved,
            "documents_used": docs_used,
        }
    }


def _build_error_response(
    query: str,
    conversation_id: Optional[str],
    total_time_ms: float,
    error: Optional[str] = None,
) -> dict:
    """Build a fallback response for errors or empty results."""
    response = {
        "answer": "I don't have specific information about that in my documents.",
        "sources": [],
        "conversation_id": conversation_id,
        "metrics": {
            "total_time_ms": total_time_ms,
            "documents_retrieved": 0,
            "documents_used": 0,
        }
    }
    if error:
        response["metrics"]["error"] = error
    return response


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
        Answer with sources and metrics
    """
    logger.info(f"Received query: {q}")
    start_time = time.time()

    try:
        answer = await rag_engine.generate_answer(q, conversation_id)
        total_time_ms = (time.time() - start_time) * 1000

        if not answer or not answer.answer:
            logger.warning(f"No answer generated for query: {q}")
            return _build_error_response(q, conversation_id, total_time_ms)

        return _build_query_response(answer, conversation_id, total_time_ms)

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        total_time_ms = (time.time() - start_time) * 1000
        return _build_error_response(q, conversation_id, total_time_ms, str(e))


def setup_query_enhancers(
    app_settings: Settings,
    llm_component: LLMComponent
) -> List[QueryEnhancer]:
    """Set up query enhancers based on settings."""
    # Hyponym map - in production, load from config file
    hyponym_map = {
        "medication": ["drug", "pill", "capsule", "tablet", "prescription"],
        "doctor": ["physician", "specialist", "clinician", "surgeon", "practitioner"],
        "symptoms": ["signs", "indications", "manifestations"],
    }

    enhancers: List[QueryEnhancer] = [
        StopWordRemovalEnhancer(),
        HyponymExpansionEnhancer(hyponym_map),
    ]

    if app_settings.llm.mode != "mock":
        enhancers.append(LLMQueryExpansionEnhancer(llm_component))

    return enhancers


def setup_document_processors(
    app_settings: Settings,
    embedding_component: EmbeddingComponent,
    llm_component: LLMComponent
) -> List[DocumentPostProcessor]:
    """Set up document post-processors based on settings."""
    processors: List[DocumentPostProcessor] = [
        ThresholdFilter(threshold=app_settings.rag.similarity_threshold),
        SemanticReranker(embedding_component),
    ]

    if app_settings.llm.mode != "mock":
        processors.append(LLMReranker(llm_component))

    processors.append(DiversityReranker(embedding_component, diversity_weight=0.3))

    return processors


def initialize_components(app_settings: Settings, app_injector: Injector):
    """
    Initialize RAG components. Falls back to mock mode on failure.

    Returns a new settings object if fallback was needed (immutable pattern).
    """
    try:
        llm_component = app_injector.get(LLMComponent)
        embedding_component = app_injector.get(EmbeddingComponent)

        query_enhancers = setup_query_enhancers(app_settings, llm_component)
        document_processors = setup_document_processors(
            app_settings, embedding_component, llm_component
        )

        return app_settings, query_enhancers, document_processors

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.error("\nTROUBLESHOOTING STEPS:")
        logger.error("1. Check that the model file exists in the models directory")
        logger.error("2. Run 'poetry run python initialize_models.py' to download the model")
        logger.error("3. Or set 'llm.mode: mock' in settings.yaml for a mock implementation")
        logger.error("4. Run 'poetry run python setup_all.py' for complete setup\n")

        logger.warning("Using mock components as fallback...")

        # Create new settings with mock mode (immutable - don't modify original)
        mock_settings_dict = app_settings.model_dump()
        mock_settings_dict["llm"]["mode"] = "mock"
        mock_settings_dict["embedding"]["mode"] = "mock"
        mock_settings = Settings.model_validate(mock_settings_dict)

        # Create new injector with mock settings
        from rag_bench.dependency_injection import AppModule
        mock_injector = Injector([AppModule()])

        llm_component = mock_injector.get(LLMComponent)
        embedding_component = mock_injector.get(EmbeddingComponent)

        query_enhancers = setup_query_enhancers(mock_settings, llm_component)
        document_processors = setup_document_processors(
            mock_settings, embedding_component, llm_component
        )

        return mock_settings, query_enhancers, document_processors


# Initialize components at module load
settings, query_enhancers, document_processors = initialize_components(settings, injector)


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
