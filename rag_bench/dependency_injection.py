from typing import Optional, Dict, Any, Callable, Type
import logging

from injector import Injector, Module, singleton, provider, Binder, inject

from rag_bench.settings.settings import Settings, settings
from rag_bench.components.llm_component import OpenAILLMComponent, AnthropicLLMComponent, MockLLMComponent, LocalLLMComponent
from rag_bench.components.vector_store_component import PGVectorStoreComponent, QdrantVectorStoreComponent, MockVectorStoreComponent
from rag_bench.components.embedding_component import OpenAIEmbeddingComponent, MockEmbeddingComponent, HuggingFaceEmbeddingComponent
from rag_bench.core.types import LLMComponent, VectorStoreComponent, EmbeddingComponent


logger = logging.getLogger(__name__)


class AppModule(Module):
    """Main dependency injection module for the application."""
    
    def configure(self, binder: Binder) -> None:
        """Configure bindings for the injector."""
        # Bind the settings singleton
        binder.bind(Settings, to=settings(), scope=singleton)
    
    @provider
    @singleton
    def provide_llm_component(self, settings: Settings) -> LLMComponent:
        """Provide the appropriate LLM component based on settings."""
        llm_mode = settings.llm.mode
        
        if llm_mode == "openai":
            return OpenAILLMComponent(settings)
        elif llm_mode == "anthropic":
            return AnthropicLLMComponent(settings)
        elif llm_mode == "local":
            return LocalLLMComponent(settings)
        elif llm_mode == "mock":
            return MockLLMComponent(settings)
        else:
            raise ValueError(f"Unknown LLM mode: {llm_mode}")
    
    @provider
    @singleton
    def provide_vector_store_component(self, settings: Settings) -> VectorStoreComponent:
        """Provide the appropriate vector store component based on settings."""
        vector_store_mode = settings.vectorstore.mode
        
        if vector_store_mode == "pgvector":
            return PGVectorStoreComponent(settings)
        elif vector_store_mode == "qdrant":
            return QdrantVectorStoreComponent(settings)
        elif vector_store_mode == "mock":
            return MockVectorStoreComponent(settings)
        else:
            raise ValueError(f"Unknown vector store mode: {vector_store_mode}")
    
    @provider
    @singleton
    def provide_embedding_component(self, settings: Settings) -> EmbeddingComponent:
        """Provide the appropriate embedding component based on settings."""
        embedding_mode = settings.embedding.mode
        
        if embedding_mode == "openai":
            return OpenAIEmbeddingComponent(settings)
        elif embedding_mode == "huggingface":
            return HuggingFaceEmbeddingComponent(settings)
        elif embedding_mode == "mock":
            return MockEmbeddingComponent(settings)
        else:
            raise ValueError(f"Unknown embedding mode: {embedding_mode}")


# Global injector instance
global_injector = Injector([AppModule()])


@inject
def provide_rag_engine(settings: Settings, 
                     llm_component: LLMComponent, 
                     vector_store_component: VectorStoreComponent) -> Any:
    """Provide an instance of RAGEngine."""
    from rag_bench.core.engine import RAGEngine
    return RAGEngine(
        settings=settings,
        llm_component=llm_component,
        vector_store_component=vector_store_component,
        query_enhancers=[],
        document_post_processors=[]
    )

def configure_injection(binder: Binder) -> None:
    """Configure dependency injection for the application."""
    app_module = AppModule()
    app_module.configure(binder)
    
    # Bind components
    binder.bind(LLMComponent, to=app_module.provide_llm_component)
    binder.bind(VectorStoreComponent, to=app_module.provide_vector_store_component)
    binder.bind(EmbeddingComponent, to=app_module.provide_embedding_component)
    
    # Register RAGEngine
    from rag_bench.core.engine import RAGEngine
    binder.bind(RAGEngine, to=provide_rag_engine, scope=singleton)