"""
Application settings and configuration models.

Settings are loaded lazily on first access to avoid import-time side effects.
"""
from functools import lru_cache
from typing import Literal, Optional, List, Dict

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerSettings(BaseModel):
    """Server configuration settings."""
    env_name: str = Field(
        "development",
        description="Environment name (production, development, test)"
    )
    port: int = Field(8000, description="Port of the FastAPI Server")


class LLMSettings(BaseModel):
    """Configuration for the LLM provider."""
    mode: Literal["openai", "anthropic", "local", "mock"] = Field(
        "openai",
        description="LLM provider to use"
    )


class EmbeddingSettings(BaseModel):
    """Configuration for the embedding provider."""
    mode: Literal["openai", "huggingface", "mock"] = Field(
        "openai",
        description="Embedding provider to use"
    )


class VectorStoreSettings(BaseModel):
    """Configuration for the vector store."""
    mode: Literal["pgvector", "qdrant", "mock"] = Field(
        "pgvector",
        description="Vector store to use"
    )


class PostgresSettings(BaseModel):
    """PostgreSQL database connection settings."""
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    user: str = Field("postgres", description="Database user")
    password: str = Field("postgres", description="Database password")
    database: str = Field("postgres", description="Database name")
    schema_name: str = Field("public", description="Schema name")

    def to_uri(self) -> str:
        """Convert settings to a PostgreSQL connection URI string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class QdrantSettings(BaseModel):
    """Qdrant vector database settings."""
    collection_name: str = Field("documents", description="Collection name")
    url: Optional[str] = Field(None, description="URL of the Qdrant service")
    port: int = Field(6333, description="REST API port")
    host: str = Field("localhost", description="Host name")
    api_key: Optional[str] = Field(None, description="API key for Qdrant Cloud")
    prefer_grpc: bool = Field(False, description="Use gRPC when possible")


class RagSettings(BaseModel):
    """Settings for the RAG pipeline."""
    similarity_top_k: int = Field(3, description="Number of documents to retrieve")
    similarity_threshold: float = Field(0.7, description="Minimum similarity score")


class QueryEnhancementSettings(BaseModel):
    """Settings for query enhancement."""
    use_stop_word_removal: bool = Field(True, description="Remove common stop words")
    use_hyponym_expansion: bool = Field(False, description="Expand queries with hyponyms")
    hyponym_map: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of terms to their hyponyms for query expansion"
    )
    use_llm_expansion: bool = Field(False, description="Use LLM for query expansion")


class OpenAISettings(BaseModel):
    """Settings for the OpenAI provider."""
    api_key: str
    model: str = Field("gpt-4o", description="Model for completion")
    embedding_model: str = Field("text-embedding-3-large", description="Model for embeddings")
    api_base: Optional[str] = Field(None, description="Base URL of OpenAI API")


class AnthropicSettings(BaseModel):
    """Settings for the Anthropic provider."""
    api_key: str
    model: str = Field("claude-3-opus-20240229", description="Model to use")
    api_base: Optional[str] = Field(None, description="Base URL of Anthropic API")


class LocalLLMSettings(BaseModel):
    """Settings for the local LLM provider."""
    model_path: str = Field("models/llama-3-8b-instruct.gguf", description="Path to GGUF model")
    context_length: int = Field(4096, description="Context window length")
    n_gpu_layers: int = Field(0, description="Layers to offload to GPU (0 for CPU)")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")


class HuggingFaceSettings(BaseModel):
    """Settings for the HuggingFace provider."""
    embedding_model: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace model for embeddings"
    )


class Settings(BaseSettings):
    """Main settings class containing all configuration."""
    server: ServerSettings = Field(default_factory=ServerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    query_enhancement: QueryEnhancementSettings = Field(default_factory=QueryEnhancementSettings)
    openai: Optional[OpenAISettings] = None
    anthropic: Optional[AnthropicSettings] = None
    local_llm: Optional[LocalLLMSettings] = None
    huggingface: Optional[HuggingFaceSettings] = None
    qdrant: Optional[QdrantSettings] = None
    pgvector: Optional[PostgresSettings] = None
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache settings on first access.

    Uses lazy loading to avoid import-time side effects.
    Settings are cached after first load.
    """
    from rag_bench.settings.settings_loader import load_active_settings

    raw_settings = load_active_settings()
    return Settings.model_validate(raw_settings)


# Backwards compatibility alias
def settings() -> Settings:
    """Get the current settings. Alias for get_settings()."""
    return get_settings()
