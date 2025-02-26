from typing import Literal, Optional, Any, List
import re
import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerSettings(BaseModel):
    """Server configuration settings."""
    env_name: str = Field("development", description="Name of the environment (production, development, test)")
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
    host: str = Field(
        "localhost",
        description="The server hosting the Postgres database"
    )
    port: int = Field(
        5432,
        description="The port on which the Postgres database is accessible"
    )
    user: str = Field(
        "postgres",
        description="The user to use to connect to the Postgres database"
    )
    password: str = Field(
        "postgres",
        description="The password to use to connect to the Postgres database"
    )
    database: str = Field(
        "postgres",
        description="The database to use to connect to the Postgres database"
    )
    schema_name: str = Field(
        "public",
        description="The name of the schema in the Postgres database to use"
    )

    def to_uri(self) -> str:
        """Convert settings to a PostgreSQL connection URI string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class QdrantSettings(BaseModel):
    """Qdrant vector database settings."""
    collection_name: str = Field(
        "documents",
        description="Collection to find the data"
    )
    url: Optional[str] = Field(
        None,
        description="URL of the Qdrant service"
    )
    port: int = Field(
        6333, 
        description="Port of the REST API interface"
    )
    host: str = Field(
        "localhost",
        description="Host name of Qdrant service"
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for authentication in Qdrant Cloud"
    )
    prefer_grpc: bool = Field(
        False,
        description="If true - use gRPC interface whenever possible"
    )


class RagSettings(BaseModel):
    """Settings for the RAG pipeline."""
    similarity_top_k: int = Field(
        3,
        description="The number of documents returned by the RAG pipeline"
    )
    similarity_threshold: Optional[float] = Field(
        None,
        description="If set, documents retrieved from RAG must meet a certain match score"
    )


class OpenAISettings(BaseModel):
    """Settings for the OpenAI provider."""
    api_key: str
    model: str = Field(
        "gpt-4o",
        description="OpenAI Model to use for completion. Example: 'gpt-4o'."
    )
    embedding_model: str = Field(
        "text-embedding-3-large",
        description="OpenAI Model to use for embeddings. Example: 'text-embedding-3-large'."
    )
    api_base: Optional[str] = Field(
        None,
        description="Base URL of OpenAI API. Example: 'https://api.openai.com/v1'."
    )


class AnthropicSettings(BaseModel):
    """Settings for the Anthropic provider."""
    api_key: str
    model: str = Field(
        "claude-3-opus-20240229",
        description="Anthropic model to use. Example: 'claude-3-opus-20240229'."
    )
    api_base: Optional[str] = Field(
        None,
        description="Base URL of Anthropic API."
    )


class LocalLLMSettings(BaseModel):
    """Settings for the local LLM provider."""
    model_path: str = Field(
        "models/llama-3-8b-instruct.gguf",
        description="Path to the GGUF model file"
    )
    context_length: int = Field(
        4096,
        description="Context window length"
    )
    n_gpu_layers: int = Field(
        0,
        description="Number of layers to offload to GPU (0 for CPU-only)"
    )
    max_tokens: int = Field(
        1024,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        0.7,
        description="Sampling temperature"
    )
    stop_sequences: Optional[List[str]] = Field(
        None,
        description="Sequences that will stop generation"
    )


class HuggingFaceSettings(BaseModel):
    """Settings for the HuggingFace provider."""
    embedding_model: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace model to use for embeddings"
    )


class Settings(BaseSettings):
    """Main settings class that contains all configuration."""
    server: ServerSettings = Field(default_factory=ServerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    openai: Optional[OpenAISettings] = None
    anthropic: Optional[AnthropicSettings] = None
    local_llm: Optional[LocalLLMSettings] = None
    huggingface: Optional[HuggingFaceSettings] = None
    qdrant: Optional[QdrantSettings] = None
    pgvector: Optional[PostgresSettings] = None
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)


# Import here to avoid circular imports
from rag_bench.settings.settings_loader import load_active_settings

# Load settings from files with environment variable support
unsafe_settings = load_active_settings()
unsafe_typed_settings = Settings.model_validate(unsafe_settings)


def settings() -> Settings:
    """Get the current loaded settings.
    
    For regular components, dependency injection is preferred.
    """
    # In a real implementation, this would use a dependency injection container
    return unsafe_typed_settings