# RAG Bench

A modular, extensible framework for building and benchmarking Retrieval Augmented Generation (RAG) systems.

## Overview

RAG Bench provides a flexible foundation for:

1. Building customizable RAG pipelines with interchangeable components
2. Benchmarking different RAG configurations and strategies 
3. Evaluating performance across a range of metrics
4. Experimenting with advanced techniques like query enhancement and reranking

The system is designed to be extended for domain-specific applications while maintaining a consistent architecture.

## Template Structure

This template provides a complete RAG system with all essential components implemented:

```
rag_bench/
├── components/               # Core component implementations
│   ├── embedding_component.py
│   ├── llm_component.py
│   └── vector_store_component.py
├── core/                     # Core system logic and types
│   ├── document_processors.py
│   ├── engine.py
│   ├── query_enhancers.py
│   └── types.py
├── db/                       # Database connections and models
│   └── base.py
├── dependency_injection.py   # Dependency injection configuration
├── evaluation/               # Benchmarking and evaluation framework
│   ├── benchmark.py
│   ├── metrics.py
│   ├── run_benchmark.py
│   └── sample_data/
├── main.py                   # Application entry point
├── models/                   # Data models
│   └── document.py
├── routers/                  # API routes
│   └── api_v1/
├── settings/                 # Configuration and settings
│   ├── settings.py
│   └── settings_loader.py
└── workflows/                # Document processing workflows
    └── ingest.py
```

## Key Features

- **Modular Architecture**: Swap components without changing the core system
- **Comprehensive Evaluation**: Measure retrieval quality, answer correctness, latency, and more
- **Multiple Strategies**: Compare different retrieval, processing, and generation approaches
- **Benchmarking Framework**: Run standardized tests across configurations
- **Extension Points**: Add custom implementations for specific use cases

## Components

RAG Bench includes the following core components:

### Document Ingestion
- **Document Ingestor**: Processes and chunks documents with metadata preservation
- **Text Splitter**: Divides documents into appropriately sized chunks with overlap
- **Metadata Management**: Preserves document provenance and relationships

### Retrieval & Processing
- **Embedding Components**: Generate vector representations of text using configurable models
- **Vector Store**: Efficiently store and retrieve embeddings using PGVector
- **Query Enhancers**: Multiple strategies to improve query effectiveness:
  - LLM-based query expansion for broader semantic coverage
  - Hyponym expansion for adding related terms
  - Stop word removal for focusing on meaningful terms
  - Hybrid approaches that combine multiple techniques
- **Document Processors**: Advanced filtering and reranking:
  - Threshold filtering based on similarity scores
  - Semantic reranking using embedding models
  - LLM-based reranking for nuanced relevance assessments
  - Diversity reranking to reduce redundancy in results

### Generation & Evaluation
- **LLM Integration**: Flexible integration with multiple LLM providers
- **RAG Engine**: Core orchestration of the retrieval and generation process
- **Evaluation Framework**: Comprehensive benchmarking system:
  - Metrics collection for retrieval quality and generation performance
  - LLM-based evaluation of answer correctness, completeness, and helpfulness
  - Precision/recall calculation against known relevant documents
  - Comparative reporting across different configurations

## Getting Started

### Prerequisites

- Python 3.10+ (3.12 recommended for best compatibility)
- PostgreSQL with pgvector extension (automatically detected by setup script)
- Poetry for dependency management

> **Note**: The `setup_all.py` script will attempt to detect your PostgreSQL installation automatically and fall back to mock mode if not found.

### Installation

#### 1. Set up Python 3.12 with pyenv (recommended)

```bash
# Install pyenv (macOS with Homebrew)
brew update
brew install pyenv

# For zsh (default on newer macOS)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# For bash
# echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
# echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
# echo 'eval "$(pyenv init -)"' >> ~/.bash_profile

# Restart your terminal or source the configuration
source ~/.zshrc  # or source ~/.bash_profile for bash

# Install Python 3.12
pyenv install 3.12
```

#### 2. Clone the repository and set up the environment

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-bench.git
cd rag-bench

# Set Python 3.12 as the version for this directory
pyenv local 3.12

# Install Poetry if you don't have it
# curl -sSL https://install.python-poetry.org | python3 -

# Configure Poetry to use Python 3.12
poetry env use $(pyenv which python)

# Install dependencies
poetry install
```

#### 3. Set up PostgreSQL with pgvector

##### MacOS (Homebrew)

```bash
# Install PostgreSQL
brew install postgresql@16

# Start PostgreSQL service
brew services start postgresql@16

# Add PostgreSQL to your PATH (one-time setup)
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
# For permanent setup, add this line to your .zshrc or .bash_profile

# Create database
createdb rag_bench

# Install pgvector
brew install pgvector

# For PostgreSQL 16, you need to install pgvector from source
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

# Enable the pgvector extension
psql -d rag_bench -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

##### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib build-essential postgresql-server-dev-all

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Switch to postgres user
sudo -u postgres psql

# Create a database (run inside psql)
CREATE DATABASE rag_bench;
\q

# Install pgvector from source
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable pgvector extension
sudo -u postgres psql -d rag_bench -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### 4. Configure Database Connection

Update the `settings.yaml` file with your PostgreSQL credentials:

```yaml
# PostgreSQL settings
postgres:
  host: localhost
  port: 5432
  user: yourusername  # Change to your system username
  password: ""  # Set password if required
  database: rag_bench
  schema_name: public

pgvector:
  host: localhost
  port: 5432
  user: yourusername  # Change to your system username
  password: ""  # Set password if required
  database: rag_bench
  schema_name: public
```

#### 5. Authenticate with Hugging Face (Optional)

The initialization script will download a small, freely available LLM model by default (TinyLlama 1.1B). However, if you want to use larger models or restricted models like Llama-3, you'll need to authenticate with Hugging Face:

1. Create a Hugging Face account at https://huggingface.co/join if you don't already have one

2. Generate an access token:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "rag-bench-access")
   - Select "read" access
   - Click "Generate token"
   - Copy the token

3. Login using the Hugging Face CLI:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Login with your token
huggingface-cli login
# Or alternative method:
python -c "from huggingface_hub import login; login()"
```

Enter your token when prompted. This will save your credentials locally.

4. View available models and specify one to download:

```bash
# List all available models
poetry run python initialize_models.py --list-models

# Download a specific model
poetry run python initialize_models.py --model phi-2.Q4_K_M.gguf
```

Available models include:
- TinyLlama 1.1B (700MB) - Default, fastest but least capable
- Phi-2 (1.3GB) - Good balance of size and quality
- Orca-2 7B (3.8GB) - Higher quality, requires more RAM
- Llama-3 8B (4.6GB) - Highest quality, requires special access approval

#### 6. Quick Setup

For the easiest setup experience, we provide a unified setup script that handles everything:

```bash
# Run the all-in-one setup script
poetry run python setup_all.py
```

This script will:
1. Check PostgreSQL availability (automatically finds Homebrew installations)
2. Create the database and enable pgvector extension
3. Download a small LLM model for local inference (TinyLlama 1.1B)
4. **Drop existing vector tables** to prevent dimension mismatch errors
5. Ingest sample documents
6. Provide instructions for testing the system

The script is designed to be robust and will:
- Automatically detect PostgreSQL installations in standard locations
- Fall back to mock mode if PostgreSQL isn't found
- Fall back to mock mode if model download fails
- Handle clean reinstalls by dropping and recreating tables
- Work with minimal user intervention

> **Note**: If you prefer to perform these steps manually, you can run each component separately:

```bash
# Download LLM model
poetry run python initialize_models.py

# See available models
poetry run python initialize_models.py --list-models

# Ingest sample documents
poetry run python ingest_docs.py
```

> **Note**: LLM models are large files (4-6GB). Please ensure you have sufficient disk space.

#### 7. Troubleshooting

If you encounter any of these issues:

**Vector Dimension Mismatch Error**
```
DimensionError: Cannot insert 768-dimensional vector into 1536-dimensional column
```
Solution: Re-run `setup_all.py` which will drop existing tables with mismatched dimensions.

**PostgreSQL Not Found**
If PostgreSQL isn't found in the standard paths, the script will automatically fall back to mock mode, which still allows you to test the system functionality without a database.

**LLM Model Errors**
If model downloads fail due to Hugging Face rate limits or network issues, the system will fall back to mock LLM mode.

### Configuration

The system can run in different modes:

1. **Fully Local Mode** - All components run locally (no API keys needed)
2. **Hybrid Mode** - Some components use local resources, others use cloud APIs
3. **Cloud Mode** - All components use cloud APIs (requires API keys)

Edit `settings.yaml` to configure your preferred mode:

#### Fully Local Mode

This mode uses local LLM inference, local embeddings, and PostgreSQL:

```yaml
# Local mode (no API keys required)
llm:
  mode: local
embedding:
  mode: huggingface
vectorstore:
  mode: pgvector

# Local LLM settings
local_llm:
  model_path: models/llama-3-8b-instruct.gguf
  context_length: 4096
  n_gpu_layers: 0  # Increase for GPU acceleration
  max_tokens: 1024
  temperature: 0.7

# HuggingFace embedding settings  
huggingface:
  embedding_model: sentence-transformers/all-mpnet-base-v2

# PostgreSQL connection settings
pgvector:
  host: localhost
  port: 5432
  user: yourusername  # Change this to your system username
  password: ""  # Update with your password if needed
  database: rag_bench
```

#### Cloud Mode (OpenAI)

This mode uses OpenAI for LLM inference and embeddings:

```yaml
# Cloud mode (API keys required)
llm:
  mode: openai
embedding:
  mode: openai
vectorstore:
  mode: pgvector  # Still uses local PostgreSQL

# OpenAI settings
openai:
  api_key: ${OPENAI_API_KEY}  # Set this environment variable
  model: gpt-4o
  embedding_model: text-embedding-3-large

# PostgreSQL connection settings
pgvector:
  host: localhost
  port: 5432
  user: yourusername  # Change this to your system username
  password: ""  # Update with your password if needed
  database: rag_bench
```

#### Hybrid Mode (Local LLM + OpenAI Embeddings)

This mode combines local LLM with OpenAI embeddings:

```yaml
# Hybrid mode
llm:
  mode: local
embedding:
  mode: openai  # Uses OpenAI for embeddings
vectorstore:
  mode: pgvector

# Local LLM settings
local_llm:
  model_path: models/llama-3-8b-instruct.gguf
  context_length: 4096
  n_gpu_layers: 0
  
# OpenAI settings (only for embeddings)
openai:
  api_key: ${OPENAI_API_KEY}
  embedding_model: text-embedding-3-large
```

### Installation and Setup

For the quickest setup:

```bash
# Run the all-in-one setup script
poetry run python setup_all.py
```

This script handles:
1. PostgreSQL detection and initialization
2. Database and pgvector extension setup
3. Downloading LLM models
4. Creating tables and schema
5. Ingesting sample documents

If you encounter any issues, the script will provide troubleshooting guidance.

### Running the Server

After setup is complete, start the server:

```bash
# Start the server
poetry run python -m rag_bench.main
```

The server will run at http://localhost:8000 by default.

### Testing the Installation

After starting the server, verify it's working correctly using the test script:

```bash
# Make the script executable
chmod +x simple_test.sh

# Run the test script
./simple_test.sh
```

This script makes basic curl requests to test if the server is responding properly with two test queries:
1. "What is RAG?"
2. "What are the key components of a RAG system?"

The response should include an answer and sources with relevance scores. If you see responses formatted as JSON, the server is functioning correctly.

Example output:
```json
{
  "answer": "RAG is a technique that enhances LLM outputs with external knowledge.",
  "sources": [
    {
      "source": "sample",
      "relevance_score": 0.7988745719194412,
      "title": "RAG Introduction"
    },
    {
      "source": "sample",
      "relevance_score": 0.8848782032728195,
      "title": "LLMs"
    },
    {
      "source": "sample",
      "relevance_score": 0.9188621118664742,
      "title": "Embeddings"
    }
  ]
}
```

### Ingesting Documents

Before making queries, you need to ingest documents into the vector database. Here's a basic example:

```python
# Example script to ingest documents (save as ingest_docs.py)
import asyncio
from langchain.schema import Document as LangchainDocument
from rag_bench.settings.settings import Settings
from rag_bench.settings.settings_loader import load_settings
from rag_bench.dependency_injection import get_vector_store_component

async def ingest_sample_documents():
    # Load settings
    settings_dict = load_settings("settings.yaml")
    settings = Settings.model_validate(settings_dict)
    
    # Create vector store component
    vector_store = get_vector_store_component(settings)
    
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
        )
    ]
    
    # Add documents to vector store
    await vector_store.aadd_documents(documents)
    print(f"Ingested {len(documents)} documents")

if __name__ == "__main__":
    asyncio.run(ingest_sample_documents())
```

Run the script to ingest the sample documents:

```bash
poetry run python ingest_docs.py
```

### Testing the System

Once documents are ingested and the server is running, you can test the system:

1. **Using the API endpoint**:
   ```bash
   curl "http://localhost:8000/api/v1/query?q=What%20is%20RAG"
   ```

2. **Using a web browser**:
   - Open http://localhost:8000/api/v1/query?q=What%20is%20RAG in your browser

You should receive a response that includes information retrieved from the documents along with sources.

## Benchmarking

RAG Bench includes a comprehensive benchmarking framework for systematic evaluation of different configurations:

```bash
# Run a benchmark with different configurations
poetry run python -m rag_bench.evaluation.run_benchmark --config rag_bench/evaluation/sample_data/benchmark_config.json

# Or use the convenience script
./benchmark.sh
```

The benchmark will:
1. Compare different RAG configurations (baseline, query expansion, reranking, etc.)
2. Run each query from the evaluation set through each configuration
3. Generate metrics for retrieval quality, answer quality, and performance
4. Output detailed results to the `benchmark_results` directory

### Benchmark Configuration

The benchmarking framework allows comparing multiple system configurations:

```json
{
  "name": "rag_benchmark_basic",
  "description": "Basic RAG benchmark comparing different configurations",
  "evaluation_set_path": "path/to/evaluation_set.json",
  "output_dir": "benchmark_results",
  "use_llm_evaluation": true,
  "num_iterations": 1,
  "configurations": [
    {
      "name": "baseline",
      "similarity_top_k": 3,
      "similarity_threshold": 0.7,
      "use_reranking": false,
      "use_query_expansion": false
    },
    {
      "name": "with_reranking",
      "similarity_top_k": 5,
      "similarity_threshold": 0.5,
      "use_reranking": true,
      "reranker_type": "semantic",
      "use_query_expansion": false
    },
    {
      "name": "full_pipeline",
      "similarity_top_k": 5,
      "similarity_threshold": 0.5,
      "use_reranking": true,
      "reranker_type": "hybrid",
      "use_query_expansion": true,
      "query_expansion_type": "llm"
    }
  ]
}
```

### Evaluation Metrics

The system collects and compares multiple metrics across configurations:

- **Retrieval Performance**: Precision, recall, document scores
- **Runtime Performance**: Total time, retrieval time, generation time
- **Answer Quality**: Correctness, completeness, conciseness, groundedness
- **Resource Usage**: Number of documents retrieved and used

### Output Reports

Benchmarks generate several output files for analysis:

- **Summary Reports**: High-level comparison of configurations
- **Detailed CSV Data**: Complete metrics for each query
- **Comparison Charts**: Visual comparison of key metrics
- **Raw JSON Results**: Complete data for further analysis

## Extending the System

This template is designed to be extended for domain-specific applications. Here are the key extension points:

### Custom Components

Create custom implementations by extending the base classes:

#### Custom Query Enhancers

```python
from rag_bench.core.types import QueryEnhancer
from typing import Optional

class DomainSpecificEnhancer(QueryEnhancer):
    """Enhances queries with domain-specific knowledge."""
    
    async def enhance(self, query: str, conversation_id: Optional[str] = None) -> str:
        # Custom domain-specific logic here
        # Example: Add industry terminology, expand domain abbreviations, etc.
        return enhanced_query
```

#### Custom Document Processors

```python
from rag_bench.core.types import DocumentPostProcessor, DocumentWithScore
from typing import List

class DomainRelevanceProcessor(DocumentPostProcessor):
    """Processes documents based on domain-specific relevance criteria."""
    
    async def process(self, documents: List[DocumentWithScore], query: str) -> List[DocumentWithScore]:
        # Custom filtering or reranking logic
        # Example: Apply domain-specific weighting, filter by recency, etc.
        return processed_documents
```

### Custom Evaluation Sets

Create evaluation sets with queries and expected answers relevant to your domain:

```json
{
  "name": "Domain Specific Questions",
  "description": "Evaluation set for testing domain-specific knowledge",
  "queries": [
    {
      "id": "domain-001",
      "query": "What are the key requirements for X in industry Y?",
      "expected_answer": "The key requirements for X in industry Y include...",
      "relevant_doc_ids": ["doc-industry-y-1", "doc-requirements-x-1"]
    },
    {
      "id": "domain-002",
      "query": "How does process Z affect outcomes in scenario W?",
      "expected_answer": "Process Z affects outcomes in scenario W by...",
      "relevant_doc_ids": ["doc-process-z-1", "doc-scenario-w-1"]
    }
  ],
  "metadata": {
    "domain": "Your Domain",
    "version": "1.0"
  }
}
```

## API Reference

The system exposes a REST API for interacting with the RAG pipeline:

- `GET /api/v1/query?q=your_query` - Simple query endpoint
- `POST /api/v1/chat/message` - Chat interface with conversation history
- `POST /api/v1/ingest` - Add documents to the system

## Customization Guide

To adapt this template for your specific use case:

1. **Domain-Specific Data**: Add your own document ingestion pipelines in the `workflows` directory
2. **Custom Enhancers**: Implement domain-specific query enhancers for terminology, abbreviations, etc.
3. **Custom Processors**: Add specialized document processors for your content types
4. **Evaluation Sets**: Create benchmark data relevant to your domain
5. **UI Integration**: Extend the API with additional endpoints as needed

## Use Cases

This template can be adapted for various RAG applications:

- **Customer Support**: Connect to product documentation and support tickets
- **Legal Research**: Link to case law, statutes, and legal documents
- **Financial Analysis**: Connect to financial reports, news, and market data
- **Medical Information**: Adapt for connecting to medical literature and clinical guidelines
- **Technical Documentation**: Build a system for software documentation and code examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.
