#!/usr/bin/env python3
"""
Complete setup script for RAG Bench.

This script:
1. Downloads a local LLM model
2. Sets up the PostgreSQL database
3. Ingests sample documents
4. Validates the configuration

Usage:
    python setup_all.py
"""
import os
import sys
import logging
import subprocess
import asyncio
import yaml
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if running inside Poetry environment
def in_poetry_env() -> bool:
    """Check if script is running inside Poetry environment."""
    return "POETRY_ACTIVE" in os.environ or "VIRTUAL_ENV" in os.environ

def run_command(command: List[str], description: str) -> bool:
    """Run a command and return True if successful."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

async def main():
    """Run the complete setup process."""
    if not in_poetry_env():
        logger.error("This script should be run inside a Poetry environment.")
        logger.error("Run: poetry run python setup_all.py")
        sys.exit(1)
    
    # Step 1: Check if PostgreSQL is available
    logger.info("Checking PostgreSQL availability...")
    
    # Try different possible paths for psql (including Homebrew locations)
    psql_paths = [
        "psql",  # Default if in PATH
        "/opt/homebrew/opt/postgresql@16/bin/psql",  # Homebrew on Apple Silicon
        "/opt/homebrew/opt/postgresql@15/bin/psql",  # Older PostgreSQL on Apple Silicon
        "/opt/homebrew/bin/psql",  # Homebrew generic path
        "/usr/local/opt/postgresql@16/bin/psql",  # Homebrew on Intel Mac
        "/usr/local/opt/postgresql@15/bin/psql",  # Older PostgreSQL on Intel Mac
        "/usr/local/bin/psql"  # Homebrew generic path on Intel
    ]
    
    postgres_available = False
    psql_path = None
    
    for path in psql_paths:
        try:
            result = subprocess.run([path, "--version"], check=True, capture_output=True, text=True)
            postgres_available = True
            psql_path = path
            logger.info(f"Found PostgreSQL at: {path}")
            logger.info(f"Version: {result.stdout.strip()}")
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    if postgres_available:
        # Get the base directory of psql for other commands
        pg_bin_dir = str(Path(psql_path).parent)
        
        # Create database if it doesn't exist
        create_db_cmd = ["createdb", "rag_bench"]
        create_db_result = run_command(create_db_cmd, "Create rag_bench database")
        if not create_db_result:
            logger.info("Database might already exist, continuing...")
        
        # Add pgvector extension
        add_extension_cmd = [psql_path, "rag_bench", "-c", "CREATE EXTENSION IF NOT EXISTS vector;"]
        run_command(add_extension_cmd, "Add pgvector extension")
        
        # Drop existing collections to avoid dimension mismatch
        drop_tables_cmd = [
            psql_path, "rag_bench", "-c", 
            "DROP TABLE IF EXISTS langchain_pg_embedding CASCADE; DROP TABLE IF EXISTS langchain_pg_collection CASCADE;"
        ]
        run_command(drop_tables_cmd, "Drop existing vector database tables")
    else:
        logger.warning("PostgreSQL not found. Will use mock vector store.")
        # Update settings to use mock mode
        settings_path = Path("settings.yaml")
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    settings = yaml.safe_load(f)
                
                # Update settings to use mock mode
                settings["vectorstore"]["mode"] = "mock"
                
                with open(settings_path, "w") as f:
                    yaml.dump(settings, f, sort_keys=False)
                
                logger.info("Updated settings.yaml to use mock vector store mode")
            except Exception as e:
                logger.error(f"Failed to update settings: {e}")
    
    # Step 2: Download LLM model
    logger.info("Setting up LLM model...")
    model_setup_result = run_command(["python", "initialize_models.py"], "Download LLM model")
    
    if not model_setup_result:
        logger.warning("Failed to download LLM model. Switching to mock LLM mode.")
        settings_path = Path("settings.yaml")
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    settings = yaml.safe_load(f)
                
                # Update settings to use mock LLM and embedding modes
                settings["llm"]["mode"] = "mock"
                settings["embedding"]["mode"] = "mock"
                
                with open(settings_path, "w") as f:
                    yaml.dump(settings, f, sort_keys=False)
                
                logger.info("Updated settings.yaml to use mock LLM and embedding modes")
            except Exception as e:
                logger.error(f"Failed to update settings: {e}")
    
    # Step 3: Ingest sample documents
    logger.info("Ingesting sample documents...")
    try:
        # Import and run ingestion script
        import ingest_docs
        await ingest_docs.ingest_sample_documents()
        logger.info("Sample documents ingested successfully")
    except Exception as e:
        logger.error(f"Failed to ingest sample documents: {e}")
    
    # Step 4: Final instructions
    logger.info("\n" + "="*80)
    logger.info("RAG Bench setup complete!")
    logger.info("="*80)
    logger.info("\nYou can now run the server with:")
    logger.info("  poetry run python -m rag_bench.main")
    logger.info("\nThen test it with:")
    logger.info("  curl \"http://localhost:8000/api/v1/query?q=What%20is%20RAG\"")
    logger.info("Or visit in your browser:")
    logger.info("  http://localhost:8000/api/v1/query?q=What%20is%20RAG")
    logger.info("\nHappy experimenting with RAG!")

if __name__ == "__main__":
    asyncio.run(main())