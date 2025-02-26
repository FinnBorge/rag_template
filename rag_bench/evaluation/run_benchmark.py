#!/usr/bin/env python
"""
Script to run RAG benchmarks with various configurations.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any

from rag_bench.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner
from rag_bench.dependency_injection import configure_injection
from rag_bench.settings.settings import Settings
from rag_bench.settings.settings_loader import load_settings
from rag_bench.core.engine import RAGEngine
from rag_bench.core.types import LLMComponent
from rag_bench.core.document_processors import (
    ThresholdFilter, 
    SemanticReranker, 
    LLMReranker,
    DiversityReranker,
    ProcessingPipeline
)
from rag_bench.core.query_enhancers import (
    HyponymExpansionEnhancer,
    LLMQueryExpansionEnhancer,
    HybridQueryEnhancer,
    StopWordRemovalEnhancer
)

from injector import Injector, singleton

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)

logger = logging.getLogger(__name__)


def configure_injector(settings_path: str) -> Injector:
    """
    Configure the dependency injector with the provided settings.
    
    Args:
        settings_path: Path to the settings file
        
    Returns:
        Configured injector
    """
    # Load settings
    settings = load_settings(settings_path)
    
    # Create injector
    injector = Injector([
        lambda binder: binder.bind(Settings, to=settings, scope=singleton),
        configure_injection
    ])
    
    return injector


def _apply_configuration(
    rag_engine: RAGEngine,
    configuration: Dict[str, Any],
    llm_component: LLMComponent
):
    """
    Apply a benchmark configuration to the RAG engine.
    
    Args:
        rag_engine: The RAG engine to configure
        configuration: Configuration dictionary
        llm_component: LLM component for use in enhancers
    """
    # Update similarity search parameters
    if "similarity_top_k" in configuration:
        rag_engine.settings.rag.similarity_top_k = configuration["similarity_top_k"]
    
    if "similarity_threshold" in configuration:
        rag_engine.settings.rag.similarity_threshold = configuration["similarity_threshold"]
    
    # Configure query enhancers
    query_enhancers = []
    
    # Always add stop word removal as a basic enhancer
    query_enhancers.append(StopWordRemovalEnhancer())
    
    # Add query expansion if configured
    if configuration.get("use_query_expansion", False):
        expansion_type = configuration.get("query_expansion_type", "llm")
        
        if expansion_type == "llm":
            query_enhancers.append(LLMQueryExpansionEnhancer(llm_component))
        elif expansion_type == "hyponym":
            # Example hyponym map - in a real implementation, this would be loaded from a data file
            hyponym_map = {
                "document": ["text", "paper", "file", "content"],
                "model": ["algorithm", "system", "framework"],
                "search": ["retrieval", "query", "lookup", "find"],
            }
            query_enhancers.append(HyponymExpansionEnhancer(hyponym_map))
        elif expansion_type == "hybrid":
            # Create hybrid enhancer with both methods
            hyponym_map = {
                "document": ["text", "paper", "file", "content"],
                "model": ["algorithm", "system", "framework"],
                "search": ["retrieval", "query", "lookup", "find"],
            }
            sub_enhancers = [
                HyponymExpansionEnhancer(hyponym_map),
                LLMQueryExpansionEnhancer(llm_component)
            ]
            query_enhancers.append(HybridQueryEnhancer(sub_enhancers))
    
    # Set query enhancers
    rag_engine.query_enhancers = query_enhancers
    
    # Configure document post-processors
    document_processors = []
    
    # Add threshold filter
    document_processors.append(
        ThresholdFilter(threshold=configuration.get("similarity_threshold", 0.7))
    )
    
    # Add reranking if configured
    if configuration.get("use_reranking", False):
        reranker_type = configuration.get("reranker_type", "semantic")
        
        if reranker_type == "semantic":
            document_processors.append(
                SemanticReranker(rag_engine.enhancer_component)
            )
        elif reranker_type == "llm":
            document_processors.append(
                LLMReranker(llm_component)
            )
        elif reranker_type == "diversity":
            document_processors.append(
                DiversityReranker(rag_engine.enhancer_component, diversity_weight=0.3)
            )
        elif reranker_type == "hybrid":
            # Use multiple rerankers in sequence
            document_processors.append(
                SemanticReranker(rag_engine.enhancer_component)
            )
            document_processors.append(
                DiversityReranker(rag_engine.enhancer_component, diversity_weight=0.3)
            )
    
    # Set document processors
    rag_engine.document_post_processors = document_processors


class ConfigurableBenchmarkRunner(BenchmarkRunner):
    """Extended benchmark runner that can apply specific configurations."""
    
    def _apply_configuration(self, configuration: Dict[str, Any]):
        """Apply configuration to the RAG engine."""
        _apply_configuration(self.rag_engine, configuration, self.llm_component)
        logger.info(f"Applied configuration: {configuration.get('name', 'unnamed')}")


async def run_benchmark(config_path: str, settings_path: str):
    """
    Run a benchmark with the specified configuration.
    
    Args:
        config_path: Path to the benchmark configuration file
        settings_path: Path to the settings file
    """
    try:
        # Load benchmark configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        benchmark_config = BenchmarkConfig.parse_obj(config_data)
        logger.info(f"Loaded benchmark configuration: {benchmark_config.name}")
        
        # Configure injector
        injector = configure_injector(settings_path)
        logger.info("Configured dependency injection")
        
        # Get components
        rag_engine = injector.get(RAGEngine)
        llm_component = injector.get(LLMComponent)
        
        # Create benchmark runner
        runner = ConfigurableBenchmarkRunner(
            config=benchmark_config,
            rag_engine=rag_engine,
            llm_component=llm_component
        )
        
        # Run benchmark
        logger.info(f"Starting benchmark: {benchmark_config.name}")
        results = await runner.run_benchmark()
        logger.info(f"Benchmark completed: {benchmark_config.name}")
        
        # Return results
        return results
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="RAG Benchmarking Tool")
    parser.add_argument(
        "--config", 
        default="rag_bench/evaluation/sample_data/benchmark_config.json",
        help="Path to benchmark configuration file"
    )
    parser.add_argument(
        "--settings", 
        default="settings.yaml",
        help="Path to settings file"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    os.makedirs(config_data.get("output_dir", "benchmark_results"), exist_ok=True)
    
    # Run benchmark
    asyncio.run(run_benchmark(args.config, args.settings))


if __name__ == "__main__":
    main()