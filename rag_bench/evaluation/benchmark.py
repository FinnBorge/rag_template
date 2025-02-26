"""
Benchmark framework for evaluating RAG systems.
"""
import asyncio
import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from rag_bench.core.types import LLMComponent
from rag_bench.core.engine import RAGEngine, SourcedAnswer
from rag_bench.core.types import DocumentWithScore
from rag_bench.evaluation.metrics import (
    QueryMetrics, 
    EvaluationSet, 
    EvaluationResults, 
    LLMBasedEvaluator, 
    RetrievalEvaluator,
    MetricsCollector
)

logger = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""
    name: str
    description: str = ""
    evaluation_set_path: str
    output_dir: str = "benchmark_results"
    use_llm_evaluation: bool = True
    num_iterations: int = 1
    configurations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunner:
    """Runs benchmarks for evaluating RAG systems."""
    
    def __init__(
        self, 
        config: BenchmarkConfig,
        rag_engine: RAGEngine,
        llm_component: Optional[LLMComponent] = None
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Benchmark configuration
            rag_engine: The RAG engine to benchmark
            llm_component: LLM component for evaluations (if use_llm_evaluation is True)
        """
        self.config = config
        self.rag_engine = rag_engine
        self.llm_component = llm_component
        
        if config.use_llm_evaluation and not llm_component:
            raise ValueError("LLM component is required for LLM-based evaluation")
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator()
        self.llm_evaluator = LLMBasedEvaluator(llm_component) if llm_component else None
    
    async def load_evaluation_set(self) -> EvaluationSet:
        """Load the evaluation set from the specified path."""
        try:
            with open(self.config.evaluation_set_path, 'r') as f:
                data = json.load(f)
            
            return EvaluationSet.parse_obj(data)
        except Exception as e:
            logger.error(f"Error loading evaluation set: {str(e)}")
            raise
    
    async def _run_single_query(
        self, 
        query_data: Dict[str, Any],
        metrics_collector: MetricsCollector
    ) -> Tuple[SourcedAnswer, QueryMetrics]:
        """
        Run a single query and collect metrics.
        
        Args:
            query_data: Query data from the evaluation set
            metrics_collector: Metrics collector
            
        Returns:
            Tuple of (answer, metrics)
        """
        query_id = query_data.get("id", str(uuid.uuid4()))
        query = query_data["query"]
        expected_answer = query_data.get("expected_answer", "")
        relevant_doc_ids = query_data.get("relevant_doc_ids", [])
        
        # Start tracking the query
        metrics_collector.start_query(query_id, query)
        
        try:
            # Patch the RAG engine to collect metrics
            original_retrieve = self.rag_engine._retrieve_documents
            original_post_process = self.rag_engine._post_process_documents
            original_generate = self.rag_engine._generate_answer
            
            # Patched methods to collect metrics
            async def patched_retrieve(query):
                metrics_collector.start_retrieval()
                docs = await original_retrieve(query)
                metrics_collector.end_retrieval(docs)
                return docs
            
            async def patched_post_process(docs, query):
                processed_docs = await original_post_process(docs, query)
                return processed_docs
            
            async def patched_generate(query, docs):
                metrics_collector.start_generation()
                answer = await original_generate(query, docs)
                metrics_collector.end_generation(docs)
                return answer
            
            # Apply the patches
            self.rag_engine._retrieve_documents = patched_retrieve
            self.rag_engine._post_process_documents = patched_post_process
            self.rag_engine._generate_answer = patched_generate
            
            # Generate answer
            answer = await self.rag_engine.generate_answer(query)
            
            # Restore original methods
            self.rag_engine._retrieve_documents = original_retrieve
            self.rag_engine._post_process_documents = original_post_process
            self.rag_engine._generate_answer = original_generate
            
            # Get metrics
            metrics = metrics_collector.get_metrics()
            
            # Add retrieval precision and recall if relevant_doc_ids are provided
            if relevant_doc_ids:
                precision_recall = self.retrieval_evaluator.calculate_precision_recall(
                    metrics_collector.retrieved_docs,
                    relevant_doc_ids
                )
                metrics.retrieval_precision = precision_recall["precision"]
                metrics.retrieval_recall = precision_recall["recall"]
            
            # Add LLM-based evaluation if enabled
            if self.config.use_llm_evaluation and expected_answer and self.llm_evaluator:
                eval_metrics = await self.llm_evaluator.evaluate_answer(
                    query=query,
                    actual_answer=answer.answer,
                    expected_answer=expected_answer
                )
                
                if "correctness" in eval_metrics:
                    metrics.answer_correctness = eval_metrics["correctness"]
                if "completeness" in eval_metrics:
                    metrics.answer_completeness = eval_metrics["completeness"]
                if "conciseness" in eval_metrics:
                    metrics.answer_conciseness = eval_metrics["conciseness"]
                if "groundedness" in eval_metrics:
                    metrics.answer_groundedness = eval_metrics["groundedness"]
                if "helpfulness" in eval_metrics:
                    metrics.answer_helpfulness = eval_metrics["helpfulness"]
                
                # Calculate overall quality as average of all metrics
                quality_metrics = [
                    v for k, v in eval_metrics.items() 
                    if k in ["correctness", "completeness", "conciseness", "groundedness", "helpfulness"]
                ]
                if quality_metrics:
                    metrics.answer_quality = sum(quality_metrics) / len(quality_metrics)
            
            return answer, metrics
            
        except Exception as e:
            logger.error(f"Error running query {query_id}: {str(e)}")
            metrics = metrics_collector.get_metrics()
            return None, metrics
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the benchmark and return the results.
        
        Returns:
            Dictionary of benchmark results
        """
        logger.info(f"Starting benchmark: {self.config.name}")
        
        # Load the evaluation set
        evaluation_set = await self.load_evaluation_set()
        logger.info(f"Loaded evaluation set: {evaluation_set.name} with {len(evaluation_set.queries)} queries")
        
        # Initialize results for each configuration
        all_results = {}
        
        # If no configurations are specified, run with default configuration
        configurations = self.config.configurations or [{}]
        
        for config_idx, configuration in enumerate(configurations):
            config_name = configuration.get("name", f"config_{config_idx}")
            logger.info(f"Running configuration: {config_name}")
            
            # Apply configuration to the RAG engine
            self._apply_configuration(configuration)
            
            # Initialize metrics collector
            metrics_collector = MetricsCollector()
            
            # Run each query in the evaluation set
            all_query_metrics = []
            
            for iteration in range(self.config.num_iterations):
                logger.info(f"Starting iteration {iteration+1}/{self.config.num_iterations}")
                
                for query_idx, query_data in enumerate(evaluation_set.queries):
                    logger.info(f"Running query {query_idx+1}/{len(evaluation_set.queries)}: {query_data['query'][:50]}...")
                    
                    # Run the query
                    answer, metrics = await self._run_single_query(query_data, metrics_collector)
                    
                    if answer:
                        logger.info(f"Generated answer: {answer.answer[:100]}...")
                    else:
                        logger.warning(f"Failed to generate answer for query {query_idx+1}")
                    
                    all_query_metrics.append(metrics)
            
            # Create evaluation results
            results = EvaluationResults(
                evaluation_set_name=evaluation_set.name,
                configuration=configuration,
                query_metrics=all_query_metrics
            )
            
            # Compute aggregate metrics
            results.compute_aggregates()
            
            # Save results
            self._save_results(results, config_name)
            
            # Add to all results
            all_results[config_name] = results
        
        # Generate comparison report if more than one configuration
        if len(configurations) > 1:
            self._generate_comparison_report(all_results)
        
        return all_results
    
    def _apply_configuration(self, configuration: Dict[str, Any]):
        """
        Apply a configuration to the RAG engine.
        
        Args:
            configuration: Configuration dictionary
        """
        # In a real implementation, this would modify the RAG engine components
        # based on the configuration. This is a simplified example.
        pass
    
    def _save_results(self, results: EvaluationResults, config_name: str):
        """
        Save benchmark results to files.
        
        Args:
            results: Evaluation results
            config_name: Name of the configuration
        """
        # Create a safe filename from the configuration name
        safe_name = "".join(c if c.isalnum() else "_" for c in config_name)
        base_path = os.path.join(self.config.output_dir, f"{self.config.name}_{safe_name}")
        
        # Save full results as JSON
        with open(f"{base_path}_results.json", 'w') as f:
            import json
            f.write(json.dumps(results.model_dump(), indent=2))
        
        # Save metrics as CSV
        metrics_data = []
        for metric in results.query_metrics:
            data = {
                "query_id": metric.query_id,
                "query": metric.query,
                "total_time_ms": metric.total_time_ms,
                "retrieval_time_ms": metric.retrieval_time_ms,
                "generation_time_ms": metric.generation_time_ms,
                "num_docs_retrieved": metric.num_docs_retrieved,
                "num_docs_used": metric.num_docs_used,
                "mean_doc_score": sum(metric.doc_scores) / len(metric.doc_scores) if metric.doc_scores else 0
            }
            
            # Add optional metrics if available
            if metric.answer_quality is not None:
                data["answer_quality"] = metric.answer_quality
            if metric.retrieval_precision is not None:
                data["retrieval_precision"] = metric.retrieval_precision
            if metric.retrieval_recall is not None:
                data["retrieval_recall"] = metric.retrieval_recall
            if metric.answer_correctness is not None:
                data["answer_correctness"] = metric.answer_correctness
            if metric.answer_completeness is not None:
                data["answer_completeness"] = metric.answer_completeness
            if metric.answer_conciseness is not None:
                data["answer_conciseness"] = metric.answer_conciseness
            if metric.answer_groundedness is not None:
                data["answer_groundedness"] = metric.answer_groundedness
            if metric.answer_helpfulness is not None:
                data["answer_helpfulness"] = metric.answer_helpfulness
            
            metrics_data.append(data)
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(f"{base_path}_metrics.csv", index=False)
        
        # Save aggregate metrics
        with open(f"{base_path}_summary.txt", 'w') as f:
            f.write(f"Benchmark: {self.config.name}\n")
            f.write(f"Configuration: {config_name}\n")
            f.write(f"Evaluation set: {results.evaluation_set_name}\n")
            f.write(f"Queries: {len(results.query_metrics)}\n\n")
            
            f.write("Aggregate Metrics:\n")
            f.write(f"Mean total time: {results.mean_total_time_ms:.2f} ms\n")
            f.write(f"Mean retrieval time: {results.mean_retrieval_time_ms:.2f} ms\n")
            f.write(f"Mean generation time: {results.mean_generation_time_ms:.2f} ms\n")
            f.write(f"Mean docs retrieved: {results.mean_num_docs_retrieved:.2f}\n")
            f.write(f"Mean docs used: {results.mean_num_docs_used:.2f}\n")
            f.write(f"Mean doc score: {results.mean_doc_score:.4f}\n")
            
            if results.mean_answer_quality is not None:
                f.write(f"Mean answer quality: {results.mean_answer_quality:.4f}\n")
            if results.mean_retrieval_precision is not None:
                f.write(f"Mean retrieval precision: {results.mean_retrieval_precision:.4f}\n")
            if results.mean_retrieval_recall is not None:
                f.write(f"Mean retrieval recall: {results.mean_retrieval_recall:.4f}\n")
            if results.mean_answer_correctness is not None:
                f.write(f"Mean answer correctness: {results.mean_answer_correctness:.4f}\n")
            if results.mean_answer_completeness is not None:
                f.write(f"Mean answer completeness: {results.mean_answer_completeness:.4f}\n")
            if results.mean_answer_conciseness is not None:
                f.write(f"Mean answer conciseness: {results.mean_answer_conciseness:.4f}\n")
            if results.mean_answer_groundedness is not None:
                f.write(f"Mean answer groundedness: {results.mean_answer_groundedness:.4f}\n")
            if results.mean_answer_helpfulness is not None:
                f.write(f"Mean answer helpfulness: {results.mean_answer_helpfulness:.4f}\n")
        
        logger.info(f"Results saved to {base_path}_*")
    
    def _generate_comparison_report(self, all_results: Dict[str, EvaluationResults]):
        """
        Generate a comparison report for multiple configurations.
        
        Args:
            all_results: Dictionary of results for each configuration
        """
        comparison_path = os.path.join(self.config.output_dir, f"{self.config.name}_comparison.csv")
        
        # Create comparison data
        comparison_data = []
        
        for config_name, results in all_results.items():
            data = {
                "configuration": config_name,
                "mean_total_time_ms": results.mean_total_time_ms,
                "mean_retrieval_time_ms": results.mean_retrieval_time_ms,
                "mean_generation_time_ms": results.mean_generation_time_ms,
                "mean_num_docs_retrieved": results.mean_num_docs_retrieved,
                "mean_num_docs_used": results.mean_num_docs_used,
                "mean_doc_score": results.mean_doc_score
            }
            
            # Add optional metrics if available
            if results.mean_answer_quality is not None:
                data["mean_answer_quality"] = results.mean_answer_quality
            if results.mean_retrieval_precision is not None:
                data["mean_retrieval_precision"] = results.mean_retrieval_precision
            if results.mean_retrieval_recall is not None:
                data["mean_retrieval_recall"] = results.mean_retrieval_recall
            if results.mean_answer_correctness is not None:
                data["mean_answer_correctness"] = results.mean_answer_correctness
            if results.mean_answer_completeness is not None:
                data["mean_answer_completeness"] = results.mean_answer_completeness
            if results.mean_answer_conciseness is not None:
                data["mean_answer_conciseness"] = results.mean_answer_conciseness
            if results.mean_answer_groundedness is not None:
                data["mean_answer_groundedness"] = results.mean_answer_groundedness
            if results.mean_answer_helpfulness is not None:
                data["mean_answer_helpfulness"] = results.mean_answer_helpfulness
            
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(comparison_path, index=False)
        
        logger.info(f"Comparison report saved to {comparison_path}")


async def run_benchmark_cli():
    """Command-line interface for running benchmarks."""
    import argparse
    import importlib.util
    
    parser = argparse.ArgumentParser(description="RAG Benchmarking Tool")
    parser.add_argument("config", help="Path to benchmark configuration file")
    args = parser.parse_args()
    
    try:
        # Load the configuration
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        
        # Parse configuration
        config = BenchmarkConfig.parse_obj(config_data)
        
        # Dynamic import to get the injector and necessary components
        spec = importlib.util.spec_from_file_location("main", "rag_bench/main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Get components from the injector
        rag_engine = main_module.injector.get(RAGEngine)
        llm_component = main_module.injector.get(LLMComponent)
        
        # Create and run the benchmark
        runner = BenchmarkRunner(config, rag_engine, llm_component)
        results = await runner.run_benchmark()
        
        logger.info("Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_benchmark_cli())