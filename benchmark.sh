#\!/bin/bash
# Simple script to run a RAG benchmark

# Clean the benchmark_results directory
rm -rf benchmark_results
mkdir -p benchmark_results

# Run the benchmark with the default config
poetry run python -m rag_bench.evaluation.run_benchmark --config rag_bench/evaluation/sample_data/benchmark_config.json

# Show results
echo ""
echo "Benchmark complete\! Results are in the benchmark_results directory."
echo "Summary files:"
ls -la benchmark_results/*_summary.txt
