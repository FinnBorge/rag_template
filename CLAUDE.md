# RAG Bench Development Guide

## Commands
- Run server: `poetry run python -m rag_bench.main`
- Run benchmark: `poetry run python -m rag_bench.evaluation.run_benchmark --config path/to/benchmark_config.json`
- Run tests: `poetry run pytest`
- Install dependencies: `poetry install`

## Code Style
- **Imports**: Standard library first, third-party second, local modules last
- **Type Annotations**: Always include type hints (parameters and return values)
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Async**: Use async/await for all I/O operations; prefix async method names with 'a'
- **Error Handling**: Use try/except with specific exceptions and helpful error messages
- **Documentation**: Docstrings for all modules, classes, and functions
- **Classes**: Use Protocol classes for interfaces, ABC for abstract classes

## Project Structure
- Use dependency injection for component creation and configuration
- Components are in separate directories with clear responsibilities
- Settings are loaded from YAML and environment variables
- New implementations should follow existing patterns