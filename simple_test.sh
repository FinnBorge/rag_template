#\!/bin/bash
# Simple test script for RAG Bench server

echo "Testing RAG Bench Server..."
echo "================================================"

# Test basic query
echo "Query: What is RAG?"
curl -s "http://localhost:8000/api/v1/query?q=What%20is%20RAG"
echo -e "\n\n"

# Test complex query
echo "Query: What are the key components of a RAG system?"
curl -s "http://localhost:8000/api/v1/query?q=What%20are%20the%20key%20components%20of%20a%20RAG%20system"
echo -e "\n\n"

echo "Tests completed\!"
