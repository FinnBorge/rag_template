"""
Document post-processors for improving RAG performance.
"""
import logging
from typing import List, Optional, Dict, Any, Callable

import numpy as np
from pydantic import BaseModel

from rag_bench.core.types import DocumentWithScore, DocumentPostProcessor, EmbeddingComponent, LLMComponent

logger = logging.getLogger(__name__)


class ThresholdFilter(DocumentPostProcessor):
    """
    Filters documents based on a similarity threshold.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize with a similarity threshold.
        
        Args:
            threshold: Minimum similarity score to keep a document
        """
        self.threshold = threshold
    
    async def process(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """
        Filter documents that don't meet the threshold score.
        
        Args:
            documents: List of documents with scores
            query: The original query
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = [
            doc for doc in documents 
            if doc.score >= self.threshold
        ]
        
        logger.info(
            f"Threshold filter: {len(documents)} -> {len(filtered_docs)} documents "
            f"(threshold={self.threshold})"
        )
        
        return filtered_docs


class SemanticReranker(DocumentPostProcessor):
    """
    Reranks documents based on semantic similarity to the query.
    """
    
    def __init__(self, embedding_component: EmbeddingComponent):
        """
        Initialize with an embedding component.
        
        Args:
            embedding_component: Component for generating embeddings
        """
        self.embedding_component = embedding_component
    
    async def process(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """
        Rerank documents based on semantic similarity to the query.
        
        Args:
            documents: List of documents with scores
            query: The original query
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Get embeddings for the query and all documents
        query_embedding = self.embedding_component.embed_query(query)
        
        doc_texts = [doc.document.page_content for doc in documents]
        doc_embeddings = self.embedding_component.embed_documents(doc_texts)
        
        # Calculate cosine similarity between query and each document
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Calculate new scores based on semantic similarity
        new_scores = [
            cosine_similarity(query_embedding, doc_emb) 
            for doc_emb in doc_embeddings
        ]
        
        # Create new DocumentWithScore objects with updated scores
        reranked_docs = [
            DocumentWithScore(document=doc.document, score=score)
            for doc, score in zip(documents, new_scores)
        ]
        
        # Sort by new scores in descending order
        reranked_docs.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Semantically reranked {len(documents)} documents")
        
        return reranked_docs


class LLMReranker(DocumentPostProcessor):
    """
    Uses an LLM to rerank documents based on relevance to the query.
    """
    
    RERANK_TEMPLATE = """
    You are an AI assistant that helps rank documents based on their relevance to a query.
    Your task is to score the relevance of each document to the query on a scale from 0 to 10,
    where 0 means completely irrelevant and 10 means perfectly relevant.
    
    Query: {query}
    
    Documents:
    {documents}
    
    For each document, assign a relevance score from 0 to 10.
    Return your response as a comma-separated list of numbers, one for each document, in the same order.
    For example: 8,5,9,2,7
    
    IMPORTANT: Return ONLY the comma-separated list of numbers, nothing else.
    """
    
    def __init__(self, llm_component: LLMComponent):
        """
        Initialize with an LLM component.
        
        Args:
            llm_component: Component for LLM operations
        """
        self.llm_component = llm_component
    
    async def process(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """
        Use an LLM to rerank documents based on relevance to the query.
        
        Args:
            documents: List of documents with scores
            query: The original query
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Format documents for the prompt
        doc_texts = []
        for i, doc in enumerate(documents):
            doc_text = doc.document.page_content[:500]  # Truncate long documents
            doc_texts.append(f"Document {i+1}: {doc_text}")
        
        formatted_docs = "\n\n".join(doc_texts)
        
        try:
            # Get relevance scores from the LLM
            response = await self.llm_component.agenerate(
                template=self.RERANK_TEMPLATE,
                query=query,
                documents=formatted_docs
            )
            
            # Parse scores from the response
            try:
                scores = [float(score.strip()) for score in response.split(",")]
                
                # Normalize scores to [0, 1] range
                if scores:
                    max_score = max(scores)
                    if max_score > 0:
                        scores = [score / 10 for score in scores]
                
                # Make sure we have the right number of scores
                if len(scores) != len(documents):
                    logger.warning(
                        f"LLM returned {len(scores)} scores for {len(documents)} documents. "
                        f"Using original scores."
                    )
                    return documents
                
                # Create new DocumentWithScore objects with updated scores
                reranked_docs = [
                    DocumentWithScore(document=doc.document, score=score)
                    for doc, score in zip(documents, scores)
                ]
                
                # Sort by new scores in descending order
                reranked_docs.sort(key=lambda x: x.score, reverse=True)
                
                logger.info(f"LLM reranked {len(documents)} documents")
                
                return reranked_docs
            except Exception as e:
                logger.error(f"Error parsing LLM reranking response: {str(e)}")
                return documents
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
            return documents


class DiversityReranker(DocumentPostProcessor):
    """
    Reranks documents to maximize diversity of information.
    """
    
    def __init__(self, embedding_component: EmbeddingComponent, diversity_weight: float = 0.5):
        """
        Initialize with an embedding component and diversity weight.
        
        Args:
            embedding_component: Component for generating embeddings
            diversity_weight: Weight for diversity vs relevance (0-1)
        """
        self.embedding_component = embedding_component
        self.diversity_weight = diversity_weight
    
    async def process(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """
        Rerank documents to maximize diversity while maintaining relevance.
        
        Args:
            documents: List of documents with scores
            query: The original query
            
        Returns:
            Reranked list of documents
        """
        if len(documents) <= 1:
            return documents
        
        # Extract document contents
        doc_texts = [doc.document.page_content for doc in documents]
        
        # Get embeddings for all documents
        embeddings = self.embedding_component.embed_documents(doc_texts)
        
        # Initialize with the highest scoring document
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        selected_indices = [0]
        remaining_indices = list(range(1, len(documents)))
        
        # Greedy selection to maximize diversity
        while remaining_indices and len(selected_indices) < len(documents):
            # Calculate diversity scores (negative of max similarity to selected docs)
            max_similarities = []
            
            for i in remaining_indices:
                similarities = [
                    np.dot(embeddings[i], embeddings[j]) / 
                    (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    for j in selected_indices
                ]
                max_similarities.append(max(similarities))
            
            # Calculate combined score: (1-w) * relevance + w * diversity
            combined_scores = [
                (1 - self.diversity_weight) * sorted_docs[i].score - 
                self.diversity_weight * max_similarities[remaining_indices.index(i)]
                for i in remaining_indices
            ]
            
            # Select the document with the highest combined score
            best_idx = remaining_indices[combined_scores.index(max(combined_scores))]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Reorder the documents
        reranked_docs = [sorted_docs[i] for i in selected_indices]
        
        logger.info(f"Diversity reranked {len(documents)} documents")
        
        return reranked_docs


class ProcessingPipeline(DocumentPostProcessor):
    """
    Applies multiple document processors in sequence.
    """
    
    def __init__(self, processors: List[DocumentPostProcessor]):
        """
        Initialize with a list of processors to apply in sequence.
        
        Args:
            processors: List of DocumentPostProcessor instances
        """
        self.processors = processors
    
    async def process(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """
        Apply each processor in sequence to the documents.
        
        Args:
            documents: List of documents with scores
            query: The original query
            
        Returns:
            Documents after all processors have been applied
        """
        current_docs = documents
        
        for processor in self.processors:
            current_docs = await processor.process(current_docs, query)
            
            # If a processor filtered out all documents, stop processing
            if not current_docs:
                logger.warning("All documents filtered out during processing pipeline")
                break
        
        return current_docs