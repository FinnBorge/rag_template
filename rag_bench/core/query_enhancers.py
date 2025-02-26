"""
Query enhancers for improving RAG performance.
"""
import logging
import re
from typing import List, Optional, Dict, Any, Set

from pydantic import BaseModel, Field

from rag_bench.core.types import QueryEnhancer, LLMComponent

logger = logging.getLogger(__name__)


class HyponymExpansionEnhancer(QueryEnhancer):
    """
    Expands queries with hyponyms (more specific terms) to improve recall.
    Uses a static mapping of terms to their hyponyms.
    """
    
    def __init__(self, hyponym_map: Dict[str, List[str]]):
        """
        Initialize with a mapping of terms to their hyponyms.
        
        Args:
            hyponym_map: Dictionary mapping terms to lists of hyponyms
        """
        self.hyponym_map = hyponym_map
    
    async def enhance(self, query: str, conversation_id: Optional[str] = None) -> str:
        """
        Enhance a query by adding hyponym terms for improved recall.
        
        Args:
            query: The original query string
            conversation_id: Optional conversation ID for context
            
        Returns:
            Enhanced query with hyponym terms
        """
        original_query = query
        
        # Check if any terms in the hyponym map are in the query
        for term, hyponyms in self.hyponym_map.items():
            if re.search(rf'\b{re.escape(term)}\b', query, re.IGNORECASE):
                # Add hyponyms as OR conditions if they're not already in the query
                hyponym_terms = []
                for hyponym in hyponyms:
                    if not re.search(rf'\b{re.escape(hyponym)}\b', query, re.IGNORECASE):
                        hyponym_terms.append(hyponym)
                
                if hyponym_terms:
                    hyponym_str = " OR ".join(hyponym_terms)
                    query = f"{query} ({hyponym_str})"
        
        if query != original_query:
            logger.info(f"Enhanced query with hyponyms: {original_query} -> {query}")
        
        return query


class LLMQueryExpansionEnhancer(QueryEnhancer):
    """
    Uses an LLM to expand the query with additional relevant terms.
    """
    
    EXPANSION_TEMPLATE = """
    You are an AI assistant that helps improve search queries for a retrieval system.
    Your task is to expand the original query with additional relevant terms to improve search results.
    
    Original query: {query}
    
    Generate an expanded version of this query that:
    1. Includes synonyms of key terms
    2. Adds related concepts that might be relevant
    3. Reformulates ambiguous terms to be more specific
    4. Maintains the original intent of the query
    
    Return ONLY the expanded query text, without any explanations or additional commentary.
    """
    
    def __init__(self, llm_component: LLMComponent):
        """
        Initialize with an LLM component.
        
        Args:
            llm_component: The LLM component to use for query expansion
        """
        self.llm_component = llm_component
    
    async def enhance(self, query: str, conversation_id: Optional[str] = None) -> str:
        """
        Enhance a query using the LLM to generate an expanded version.
        
        Args:
            query: The original query string
            conversation_id: Optional conversation ID for context
            
        Returns:
            Enhanced query with additional terms from the LLM
        """
        try:
            # Generate expanded query using the LLM
            expanded_query = await self.llm_component.agenerate(
                template=self.EXPANSION_TEMPLATE,
                query=query
            )
            
            # Log the expansion for debugging
            logger.info(f"LLM expanded query: {query} -> {expanded_query}")
            
            return expanded_query.strip()
        except Exception as e:
            logger.error(f"Error in LLM query expansion: {str(e)}")
            # Return original query if expansion fails
            return query


class HybridQueryEnhancer(QueryEnhancer):
    """
    Combines multiple query enhancers into a pipeline.
    """
    
    def __init__(self, enhancers: List[QueryEnhancer]):
        """
        Initialize with a list of enhancers to apply in sequence.
        
        Args:
            enhancers: List of QueryEnhancer instances to apply
        """
        self.enhancers = enhancers
    
    async def enhance(self, query: str, conversation_id: Optional[str] = None) -> str:
        """
        Apply each enhancer in sequence to the query.
        
        Args:
            query: The original query string
            conversation_id: Optional conversation ID for context
            
        Returns:
            Query after all enhancements have been applied
        """
        current_query = query
        
        for enhancer in self.enhancers:
            current_query = await enhancer.enhance(current_query, conversation_id)
        
        return current_query


class StopWordRemovalEnhancer(QueryEnhancer):
    """
    Removes common stop words from queries to focus on meaningful terms.
    """
    
    # Common English stop words
    DEFAULT_STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
        "be", "been", "being", "in", "on", "at", "to", "for", "with", 
        "about", "by", "this", "that", "these", "those", "of"
    }
    
    def __init__(self, stop_words: Optional[Set[str]] = None):
        """
        Initialize with optional custom stop words.
        
        Args:
            stop_words: Set of stop words to remove, or None to use defaults
        """
        self.stop_words = stop_words or self.DEFAULT_STOP_WORDS
    
    async def enhance(self, query: str, conversation_id: Optional[str] = None) -> str:
        """
        Remove stop words from the query.
        
        Args:
            query: The original query string
            conversation_id: Optional conversation ID for context
            
        Returns:
            Query with stop words removed
        """
        # Split query into words
        words = query.split()
        
        # Remove stop words
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        
        # If all words were stop words, return the original query
        if not filtered_words:
            return query
        
        # Join the remaining words back into a query
        enhanced_query = " ".join(filtered_words)
        
        return enhanced_query