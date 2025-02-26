from typing import List, Optional, Dict, Any, Sequence, Tuple
import uuid
import logging
from datetime import datetime

from langchain.schema import Document as LangchainDocument
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LCDocument
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from injector import inject

from rag_bench.settings.settings import Settings
from rag_bench.core.types import QueryEnhancer, DocumentPostProcessor, DocumentWithScore, LLMComponent, VectorStoreComponent


logger = logging.getLogger(__name__)


class SourcedAnswer(BaseModel):
    """Response object with answer text and source documents."""
    answer: str
    sources: List[Dict[str, Any]]


class RAGEngine:
    """The core RAG engine that handles the retrieval and generation pipeline."""

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        query_enhancers: Optional[List[QueryEnhancer]] = None,
        document_post_processors: Optional[List[DocumentPostProcessor]] = None,
    ):
        self.settings = settings
        self.llm = llm_component
        self.vector_store = vector_store_component
        self.query_enhancers = query_enhancers or []
        self.document_post_processors = document_post_processors or []
        
        # Default text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    async def _enhance_query(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Apply all registered query enhancers to the query."""
        enhanced_query = query
        
        for enhancer in self.query_enhancers:
            try:
                enhanced_query = await enhancer.enhance(enhanced_query, conversation_id)
                logger.debug(f"Enhanced query with {enhancer.__class__.__name__}: {enhanced_query}")
            except Exception as e:
                logger.error(f"Error in query enhancer {enhancer.__class__.__name__}: {e}")
        
        return enhanced_query

    async def _retrieve_documents(self, query: str) -> List[DocumentWithScore]:
        """Retrieve relevant documents from the vector store."""
        similarity_top_k = self.settings.rag.similarity_top_k
        similarity_threshold = self.settings.rag.similarity_threshold
        
        # Retrieve documents from vector store
        results = await self.vector_store.asimilarity_search_with_scores(
            query=query,
            k=similarity_top_k
        )
        
        # Convert to DocumentWithScore
        documents_with_scores = []
        for doc, score in results:
            # Skip documents below threshold if specified
            if similarity_threshold is not None and score < similarity_threshold:
                continue
                
            # Create DocumentWithScore object using the constructor
            doc_with_score = DocumentWithScore(document=doc, score=score)
            documents_with_scores.append(doc_with_score)
        
        return documents_with_scores

    async def _post_process_documents(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """Apply document post-processors to retrieved documents."""
        processed_docs = documents
        
        for processor in self.document_post_processors:
            try:
                processed_docs = await processor.process(processed_docs, query)
                logger.debug(f"Processed documents with {processor.__class__.__name__}, remaining: {len(processed_docs)}")
            except Exception as e:
                logger.error(f"Error in document post-processor {processor.__class__.__name__}: {e}")
        
        return processed_docs

    def _format_documents_for_prompt(self, documents: List[DocumentWithScore]) -> str:
        """Format documents for inclusion in the prompt."""
        formatted_docs = []
        
        for i, doc_with_score in enumerate(documents):
            doc = doc_with_score.document
            source = doc.metadata.get("source", "Unknown")
            url = doc.metadata.get("url", "")
            
            # Format document with source information
            formatted_doc = f"Document {i+1} [Source: {source}]\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)

    def _prepare_sources_for_response(self, documents: List[DocumentWithScore]) -> List[Dict[str, Any]]:
        """Extract source information from documents for inclusion in the response."""
        sources = []
        
        for doc_with_score in documents:
            doc = doc_with_score.document
            metadata = doc.metadata
            
            source_info = {
                "source": metadata.get("source", "Unknown"),
                "relevance_score": doc_with_score.score,
            }
            
            # Add optional metadata if available
            if "url" in metadata and metadata["url"]:
                source_info["url"] = metadata["url"]
            if "title" in metadata and metadata["title"]:
                source_info["title"] = metadata["title"]
            if "date" in metadata and metadata["date"]:
                source_info["date"] = metadata["date"]
                
            sources.append(source_info)
            
        return sources

    async def _generate_answer(self, query: str, documents: List[DocumentWithScore]) -> SourcedAnswer:
        """Generate an answer using the LLM with retrieved documents as context."""
        formatted_docs = self._format_documents_for_prompt(documents)
        sources = self._prepare_sources_for_response(documents)
        
        # Define a simple template for answer generation
        template = """
        Answer the following question based on the provided documents.
        If the documents don't contain the answer, say "I don't have information about that."
        Cite sources when providing information.
        
        Documents:
        {formatted_docs}
        
        Question: {query}
        
        Answer:
        """
        
        # Generate answer
        answer = await self.llm.agenerate(
            template=template,
            formatted_docs=formatted_docs,
            query=query
        )
        
        return SourcedAnswer(
            answer=answer,
            sources=sources
        )

    async def generate_answer(
        self, 
        query: str, 
        conversation_id: Optional[str] = None
    ) -> SourcedAnswer:
        """Generate an answer for the given query using the RAG pipeline."""
        # 1. Enhance the query
        enhanced_query = await self._enhance_query(query, conversation_id)
        logger.debug(f"Enhanced query: {enhanced_query}")
        
        # 2. Retrieve documents
        retrieved_documents = await self._retrieve_documents(enhanced_query)
        logger.debug(f"Retrieved {len(retrieved_documents)} documents")
        
        # 3. Post-process documents
        processed_documents = await self._post_process_documents(retrieved_documents, enhanced_query)
        logger.debug(f"After post-processing: {len(processed_documents)} documents")
        
        # 4. Generate answer
        answer = await self._generate_answer(enhanced_query, processed_documents)
        
        return answer