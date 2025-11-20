"""Retrieval system with vector search and optional reranking."""

import asyncio
import time
from typing import List, Optional
from sentence_transformers import CrossEncoder

from .models import RetrievalResult
from .vector_store import VectorStore
from .embedding_generator import EmbeddingGenerator
from .config import get_config


class Retriever:
    """Retrieves relevant documents using vector search and optional reranking."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.config = get_config()
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.reranker: Optional[CrossEncoder] = None
        
    async def initialize_reranker(self):
        """Initialize the reranking model if enabled."""
        if self.config.retrieval.use_reranking and self.reranker is None:
            loop = asyncio.get_event_loop()
            self.reranker = await loop.run_in_executor(
                None,
                self._load_reranker
            )
    
    def _load_reranker(self) -> CrossEncoder:
        """Load the cross-encoder reranking model."""
        return CrossEncoder(
            self.config.retrieval.reranker_model,
            device=self.config.embedding.device
        )
    
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of retrieval results
        """
        if top_k is None:
            top_k = self.config.retrieval.top_k
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await self.embedding_generator.embed_query(query)
        
        # Search vector store
        search_k = max(top_k, self.config.retrieval.rerank_top_k * self.config.retrieval.search_multiplier)  # Get more for reranking
        
        vector_results = await self.vector_store.search(
            query_embedding, 
            k=search_k
        )
        
        if not vector_results:
            return []
        
        # Convert to RetrievalResult objects
        results = []
        for chunk, score in vector_results:
            document = self.vector_store.get_document(chunk.document_id)
            result = RetrievalResult(
                chunk=chunk,
                score=score,
                document=document
            )
            results.append(result)
        
        # Apply reranking if enabled
        if self.config.retrieval.use_reranking and len(results) > 1:
            results = await self._rerank_results(query, results)
        
        # Return top results
        final_results = results[:top_k]
        
        # Add retrieval metadata
        retrieval_time = time.time() - start_time
        for result in final_results:
            result.chunk.metadata['retrieval_time'] = retrieval_time
            result.chunk.metadata['query'] = query
        
        return final_results
    
    async def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original query
            results: List of retrieval results
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # Initialize reranker if needed
        if self.reranker is None:
            await self.initialize_reranker()
        
        if self.reranker is None:
            return results
        
        # Prepare query-document pairs
        pairs = []
        for result in results:
            # Use chunk content for reranking
            pairs.append([query, result.chunk.content])
        
        # Get reranking scores
        loop = asyncio.get_event_loop()
        rerank_scores = await loop.run_in_executor(
            None,
            self._compute_rerank_scores,
            pairs
        )
        
        # Update scores and sort
        for result, new_score in zip(results, rerank_scores):
            result.score = float(new_score)
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:self.config.retrieval.rerank_top_k]
    
    def _compute_rerank_scores(self, pairs: List[List[str]]) -> List[float]:
        """Compute reranking scores synchronously."""
        scores = self.reranker.predict(pairs)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    async def retrieve_with_filters(self, 
                                  query: str, 
                                  document_ids: Optional[List[str]] = None,
                                  metadata_filters: Optional[dict] = None,
                                  top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve documents with additional filtering.
        
        Args:
            query: Search query
            document_ids: Filter by specific document IDs
            metadata_filters: Filter by metadata attributes
            top_k: Number of results to return
            
        Returns:
            Filtered retrieval results
        """
        # Get initial results
        results = await self.retrieve(query, top_k=top_k * self.config.retrieval.filter_search_multiplier if top_k else None)  # Get more for filtering
        
        # Apply filters
        filtered_results = []
        
        for result in results:
            # Filter by document IDs
            if document_ids and result.chunk.document_id not in document_ids:
                continue
            
            # Filter by metadata
            if metadata_filters:
                chunk_metadata = result.chunk.metadata
                doc_metadata = result.document.metadata if result.document else {}
                combined_metadata = {**chunk_metadata, **doc_metadata}
                
                # Check if all filter conditions are met
                if not self._matches_metadata_filters(combined_metadata, metadata_filters):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results[:top_k] if top_k else filtered_results
    
    def _matches_metadata_filters(self, metadata: dict, filters: dict) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # Handle different filter types
            if isinstance(value, dict):
                # Range filters like {'gte': 0.5}
                if 'gte' in value and metadata_value < value['gte']:
                    return False
                if 'lte' in value and metadata_value > value['lte']:
                    return False
                if 'gt' in value and metadata_value <= value['gt']:
                    return False
                if 'lt' in value and metadata_value >= value['lt']:
                    return False
            elif isinstance(value, list):
                # Value must be in list
                if metadata_value not in value:
                    return False
            else:
                # Exact match
                if metadata_value != value:
                    return False
        
        return True
    
    async def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        # Get the reference chunk
        reference_chunk = self.vector_store.get_chunk(chunk_id)
        if not reference_chunk or not reference_chunk.embedding:
            return []
        
        # Search using the chunk's embedding
        vector_results = await self.vector_store.search(reference_chunk.embedding, k=top_k + 1)
        
        # Convert to RetrievalResult objects, excluding the original chunk
        results = []
        for chunk, score in vector_results:
            if chunk.id != chunk_id:  # Exclude the reference chunk itself
                document = self.vector_store.get_document(chunk.document_id)
                result = RetrievalResult(
                    chunk=chunk,
                    score=score,
                    document=document
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_retrieval_stats(self) -> dict:
        """Get statistics about the retrieval system."""
        stats = {
            'vector_store_stats': self.vector_store.get_stats(),
            'embedding_model_info': self.embedding_generator.get_model_info(),
            'reranker_loaded': self.reranker is not None,
            'reranker_model': self.config.retrieval.reranker_model if self.config.retrieval.use_reranking else None,
            'config': {
                'top_k': self.config.retrieval.top_k,
                'rerank_top_k': self.config.retrieval.rerank_top_k,
                'use_reranking': self.config.retrieval.use_reranking
            }
        }
        
        return stats