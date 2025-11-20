"""Embedding generation module using sentence-transformers."""

import asyncio
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .models import TextChunk
from .config import get_config


class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers."""
    
    def __init__(self):
        self.config = get_config()
        self.model: Optional[SentenceTransformer] = None
        self._device = self.config.embedding.device
        
    async def initialize(self):
        """Initialize the embedding model asynchronously."""
        if self.model is None:
            # Run model loading in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                self._load_model
            )
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        model = SentenceTransformer(
            self.config.embedding.model_name,
            device=self._device
        )
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.model is None:
            await self.initialize()
        
        # Process in batches to avoid memory issues
        batch_size = self.config.embedding.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = await self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        loop = asyncio.get_event_loop()
        
        # Run embedding generation in executor
        embeddings = await loop.run_in_executor(
            None,
            self._generate_batch_embeddings,
            texts
        )
        
        return embeddings.tolist()
    
    def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings synchronously."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return embeddings
    
    async def embed_text_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for text chunks and update them in place.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return chunks
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        if not query.strip():
            return [0.0] * self.config.embedding.dimension
        
        embeddings = await self.generate_embeddings([query])
        return embeddings[0] if embeddings else [0.0] * self.config.embedding.dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def get_embedding_stats(self, embeddings: List[List[float]]) -> dict:
        """
        Get statistics about embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Dictionary with statistics
        """
        if not embeddings:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        stats = {
            'count': len(embeddings),
            'dimension': len(embeddings[0]) if embeddings else 0,
            'mean_magnitude': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            'std_magnitude': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
            'mean_values': embeddings_array.mean(axis=0).tolist(),
            'std_values': embeddings_array.std(axis=0).tolist()
        }
        
        return stats
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        if self.model is None:
            return {
                'model_name': self.config.embedding.model_name,
                'loaded': False
            }
        
        return {
            'model_name': self.config.embedding.model_name,
            'dimension': self.config.embedding.dimension,
            'device': str(self.model.device),
            'loaded': True,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown')
        }