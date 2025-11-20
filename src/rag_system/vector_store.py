"""Vector storage and retrieval using FAISS."""

import asyncio
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import faiss

from .models import TextChunk, Document
from .config import get_config


class VectorStore:
    """FAISS-based vector storage for text chunks."""
    
    def __init__(self):
        self.config = get_config()
        self.dimension = self.config.embedding.dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[TextChunk] = []
        self.documents: Dict[str, Document] = {}
        self._storage_path = Path(self.config.vector_store.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        if self.config.vector_store.similarity_metric == "cosine":
            # For cosine similarity, use normalized vectors with L2 distance
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        else:
            # Default to L2 distance
            index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    async def initialize(self):
        """Initialize the vector store."""
        if self.index is None:
            self.index = self._create_index()
    
    async def add_chunks(self, chunks: List[TextChunk], documents: Optional[List[Document]] = None):
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of text chunks with embeddings
            documents: Optional list of source documents
        """
        if not chunks:
            return
        
        if self.index is None:
            await self.initialize()
        
        # Store documents for reference
        if documents:
            for doc in documents:
                self.documents[doc.id] = doc
        
        # Extract embeddings
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.embedding and len(chunk.embedding) == self.dimension:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if not embeddings:
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        if self.config.vector_store.similarity_metric == "cosine":
            faiss.normalize_L2(embeddings_array)
        
        # Add to index
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.index.add, embeddings_array)
        
        # Store chunks
        self.chunks.extend(valid_chunks)
    
    async def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar text chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None or not self.chunks:
            return []
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.config.vector_store.similarity_metric == "cosine":
            faiss.normalize_L2(query_array)
        
        # Search
        loop = asyncio.get_event_loop()
        scores, indices = await loop.run_in_executor(
            None, 
            self.index.search, 
            query_array, 
            min(k, len(self.chunks))
        )
        
        # Convert results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert distance to similarity score
                if self.config.vector_store.similarity_metric == "cosine":
                    similarity = float(score)  # Inner product for normalized vectors
                else:
                    similarity = 1.0 / (1.0 + float(score))  # Convert L2 distance to similarity
                
                results.append((chunk, similarity))
        
        return results
    
    async def save(self, filename: str = "vector_store"):
        """
        Save the vector store to disk.
        
        Args:
            filename: Base filename for saving
        """
        if self.index is None:
            return
        
        base_path = self._storage_path / filename
        
        # Save FAISS index
        index_path = str(base_path) + ".index"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, faiss.write_index, self.index, index_path)
        
        # Save chunks and documents
        data = {
            'chunks': [chunk.dict() for chunk in self.chunks],
            'documents': {doc_id: doc.dict() for doc_id, doc in self.documents.items()},
            'config': {
                'dimension': self.dimension,
                'similarity_metric': self.config.vector_store.similarity_metric
            }
        }
        
        data_path = str(base_path) + ".pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata as JSON for human readability
        metadata = {
            'num_vectors': len(self.chunks),
            'dimension': self.dimension,
            'num_documents': len(self.documents),
            'similarity_metric': self.config.vector_store.similarity_metric
        }
        
        meta_path = str(base_path) + "_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    async def load(self, filename: str = "vector_store") -> bool:
        """
        Load the vector store from disk.
        
        Args:
            filename: Base filename for loading
            
        Returns:
            True if successfully loaded, False otherwise
        """
        base_path = self._storage_path / filename
        index_path = str(base_path) + ".index"
        data_path = str(base_path) + ".pkl"
        
        try:
            # Check if files exist
            if not Path(index_path).exists() or not Path(data_path).exists():
                return False
            
            # Load FAISS index
            loop = asyncio.get_event_loop()
            self.index = await loop.run_in_executor(None, faiss.read_index, index_path)
            
            # Load chunks and documents
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Restore chunks
            self.chunks = [TextChunk(**chunk_data) for chunk_data in data['chunks']]
            
            # Restore documents
            self.documents = {
                doc_id: Document(**doc_data) 
                for doc_id, doc_data in data['documents'].items()
            }
            
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            'num_vectors': len(self.chunks),
            'num_documents': len(self.documents),
            'dimension': self.dimension,
            'index_type': type(self.index).__name__ if self.index else None,
            'similarity_metric': self.config.vector_store.similarity_metric,
            'storage_path': str(self._storage_path)
        }
        
        if self.index:
            stats['index_size'] = self.index.ntotal
        
        return stats
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(document_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get a chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    async def clear(self):
        """Clear all data from the vector store."""
        self.index = self._create_index()
        self.chunks = []
        self.documents = {}
    
    async def rebuild_index(self):
        """Rebuild the FAISS index from existing chunks."""
        if not self.chunks:
            return
        
        # Create new index
        self.index = self._create_index()
        
        # Extract embeddings
        embeddings = []
        for chunk in self.chunks:
            if chunk.embedding and len(chunk.embedding) == self.dimension:
                embeddings.append(chunk.embedding)
        
        if not embeddings:
            return
        
        # Convert to numpy array and add to index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if self.config.vector_store.similarity_metric == "cosine":
            faiss.normalize_L2(embeddings_array)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.index.add, embeddings_array)