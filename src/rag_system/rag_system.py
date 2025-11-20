"""Main RAWE system orchestrating all components."""

import asyncio
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import Document, QueryResult, IngestionMetrics, QueryMetrics
from .config import get_config
from .web_crawler import AsyncWebCrawler
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import Retriever
from .answer_generator import AnswerGenerator


class RAWESystem:
    """Complete RAWE system for document ingestion and question answering."""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, self.embedding_generator)
        self.answer_generator = AnswerGenerator()
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
        
        print("Initializing RAWE system...")
        
        # Initialize components that need async setup
        await self.embedding_generator.initialize()
        await self.vector_store.initialize()
        await self.answer_generator.initialize()
        
        # Try to load existing vector store
        loaded = await self.vector_store.load()
        if loaded:
            print("Loaded existing vector store.")
        else:
            print("No existing vector store found.")
        
        self.initialized = True
        print("RAWE system initialized successfully.")
    
    async def ingest_website(self, url: str) -> IngestionMetrics:
        """
        Ingest content from a website.
        
        Args:
            url: Starting URL for crawling
            
        Returns:
            Ingestion metrics
        """
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize()
        
        print(f"Starting ingestion from: {url}")
        
        # Crawl website
        crawl_start = time.time()
        async with AsyncWebCrawler() as crawler:
            documents = await crawler.crawl_website(url)
            crawl_metrics = crawler.get_crawl_metrics()
        
        crawl_time = time.time() - crawl_start
        print(f"Crawling completed in {crawl_time:.1f}s. Found {len(documents)} documents.")
        
        if not documents:
            return IngestionMetrics(
                total_time=time.time() - start_time,
                pages_scraped=crawl_metrics['pages_scraped'],
                pages_failed=crawl_metrics['pages_failed'],
                total_chunks_created=0,
                total_tokens_processed=0,
                embedding_generation_time=0.0,
                indexing_time=0.0,
                average_scraping_time=crawl_metrics['average_response_time'],
                errors=crawl_metrics['errors']
            )
        
        # Process documents into chunks
        process_start = time.time()
        all_chunks = []
        total_tokens = 0
        
        for document in documents:
            chunks = self.document_processor.chunk_document(document)
            all_chunks.extend(chunks)
            total_tokens += len(document.content.split())
        
        process_time = time.time() - process_start
        print(f"Document processing completed in {process_time:.1f}s. Created {len(all_chunks)} chunks.")
        
        # Generate embeddings
        embedding_start = time.time()
        chunks_with_embeddings = await self.embedding_generator.embed_text_chunks(all_chunks)
        embedding_time = time.time() - embedding_start
        print(f"Embedding generation completed in {embedding_time:.1f}s.")
        
        # Add to vector store
        indexing_start = time.time()
        await self.vector_store.add_chunks(chunks_with_embeddings, documents)
        indexing_time = time.time() - indexing_start
        print(f"Vector indexing completed in {indexing_time:.1f}s.")
        
        # Save vector store
        await self.vector_store.save()
        print("Vector store saved to disk.")
        
        total_time = time.time() - start_time
        
        metrics = IngestionMetrics(
            total_time=total_time,
            pages_scraped=crawl_metrics['pages_scraped'],
            pages_failed=crawl_metrics['pages_failed'],
            total_chunks_created=len(all_chunks),
            total_tokens_processed=total_tokens,
            embedding_generation_time=embedding_time,
            indexing_time=indexing_time,
            average_scraping_time=crawl_metrics['average_response_time'],
            errors=crawl_metrics['errors']
        )
        
        print("Ingestion completed successfully!")
        print(metrics.print_summary())
        
        return metrics
    
    async def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        """
        Answer a question using the RAWE system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Query result with answer and sources
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_start = time.time()
        retrieval_results = await self.retriever.retrieve(question, top_k)
        retrieval_time = time.time() - retrieval_start
        
        if not retrieval_results:
            return QueryResult(
                query=question,
                answer="I don't have any relevant information to answer this question.",
                sources=[],
                metadata={'no_results_found': True},
                processing_time=time.time() - start_time
            )
        
        # Generate answer
        result = await self.answer_generator.generate_answer(question, retrieval_results)
        
        # Update metrics with retrieval time
        result.metadata['retrieval_time'] = retrieval_time
        
        return result
    
    async def batch_query(self, questions: List[str], top_k: Optional[int] = None) -> List[QueryResult]:
        """
        Answer multiple questions concurrently.
        
        Args:
            questions: List of questions
            top_k: Number of documents to retrieve per question
            
        Returns:
            List of query results
        """
        if not self.initialized:
            await self.initialize()
        
        # Process questions concurrently
        tasks = [self.query(question, top_k) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = QueryResult(
                    query=questions[i],
                    answer=f"Error processing question: {str(result)}",
                    sources=[],
                    metadata={'error': str(result)},
                    processing_time=0.0
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def get_document_by_url(self, url: str) -> Optional[Document]:
        """Get a document by its URL."""
        # This is a simple implementation - could be optimized with URL index
        for doc_id, document in self.vector_store.documents.items():
            if document.url == url:
                return document
        return None
    
    async def search_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Search for documents without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        if not self.initialized:
            await self.initialize()
        
        retrieval_results = await self.retriever.retrieve(query, top_k)
        
        # Extract unique documents
        seen_docs = set()
        documents = []
        
        for result in retrieval_results:
            if result.document and result.document.id not in seen_docs:
                documents.append(result.document)
                seen_docs.add(result.document.id)
        
        return documents
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self.initialized:
            await self.initialize()
        
        stats = {
            'vector_store': self.vector_store.get_stats(),
            'retrieval': self.retriever.get_retrieval_stats(),
            'embedding_model': self.embedding_generator.get_model_info(),
            'answer_generator': self.answer_generator.get_model_info(),
            'config': {
                'crawler': self.config.crawler.dict(),
                'processing': self.config.processing.dict(),
                'embedding': self.config.embedding.dict(),
                'retrieval': self.config.retrieval.dict(),
                'llm': self.config.llm.dict(),
                'answer_generation': self.config.answer_generation.dict()
            }
        }
        
        return stats
    
    async def rebuild_index(self):
        """Rebuild the vector index (useful after configuration changes)."""
        if not self.initialized:
            await self.initialize()
        
        print("Rebuilding vector index...")
        await self.vector_store.rebuild_index()
        await self.vector_store.save()
        print("Index rebuilt successfully.")
    
    async def clear_all_data(self):
        """Clear all stored data (use with caution)."""
        if not self.initialized:
            await self.initialize()
        
        print("Clearing all data...")
        await self.vector_store.clear()
        
        # Clear data directories
        import shutil
        for dir_path in [self.config.raw_data_dir, self.config.processed_data_dir]:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("All data cleared.")
    
    async def export_data(self, export_path: str):
        """Export all data to a specified path."""
        if not self.initialized:
            await self.initialize()
        
        import json
        import shutil
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export documents
        documents_data = []
        for doc in self.vector_store.documents.values():
            documents_data.append(doc.dict())
        
        with open(export_dir / "documents.json", 'w') as f:
            json.dump(documents_data, f, indent=2)
        
        # Export chunks
        chunks_data = [chunk.dict() for chunk in self.vector_store.chunks]
        with open(export_dir / "chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        # Copy vector store files
        vector_store_path = Path(self.config.vector_store.storage_path)
        if vector_store_path.exists():
            shutil.copytree(vector_store_path, export_dir / "vector_store", dirs_exist_ok=True)
        
        # Export system stats
        stats = await self.get_system_stats()
        with open(export_dir / "system_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Data exported to {export_path}")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'vector_store') and self.vector_store:
            # Save vector store on cleanup
            try:
                asyncio.create_task(self.vector_store.save())
            except:
                pass