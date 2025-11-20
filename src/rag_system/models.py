"""Data models for the RAWE system."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in the RAWE system."""
    
    id: str = Field(description="Unique document identifier")
    url: str = Field(description="Source URL")
    title: str = Field(description="Document title")
    content: str = Field(description="Main content")
    short_description: str = Field(default="", description="Short description/summary")
    raw_html: str = Field(default="", description="Raw HTML content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TextChunk(BaseModel):
    """Represents a text chunk for processing."""
    
    id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk content")
    start_char: int = Field(description="Start character position in document")
    end_char: int = Field(description="End character position in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Text embedding")
    
    class Config:
        arbitrary_types_allowed = True


class RetrievalResult(BaseModel):
    """Represents a retrieval result."""
    
    chunk: TextChunk = Field(description="Retrieved text chunk")
    score: float = Field(description="Similarity/relevance score")
    document: Optional[Document] = Field(default=None, description="Source document")


class QueryResult(BaseModel):
    """Represents the result of a query."""
    
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    sources: List[RetrievalResult] = Field(description="Source documents/chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: float = Field(description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    metrics: Optional['QueryMetrics'] = Field(default=None, description="Query processing metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IngestionMetrics(BaseModel):
    """Metrics for the ingestion process."""
    
    total_time: float = Field(description="Total processing time in seconds")
    pages_scraped: int = Field(description="Number of pages successfully scraped")
    pages_failed: int = Field(description="Number of pages that failed to scrape")
    total_chunks_created: int = Field(description="Total number of text chunks created")
    total_tokens_processed: int = Field(description="Total number of tokens processed")
    embedding_generation_time: float = Field(description="Time spent generating embeddings")
    indexing_time: float = Field(description="Time spent building vector index")
    average_scraping_time: float = Field(description="Average time per page scraping")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    
    def print_summary(self) -> str:
        """Generate a formatted summary of the metrics."""
        return f"""
=== Ingestion Metrics ===
Total Time: {self.total_time:.1f}s
Pages Scraped: {self.pages_scraped}
Pages Failed: {self.pages_failed}
Total Chunks Created: {self.total_chunks_created:,}
Total Tokens Processed: {self.total_tokens_processed:,}
Embedding Generation Time: {self.embedding_generation_time:.1f}s
Indexing Time: {self.indexing_time:.1f}s
Average Scraping Time per Page: {self.average_scraping_time:.1f}s
Errors: {len(self.errors)} error(s)
        """.strip()


class QueryMetrics(BaseModel):
    """Metrics for a single query."""
    
    total_latency: float = Field(description="Total query latency in seconds")
    retrieval_time: float = Field(description="Time spent on retrieval")
    llm_time: float = Field(description="Time spent on LLM generation")
    post_processing_time: float = Field(description="Time spent on post-processing")
    documents_retrieved: int = Field(description="Number of documents retrieved")
    documents_used_in_answer: int = Field(description="Number of documents used in final answer")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")
    estimated_cost: float = Field(default=0.0, description="Estimated cost (if applicable)")
    
    def print_summary(self) -> str:
        """Generate a formatted summary of the metrics."""
        return f"""
Metrics:
  Total Latency: {self.total_latency:.1f}s
  Retrieval Time: {self.retrieval_time:.1f}s
  LLM Time: {self.llm_time:.1f}s
  Post-processing Time: {self.post_processing_time:.1f}s
  Documents Retrieved: {self.documents_retrieved}
  Documents Used in Answer: {self.documents_used_in_answer}
  Input Tokens: {self.input_tokens:,}
  Output Tokens: {self.output_tokens:,}
  Estimated Cost: ${self.estimated_cost:.4f}
        """.strip()


class CrawlStatus(BaseModel):
    """Status of a crawl operation."""
    
    url: str = Field(description="URL being crawled")
    status: str = Field(description="Status (pending, success, failed)")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    response_time: Optional[float] = Field(default=None, description="Response time in seconds")
    content_length: Optional[int] = Field(default=None, description="Content length in bytes")
    
    
class WebhookPayload(BaseModel):
    """Payload for webhook notifications."""
    
    operation: str = Field(description="Operation type (ingestion, query)")
    status: str = Field(description="Operation status (success, failed)")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Operation metrics")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }