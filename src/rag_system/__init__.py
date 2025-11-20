"""RAWE System - A production-grade Retrieval-Augmented Web Engine."""

from .config import RAWEConfig, get_config, update_config
from .models import (
    Document,
    TextChunk,
    RetrievalResult,
    QueryResult,
    IngestionMetrics,
    QueryMetrics,
    CrawlStatus,
    WebhookPayload
)
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .web_crawler import AsyncWebCrawler
from .retriever import Retriever
from .answer_generator import AnswerGenerator
from .rag_system import RAWESystem

__version__ = "1.0.0"
__author__ = "RAWE System"

__all__ = [
    # Core system
    "RAWESystem",
    
    # Configuration
    "RAWEConfig",
    "get_config",
    "update_config",
    
    # Models
    "Document",
    "TextChunk",
    "RetrievalResult",
    "QueryResult",
    "IngestionMetrics",
    "QueryMetrics",
    "CrawlStatus",
    "WebhookPayload",
    
    # Components
    "DocumentProcessor",
    "EmbeddingGenerator",
    "VectorStore",
    "AsyncWebCrawler",
    "Retriever",
    "AnswerGenerator",
]