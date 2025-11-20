"""Configuration settings for the RAWE system."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class CrawlerConfig(BaseModel):
    """Configuration for web crawling."""
    max_depth: int = Field(default=5, description="Maximum crawling depth")
    max_pages: int = Field(default=150, description="Maximum number of pages to crawl")
    delay_between_requests: float = Field(default=0.8, description="Delay between requests in seconds")
    concurrent_requests: int = Field(default=8, description="Number of concurrent requests")
    user_agent: str = Field(
        default="RAWE-System/1.0 (Educational Purpose)",
        description="User agent for requests"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    allowed_domains: List[str] = Field(
        default=[],
        description="Allowed domains for crawling (empty list means no domain restrictions)"
    )
    # URL path filters for content discovery
    primary_path_filters: List[str] = Field(
        default=["/products/", "/solutions/"],
        description="URL path patterns to identify primary content pages"
    )
    # URL content indicators for relevance filtering
    url_indicators: List[str] = Field(
        default=["/product", "/solution", "/service", "bizpay", "collect", "payout", "ramp", "wallet", "supported", "countries", "tokens", "currencies"],
        description="URL patterns that indicate relevant content"
    )
    # Content keywords for relevance filtering  
    content_indicators: List[str] = Field(
        default=["product", "solution", "service", "feature", "platform", "payment", "fintech", "financial", "banking", "stablecoin", "bizpay", "collect", "payout", "ramp", "wallet", "cross-border", "remittance", "settlement", "transaction", "enterprise", "business", "merchant", "vendor", "currency", "exchange", "rate", "conversion"],
        description="Content keywords that indicate relevant pages"
    )
    # Discovery limits
    max_primary_pages_to_crawl: int = Field(default=10, description="Maximum primary pages to crawl for child discovery")
    max_child_links_per_page: int = Field(default=5, description="Maximum child links to discover per primary page")


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    remove_empty_chunks: bool = Field(default=True, description="Remove empty chunks")
    
    # Content extraction parameters
    short_description_length: int = Field(default=200, description="Maximum length for short descriptions")
    min_content_indicators: int = Field(default=2, description="Minimum content indicators required for relevance")
    min_transfi_indicators: int = Field(default=1, description="Minimum TransFi-specific indicators for relevance")


class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    dimension: int = Field(default=384, description="Embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    device: str = Field(default="cpu", description="Device for computation")


class VectorStoreConfig(BaseModel):
    """Configuration for vector storage."""
    index_type: str = Field(default="faiss", description="Vector index type (faiss/chromadb)")
    similarity_metric: str = Field(default="cosine", description="Similarity metric")
    storage_path: str = Field(default="data/vector_store", description="Storage path")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""
    top_k: int = Field(default=5, description="Number of top results to retrieve")
    rerank_top_k: int = Field(default=5, description="Number of results after reranking")
    use_reranking: bool = Field(default=True, description="Whether to use reranking")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model name"
    )
    search_multiplier: int = Field(default=3, description="Multiplier for search_k calculation")
    filter_search_multiplier: int = Field(default=4, description="Multiplier when filtering search results")


class LLMConfig(BaseModel):
    """Configuration for language model."""
    model_name: str = Field(
        default="google/flan-t5-base",
        description="Language model name"
    )
    max_length: int = Field(default=512, description="Maximum generation length")
    temperature: float = Field(default=0.4, description="Generation temperature")
    device: str = Field(default="cpu", description="Device for computation")


class AnswerGenerationConfig(BaseModel):
    """Configuration for answer generation tuning parameters."""
    # Context preparation
    max_context_length: int = Field(default=4000, description="Maximum context length in characters")
    max_sources: int = Field(default=10, description="Maximum number of sources to use for context")
    
    # LLM generation parameters
    max_new_tokens: int = Field(default=300, description="Maximum new tokens to generate")
    min_new_tokens: int = Field(default=50, description="Minimum new tokens to generate")
    generation_temperature: float = Field(default=0.2, description="Temperature for text generation")
    repetition_penalty: float = Field(default=1.3, description="Repetition penalty for generation")
    
    # Fallback generation parameters
    max_fallback_sentences: int = Field(default=6, description="Maximum sentences in fallback generation")
    min_answer_length: int = Field(default=200, description="Minimum answer length to trigger additional context")
    additional_sentences: int = Field(default=3, description="Additional sentences to add for short answers")
    
    # Cache settings
    max_cache_size: int = Field(default=100, description="Maximum number of cached responses")
    
    # Concurrency settings
    max_concurrent_requests: int = Field(default=2, description="Maximum concurrent LLM requests to prevent tensor conflicts")
    
    # Summary generation
    summary_max_input_length: int = Field(default=1000, description="Maximum input length for summary generation")
    summary_max_sentences: int = Field(default=3, description="Maximum sentences in summary fallback")


class CLIConfig(BaseModel):
    """Configuration for CLI interfaces."""
    max_display_sources: int = Field(default=3, description="Maximum number of sources to display")
    content_snippet_length: int = Field(default=200, description="Maximum content snippet length")
    max_error_display: int = Field(default=10, description="Maximum number of errors to display")


class APIConfig(BaseModel):
    """Configuration for API service."""
    default_port: int = Field(default=8000, description="Default API server port")
    webhook_timeout: float = Field(default=30.0, description="Webhook request timeout in seconds")
    default_search_limit: int = Field(default=10, description="Default search results limit")


class WebhookConfig(BaseModel):
    """Configuration for webhook receiver."""
    default_port: int = Field(default=8001, description="Default webhook receiver port")
    min_port: int = Field(default=1, description="Minimum allowed port number")
    max_port: int = Field(default=65535, description="Maximum allowed port number")


class RAWEConfig(BaseModel):
    """Main RAWE system configuration."""
    # Component configurations
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    answer_generation: AnswerGenerationConfig = Field(default_factory=AnswerGenerationConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    
    # Paths
    data_dir: str = Field(default="data", description="Data directory")
    raw_data_dir: str = Field(default="data/raw", description="Raw data directory")
    processed_data_dir: str = Field(default="data/processed", description="Processed data directory")
    logs_dir: str = Field(default="logs", description="Logs directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        for dir_path in [self.data_dir, self.raw_data_dir, self.processed_data_dir, 
                        self.logs_dir, self.vector_store.storage_path]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = RAWEConfig()


def get_config() -> RAWEConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> RAWEConfig:
    """Update the global configuration."""
    global config
    config = RAWEConfig(**{**config.dict(), **kwargs})
    return config