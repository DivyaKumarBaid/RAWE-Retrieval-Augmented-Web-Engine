"""
FastAPI service for the RAWE system.

This service provides REST API endpoints for ingestion and querying.
It supports async ingestion with webhook callbacks and concurrent query processing.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import httpx
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import RAWESystem, get_config
from rag_system.models import QueryResult, IngestionMetrics, WebhookPayload


# Pydantic models for API
class IngestionRequest(BaseModel):
    """Request model for ingestion endpoint."""
    urls: List[str] = Field(..., description="List of URLs to ingest")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    max_depth: Optional[int] = Field(None, description="Maximum crawling depth")
    max_pages: Optional[int] = Field(None, description="Maximum number of pages")
    
    @validator('urls')
    def validate_urls(cls, v):
        if not v:
            raise ValueError("At least one URL must be provided")
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL: {url}")
        return v


class QueryRequest(BaseModel):
    """Request model for single query endpoint."""
    question: str = Field(..., description="Question to answer")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class BatchQueryRequest(BaseModel):
    """Request model for batch query endpoint."""
    questions: List[str] = Field(..., description="List of questions to answer")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve per question")
    concurrent: bool = Field(True, description="Whether to process questions concurrently")
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question must be provided")
        return [q.strip() for q in v if q.strip()]


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint."""
    message: str
    task_id: Optional[str] = None
    estimated_time: Optional[str] = None


class SystemStatsResponse(BaseModel):
    """Response model for system stats endpoint."""
    status: str
    vector_store: Dict[str, Any]
    system_info: Dict[str, Any]
    timestamp: datetime


# Global RAWE system instance
rag_system: Optional[RAWESystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global rag_system
    
    # Startup
    logging.info("Initializing RAWE system...")
    rag_system = RAWESystem()
    await rag_system.initialize()
    logging.info("RAWE system initialized successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down RAWE system...")
    # Cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="RAWE System API",
    description="REST API for the TransFi RAWE system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def send_webhook(url: str, payload: Dict[str, Any]) -> bool:
    """
    Send webhook notification.
    
    Args:
        url: Webhook URL
        payload: Payload to send
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config = get_config()
        async with httpx.AsyncClient(timeout=config.api.webhook_timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logging.info(f"Webhook sent successfully to {url}")
            return True
    except Exception as e:
        logging.error(f"Failed to send webhook to {url}: {e}")
        return False


async def run_ingestion_task(request: IngestionRequest):
    """
    Background task for ingestion.
    
    Args:
        request: Ingestion request
    """
    start_time = time.time()
    
    try:
        # Process each URL (for now, just handle the first one)
        # In a full implementation, you might want to handle multiple URLs
        primary_url = request.urls[0]
        
        # Fix for TransFi - use www subdomain as it's the working variant
        if primary_url.startswith('https://transfi.com'):
            primary_url = primary_url.replace('https://transfi.com', 'https://www.transfi.com')
            logging.info(f"Converted URL to working variant: {primary_url}")
        
        # Update configuration if provided
        config_updates = {}
        if request.max_depth is not None:
            config_updates['crawler'] = {'max_depth': request.max_depth}
        if request.max_pages is not None:
            if 'crawler' not in config_updates:
                config_updates['crawler'] = {}
            config_updates['crawler']['max_pages'] = request.max_pages
        
        if config_updates:
            from rag_system.config import update_config
            update_config(**config_updates)
        
        # Run ingestion
        metrics = await rag_system.ingest_website(primary_url)
        
        # Send success webhook if provided
        if request.callback_url:
            webhook_payload = WebhookPayload(
                operation="ingestion",
                status="success",
                metrics=metrics.dict(),
                timestamp=datetime.utcnow()
            )
            # Convert to dict with proper datetime serialization
            payload_dict = webhook_payload.dict()
            payload_dict['timestamp'] = webhook_payload.timestamp.isoformat()
            await send_webhook(request.callback_url, payload_dict)
        
        logging.info(f"Ingestion completed successfully in {time.time() - start_time:.1f}s")
        
    except Exception as e:
        logging.error(f"Ingestion failed: {e}", exc_info=True)
        
        # Send failure webhook if provided
        if request.callback_url:
            webhook_payload = WebhookPayload(
                operation="ingestion",
                status="failed",
                error_message=str(e),
                timestamp=datetime.utcnow()
            )
            # Convert to dict with proper datetime serialization
            payload_dict = webhook_payload.dict()
            payload_dict['timestamp'] = webhook_payload.timestamp.isoformat()
            await send_webhook(request.callback_url, payload_dict)


async def run_batch_query_task(request: BatchQueryRequest):
    """
    Background task for batch queries.
    
    Args:
        request: Batch query request
    """
    start_time = time.time()
    
    try:
        # Process questions
        if request.concurrent:
            results = await rag_system.batch_query(request.questions, request.top_k)
        else:
            # Process sequentially
            results = []
            for question in request.questions:
                result = await rag_system.query(question, request.top_k)
                results.append(result)
        
        # Prepare results for webhook
        results_data = [result.dict() for result in results]
        
        # Calculate summary metrics
        total_time = time.time() - start_time
        successful = len([r for r in results if 'error' not in r.metadata])
        
        metrics = {
            'total_questions': len(request.questions),
            'successful_questions': successful,
            'failed_questions': len(request.questions) - successful,
            'total_processing_time': total_time,
            'average_time_per_question': total_time / len(request.questions),
            'concurrent_processing': request.concurrent
        }
        
        # Send success webhook if provided
        if request.callback_url:
            webhook_payload = WebhookPayload(
                operation="batch_query",
                status="success",
                metrics={
                    'summary': metrics,
                    'results': results_data
                },
                timestamp=datetime.utcnow()
            )
            # Convert to dict with proper datetime serialization
            payload_dict = webhook_payload.dict()
            payload_dict['timestamp'] = webhook_payload.timestamp.isoformat()
            await send_webhook(request.callback_url, payload_dict)
        
        logging.info(f"Batch query completed successfully in {total_time:.1f}s")
        
    except Exception as e:
        logging.error(f"Batch query failed: {e}", exc_info=True)
        
        # Send failure webhook if provided
        if request.callback_url:
            webhook_payload = WebhookPayload(
                operation="batch_query",
                status="failed",
                error_message=str(e),
                timestamp=datetime.utcnow()
            )
            # Convert to dict with proper datetime serialization
            payload_dict = webhook_payload.dict()
            payload_dict['timestamp'] = webhook_payload.timestamp.isoformat()
            await send_webhook(request.callback_url, payload_dict)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "RAWE System API", "version": "1.0.0", "status": "running"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAWE system not initialized"
        )
    
    stats = await rag_system.get_system_stats()
    
    return {
        "status": "healthy",
        "vector_store_documents": stats['vector_store']['num_documents'],
        "vector_store_chunks": stats['vector_store']['num_vectors'],
        "timestamp": datetime.utcnow()
    }


@app.post("/api/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_documents(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Start document ingestion process.
    
    This endpoint starts ingestion as a background task and returns immediately.
    If a callback_url is provided, a webhook will be sent when ingestion completes.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAWE system not initialized"
        )
    
    # Start background task
    background_tasks.add_task(run_ingestion_task, request)
    
    return IngestionResponse(
        message="Ingestion started",
        estimated_time="This may take several minutes depending on the number of pages"
    )


@app.post("/api/query", response_model=QueryResult, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Answer a single question using the RAWE system.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAWE system not initialized"
        )
    
    try:
        result = await rag_system.query(request.question, request.top_k)
        return result
    except Exception as e:
        logging.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/api/query/batch", tags=["Query"])
async def batch_query_documents(request: BatchQueryRequest, background_tasks: BackgroundTasks):
    """
    Process multiple questions.
    
    If callback_url is provided, processing will be done in background and results
    will be sent via webhook. Otherwise, results are returned synchronously.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAWE system not initialized"
        )
    
    try:
        if request.callback_url:
            # Async processing with webhook
            background_tasks.add_task(run_batch_query_task, request)
            return {
                "message": "Batch query started",
                "questions_count": len(request.questions),
                "callback_url": request.callback_url
            }
        else:
            # Synchronous processing
            if request.concurrent:
                results = await rag_system.batch_query(request.questions, request.top_k)
            else:
                results = []
                for question in request.questions:
                    result = await rag_system.query(question, request.top_k)
                    results.append(result)
            
            # Calculate summary metrics
            successful = len([r for r in results if 'error' not in r.metadata])
            total_time = sum(r.processing_time for r in results)
            
            return {
                "results": results,
                "summary": {
                    "total_questions": len(request.questions),
                    "successful_questions": successful,
                    "failed_questions": len(request.questions) - successful,
                    "total_processing_time": total_time,
                    "average_time_per_question": total_time / len(request.questions),
                    "concurrent_processing": request.concurrent
                }
            }
    except Exception as e:
        logging.error(f"Batch query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query processing failed: {str(e)}"
        )


@app.get("/api/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats():
    """Get comprehensive system statistics."""
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAWE system not initialized"
        )
    
    stats = await rag_system.get_system_stats()
    
    return SystemStatsResponse(
        status="operational",
        vector_store=stats['vector_store'],
        system_info={
            'embedding_model': stats['embedding_model'],
            'answer_generator': stats['answer_generator'],
            'config': stats['config']
        },
        timestamp=datetime.utcnow()
    )


@app.get("/api/documents/search", tags=["Documents"])
async def search_documents(query: str, top_k: Optional[int] = None):
    """Search for documents without generating an answer."""
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAWE system not initialized"
        )
    
    # Use default from config if not provided
    if top_k is None:
        config = get_config()
        top_k = config.api.default_search_limit
    
    try:
        documents = await rag_system.search_documents(query, top_k)
        return {
            "query": query,
            "documents": [doc.dict() for doc in documents],
            "count": len(documents)
        }
    except Exception as e:
        logging.error(f"Document search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {str(e)}"
        )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.api.default_port,
        reload=True,
        log_level="info"
    )