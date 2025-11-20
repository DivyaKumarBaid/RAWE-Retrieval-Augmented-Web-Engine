# RAWE - Retrieval-Augmented Web Engine

A production-grade, async-first RAWE (Retrieval-Augmented Web Engine) system built for scraping website content and providing intelligent Q&A capabilities. This system combines web crawling, document processing, vector search, and local LLM-based answer generation.

## 🚀 Features

- **Async-First Architecture**: Built with asyncio for high-performance concurrent operations
- **Intelligent Web Crawling**: Smart discovery of product and solution pages with depth control
- **Advanced Document Processing**: Content extraction, cleaning, and intelligent chunking
- **Vector Search**: FAISS-powered similarity search with optional cross-encoder reranking
- **Local LLM Integration**: Uses local language models (T5-based) for answer generation
- **RESTful API**: FastAPI-based service with webhook support for async operations
- **Interactive CLI**: User-friendly guided workflows with real-time status checking
- **Direct CLI Scripts**: Advanced command-line tools for scripted operations
- **Concurrent Processing**: Optimized for batch operations with proper resource management
- **Comprehensive Metrics**: Detailed performance tracking and reporting
- **Webhook Integration**: Async notification system for long-running operations

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [API Service](#api-service)
- [Webhook Receiver](#webhook-receiver)
- [Data Flow](#data-flow)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🏃 Quick Start

### 1. Installation
```bash
# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directories (automatically created on first run)
mkdir -p data/raw data/processed data/vector_store logs
```

### 2. Run Interactive CLI (Recommended)
```bash
# Start the user-friendly interactive CLI
python rag_cli.py

# Follow the guided workflow:
# 1. Choose "Ingest Website" → Enter URL → Configure options
# 2. Choose "Ask Questions" → Interactive Q&A session  
# 3. Choose "System Status" → Check component health
```

### 3. OR Use Direct Scripts
```bash
# Direct ingestion (alternative to CLI)
python ingest.py --url https://www.transfi.com

# Direct querying (alternative to CLI)  
python query.py --question "What is BizPay and its key features?"

# Batch processing
python query.py --questions questions.txt --concurrent --output results.json
```

### 4. Start API Service
```bash
# Start the main API server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the API script
python api.py
```

### 5. Start Webhook Receiver (Optional)
```bash
# Start webhook receiver for async notifications
python webhook_receiver.py --port 8001
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for local LLM)
- 2GB+ disk space for data storage

### Step-by-Step Installation

#### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python  # Linux/Mac
where python   # Windows
```

#### 3. Install Dependencies
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Verify installation
pip list
```

#### 4. Create Data Directories
```bash
# Create required directories (optional - created automatically on first run)
mkdir -p data/raw data/processed data/vector_store logs
```

#### 4. Virtual Environment Management

```bash
# Activate virtual environment (when needed)
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Deactivate virtual environment (when done)
deactivate

# Update dependencies (when requirements.txt changes)
pip install -r requirements.txt --upgrade

# Check installed packages
pip list

# Create requirements.txt from current environment (if needed)
pip freeze > requirements.txt
```

### Environment Setup
```bash
# Set environment variables (optional)
export RAWE_DATA_DIR="./data"
export RAWE_LOG_LEVEL="INFO"
export RAWE_DEVICE="cpu"  # or "cuda" for GPU
```

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Crawler   │────│Document Processor│────│  Text Chunker   │
│                 │    │                  │    │                 │
│ • Async crawling│    │ • Content extract│    │ • Smart chunking│
│ • Domain filter │    │ • HTML cleaning  │    │ • Overlap logic │
│ • Rate limiting │    │ • Text normalize │    │ • Size control  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────────--┐
                    │Embedding Generator│
                    │                   │
                    │ • SentenceT5      │
                    │ • Batch process   │
                    │ • Async compute   │
                    └─────────────────--┘
                              │
                    ┌─────────────────┐
                    │  Vector Store   │
                    │                 │
                    │ • FAISS index   │
                    │ • Cosine sim    │
                    │ • Persistence   │
                    └─────────────────┘
                              │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │    Retriever    │────│Answer Generator │────│   Query Result  │
    │                 │    │                 │    │                 │
    │ • Vector search │    │ • Local LLM     │    │ • Structured    │
    │ • Reranking     │    │ • Context prep  │    │ • With sources  │
    │ • Filtering     │    │ • Fallback gen  │    │ • Metrics       │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Ingestion Flow**:
   ```
   URL → Web Crawling → Content Extraction → Text Chunking → 
   Embedding Generation → Vector Indexing → Storage
   ```

2. **Query Flow**:
   ```
   Question → Query Embedding → Vector Search → Reranking → 
   Context Preparation → LLM Generation → Answer + Sources
   ```

## ⚙️ Configuration

The system uses a comprehensive configuration system in `src/rag_system/config.py`. All settings are organized into logical groups:

### Core Configuration Groups

#### 1. Crawler Configuration (`CrawlerConfig`)
```python
crawler:
  max_depth: 5                    # Maximum crawling depth
  max_pages: 150                  # Maximum pages to crawl
  delay_between_requests: 0.8     # Rate limiting (seconds)
  concurrent_requests: 8          # Concurrent request limit
  timeout: 30                     # Request timeout
  user_agent: "RAWE-System/1.0"   # Custom user agent
  
  # Content Discovery
  primary_path_filters:           # URL patterns for primary content
    - "/products/"
    - "/solutions/"
  
  url_indicators:                 # URL keywords for relevance
    - "/product"
    - "/solution"
    - "bizpay"
    - "payout"
  
  content_indicators:             # Content keywords for filtering
    - "product"
    - "solution" 
    - "payment"
    - "fintech"
```

#### 2. Processing Configuration (`ProcessingConfig`)
```python
processing:
  chunk_size: 1000               # Text chunk size
  chunk_overlap: 200             # Overlap between chunks
  min_chunk_size: 100            # Minimum viable chunk size
  short_description_length: 200   # Max short description length
  min_content_indicators: 2       # Required content relevance score
```

#### 3. Embedding Configuration (`EmbeddingConfig`)
```python
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384                  # Embedding dimension
  batch_size: 32                 # Processing batch size
  device: "cpu"                  # "cpu" or "cuda"
```

#### 4. Vector Store Configuration (`VectorStoreConfig`)
```python
vector_store:
  index_type: "faiss"            # Vector index type
  similarity_metric: "cosine"     # Distance metric
  storage_path: "data/vector_store"
```

#### 5. Retrieval Configuration (`RetrievalConfig`)
```python
retrieval:
  top_k: 30                      # Initial retrieval count
  rerank_top_k: 20              # Post-rerank count
  use_reranking: true           # Enable cross-encoder reranking
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

#### 6. LLM Configuration (`LLMConfig`)
```python
llm:
  model_name: "google/flan-t5-large"
  max_length: 1024               # Generation length limit
  temperature: 0.3               # Generation randomness
  device: "cpu"                  # "cpu" or "cuda"
```

#### 7. Answer Generation Configuration (`AnswerGenerationConfig`)
```python
answer_generation:
  max_context_length: 4000       # Context window size
  max_sources: 10               # Sources to include
  max_new_tokens: 300           # Generated token limit
  generation_temperature: 0.2    # LLM temperature
  max_concurrent_requests: 2     # Concurrent LLM limit
```

### Configuration Methods

#### Get Configuration
```python
from src.rag_system.config import get_config

config = get_config()
print(f"Max pages: {config.crawler.max_pages}")
```

#### Update Configuration
```python
from src.rag_system.config import update_config

# Update specific settings
updated_config = update_config(
    crawler={'max_pages': 200, 'max_depth': 4},
    llm={'temperature': 0.5}
)
```

## 🖥️ CLI Usage

The system provides comprehensive CLI tools for all operations.

### Interactive RAWE CLI (`rag_cli.py`)

The RAWE CLI provides a user-friendly, interactive interface for all system operations with guided workflows and real-time status checking.

#### Starting the Interactive CLI
```bash
# Launch the interactive CLI
python rag_cli.py
```

#### Features Overview
The interactive CLI provides:

1. **🔥 Website Ingestion Workflow**
   - Guided URL input with validation
   - Advanced configuration options (depth, pages, concurrency)
   - Real-time progress feedback
   - Error handling and recovery suggestions

2. **🤖 Interactive Q&A Mode**
   - Continuous question-answering session
   - Dynamic retrieval parameter adjustment
   - Formatted output with sources and metrics
   - Easy exit commands (`quit`, `exit`, `q`)

3. **📄 Batch Question Processing**
   - Automatic detection of sample question files
   - File path validation and suggestions
   - Concurrent processing options
   - Detailed metrics reporting

4. **⚙️ System Status Dashboard**
   - Component availability checking
   - Data directory verification
   - Vector store status
   - Configuration file locations

5. **📖 Built-in Help System**
   - Comprehensive workflow guidance
   - Performance optimization tips
   - File format examples
   - Troubleshooting suggestions

#### Example Workflow
```bash
# 1. Start the CLI
python rag_cli.py

# 2. Choose option 1 (Ingest Website)
#    - Enter URL: https://www.transfi.com
#    - Configure advanced options if needed
#    - Watch real-time ingestion progress

# 3. Choose option 2 (Ask Questions)
#    - Enter: "What is BizPay?"
#    - View formatted answer with sources
#    - Continue asking questions or type 'quit'

# 4. Choose option 4 (System Status)
#    - View component status
#    - Check data directories
#    - Verify vector store readiness
```

#### Advanced Configuration in CLI
The interactive CLI allows real-time configuration of:

- **Crawling Parameters**: Depth, page limits, concurrency
- **Retrieval Options**: Top-k values, reranking settings  
- **Processing Modes**: Concurrent vs sequential processing
- **Output Formats**: Metrics display, verbosity levels
- **Logging Levels**: DEBUG, INFO, WARNING, ERROR

#### CLI vs Direct Script Usage
| Feature | Interactive CLI | Direct Scripts |
|---------|----------------|----------------|
| **Ease of Use** | ✅ Guided workflows | ❌ Manual parameters |
| **Status Checking** | ✅ Real-time dashboard | ❌ Manual verification |
| **Error Recovery** | ✅ Suggestions provided | ❌ Manual troubleshooting |
| **Batch Processing** | ✅ File auto-detection | ❌ Manual file paths |
| **Configuration** | ✅ Interactive prompts | ❌ Command-line flags |

### Direct Script Usage

### Ingestion CLI (`ingest.py`)

#### Basic Usage
```bash
# Ingest from URL
python ingest.py --url https://www.transfi.com

# With custom limits
python ingest.py --url https://www.transfi.com --max-pages 200 --max-depth 3

# Verbose output with timing
python ingest.py --url https://www.transfi.com --verbose
```

#### Advanced Options
```bash
# Custom output directory
python ingest.py --url https://www.transfi.com --output-dir ./custom_data

# Save raw HTML files
python ingest.py --url https://www.transfi.com --save-raw

# Custom user agent
python ingest.py --url https://www.transfi.com --user-agent "Custom Bot 1.0"

# Delay between requests
python ingest.py --url https://www.transfi.com --delay 1.0
```

#### Example Output
```
=== Ingestion Metrics ===
Total Time: 45.2s
Pages Scraped: 23
Pages Failed: 2
Total Chunks Created: 456
Total Tokens Processed: 125,340
Embedding Generation Time: 12.3s
Indexing Time: 2.1s
Average Scraping Time per Page: 1.8s
Errors: 0 error(s)
```

### Query CLI (`query.py`)

#### Single Questions
```bash
# Basic question
python query.py --question "What is BizPay?"

# With custom result count
python query.py --question "What is BizPay?" --top-k 5

# Verbose output with sources
python query.py --question "What is BizPay?" --verbose
```

#### Multiple Questions
```bash
# From file (one question per line)
python query.py --questions questions.txt

# Concurrent processing
python query.py --questions questions.txt --concurrent

# Save results to file
python query.py --questions questions.txt --output results.json

# Custom format
python query.py --questions questions.txt --output results.csv --format csv
```

#### Example Output
```
Question: What is BizPay and its key features?
Answer: BizPay is TransFi's comprehensive payment solution that enables businesses to accept and process payments across multiple channels. Key features include multi-currency support, real-time settlement, fraud protection, and seamless integration with existing business systems.

Sources:
  1. BizPay - https://www.transfi.com/products/bizpay
     Snippet: "BizPay enables businesses to accept payments in over 100 currencies..."
  2. Solutions Overview - https://www.transfi.com/solutions
     Snippet: "Our payment infrastructure supports real-time processing..."

Metrics:
  Total Latency: 2.4s
  Retrieval Time: 0.3s
  LLM Time: 2.0s
  Post-processing Time: 0.1s
  Documents Retrieved: 5
  Documents Used in Answer: 2
  Input Tokens: 1,234
  Output Tokens: 456
  Estimated Cost: $0.0089
```

#### Question File Format
```text
# questions.txt
What is BizPay?
How does TransFi's payment processing work?
What currencies does TransFi support?
What are the key features of TransFi's solutions?
```

## 🌐 API Service

The FastAPI-based service provides RESTful endpoints for all system operations.

### Starting the API Server

#### Method 1: Direct
```bash
python api.py
# Starts server on http://localhost:8000
```

#### Method 2: Uvicorn
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Method 3: Custom Configuration
```bash
# With custom settings
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --workers 4
```

### API Endpoints

#### Health Check
```http
GET /health
```

#### System Statistics
```http
GET /api/stats
```

#### Document Search (No Answer Generation)
```http
GET /api/documents/search?query=BizPay&top_k=10
```

#### Single Query
```http
POST /api/query
Content-Type: application/json

{
  "question": "What is BizPay?",
  "top_k": 5
}
```

#### Batch Query (Synchronous)
```http
POST /api/query/batch
Content-Type: application/json

{
  "questions": ["What is BizPay?", "How does payment processing work?"],
  "concurrent": true,
  "top_k": 5
}
```

#### Batch Query (Asynchronous with Webhook)
```http
POST /api/query/batch
Content-Type: application/json

{
  "questions": ["What is BizPay?", "How does payment processing work?"],
  "callback_url": "http://localhost:8001/webhook",
  "concurrent": true
}
```

#### Ingestion (Asynchronous)
```http
POST /api/ingest
Content-Type: application/json

{
  "urls": ["https://www.transfi.com"],
  "callback_url": "http://localhost:8001/webhook",
  "max_depth": 3,
  "max_pages": 150
}
```

## 📡 Webhook Receiver

The webhook receiver handles async notifications from long-running operations.

### Starting the Webhook Receiver

```bash
# Default port (8001)
python webhook_receiver.py

# Custom port
python webhook_receiver.py --port 8002

# With JSON formatting
python webhook_receiver.py --port 8001 --json

# Save to log file
python webhook_receiver.py --port 8001 --log-file webhooks.log
```

### Webhook Payload Structure

#### Ingestion Completion
```json
{
  "operation": "ingestion",
  "status": "success",
  "metrics": {
    "total_time": 45.2,
    "pages_scraped": 23,
    "pages_failed": 2,
    "total_chunks_created": 456,
    "total_tokens_processed": 125340,
    "embedding_generation_time": 12.3,
    "indexing_time": 2.1
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Batch Query Completion
```json
{
  "operation": "batch_query",
  "status": "success",
  "metrics": {
    "summary": {
      "total_questions": 5,
      "successful_questions": 5,
      "failed_questions": 0,
      "total_processing_time": 12.5,
      "average_time_per_question": 2.5
    },
    "results": [...]
  },
  "timestamp": "2024-01-15T10:35:00Z"
}
```

## 🔄 Data Flow

### Ingestion Process

1. **URL Input**: Provide starting URL(s)
2. **Smart Discovery**: System discovers product/solution pages from navigation
3. **Concurrent Crawling**: Async crawling with rate limiting and domain filtering
4. **Content Extraction**: Uses readability + BeautifulSoup for clean content
5. **Text Chunking**: Intelligent chunking with configurable size and overlap
6. **Embedding Generation**: Batch processing with sentence-transformers
7. **Vector Indexing**: FAISS indexing with cosine similarity
8. **Persistence**: Save to disk with metadata

### Query Process

1. **Question Input**: Single question or batch of questions
2. **Query Embedding**: Generate embedding for the question
3. **Vector Search**: FAISS similarity search with configurable top-k
4. **Reranking**: Optional cross-encoder reranking for better relevance
5. **Context Preparation**: Combine relevant chunks with source attribution
6. **LLM Generation**: Local T5 model generates contextual answer
7. **Post-processing**: Clean response and add citations
8. **Result Assembly**: Structured result with answer, sources, and metrics

## 🎯 Advanced Features

### Concurrent Processing

The system is optimized for concurrent operations:

```python
# Concurrent ingestion
results = await asyncio.gather(*[
    rag_system.ingest_website(url) for url in urls
])

# Concurrent queries with resource management
results = await rag_system.batch_query(questions, concurrent=True)
```

### Smart Content Discovery

Automatically discovers relevant pages:
- Follows navigation structure
- Prioritizes "Learn More" links
- Filters by URL patterns and content keywords
- Configurable discovery depth and breadth

### Fallback Mechanisms

Robust fallback systems:
- LLM failure → Extractive summarization
- Network issues → Retry with backoff
- Parsing errors → Alternative extraction methods

### Performance Optimization

- Async-first architecture
- Batch processing for embeddings
- Connection pooling for web requests
- Vector index caching
- Response caching with LRU eviction

### Metrics and Monitoring

Comprehensive metrics collection:
- Ingestion metrics (time, success rate, errors)
- Query metrics (latency, token usage, accuracy)
- System metrics (memory, disk usage)
- Performance metrics (throughput, concurrency)

## 📚 API Reference

### Curl Examples

#### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

#### 2. System Statistics
```bash
curl -X GET "http://localhost:8000/api/stats"
```

#### 3. Start Ingestion
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://www.transfi.com"],
    "callback_url": "http://localhost:8001/webhook",
    "max_depth": 3,
    "max_pages": 150
  }'
```

#### 4. Single Query
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is BizPay and its key features?",
    "top_k": 5
  }'
```

#### 5. Batch Query (Synchronous)
```bash
curl -X POST "http://localhost:8000/api/query/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is BizPay?",
      "How does TransFi payment processing work?",
      "What currencies does TransFi support?"
    ],
    "concurrent": true,
    "top_k": 5
  }'
```

#### 6. Batch Query (Asynchronous)
```bash
curl -X POST "http://localhost:8000/api/query/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is BizPay?",
      "How does TransFi payment processing work?",
      "What currencies does TransFi support?",
      "What are the key features of TransFi solutions?",
      "How does cross-border payment settlement work?"
    ],
    "callback_url": "http://localhost:8001/webhook",
    "concurrent": true,
    "top_k": 10
  }'
```

#### 7. Document Search
```bash
curl -X GET "http://localhost:8000/api/documents/search?query=payment%20processing&top_k=10"
```

#### 8. Advanced Ingestion with Custom Parameters
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://www.transfi.com",
      "https://www.transfi.com/products",
      "https://www.transfi.com/solutions"
    ],
    "callback_url": "http://localhost:8001/webhook",
    "max_depth": 4,
    "max_pages": 200
  }'
```

### Response Examples

#### Successful Query Response
```json
{
  "query": "What is BizPay and its key features?",
  "answer": "BizPay is TransFi's comprehensive payment solution that enables businesses to accept and process payments across multiple channels. Key features include multi-currency support with over 100 currencies, real-time settlement, advanced fraud protection, seamless API integration, and support for both traditional and digital payment methods including stablecoins.",
  "sources": [
    {
      "chunk": {
        "id": "doc123_chunk_0",
        "content": "BizPay enables businesses to accept payments in over 100 currencies with real-time settlement and advanced fraud protection...",
        "metadata": {
          "document_title": "BizPay - Business Payment Solutions",
          "document_url": "https://www.transfi.com/products/bizpay"
        }
      },
      "score": 0.92,
      "document": {
        "title": "BizPay - Business Payment Solutions",
        "url": "https://www.transfi.com/products/bizpay"
      }
    }
  ],
  "metadata": {
    "model_name": "google/flan-t5-large",
    "context_length": 2340,
    "generation_params": {
      "max_length": 1024,
      "temperature": 0.3
    }
  },
  "processing_time": 2.4,
  "metrics": {
    "total_latency": 2.4,
    "retrieval_time": 0.3,
    "llm_time": 2.0,
    "post_processing_time": 0.1,
    "documents_retrieved": 5,
    "documents_used_in_answer": 2,
    "input_tokens": 1234,
    "output_tokens": 456,
    "estimated_cost": 0.0
  }
}
```

#### Batch Query Response
```json
{
  "results": [
    {
      "query": "What is BizPay?",
      "answer": "BizPay is TransFi's business payment solution...",
      "sources": [...],
      "processing_time": 2.1
    },
    {
      "query": "How does payment processing work?",
      "answer": "TransFi's payment processing works through...",
      "sources": [...],
      "processing_time": 2.3
    }
  ],
  "summary": {
    "total_questions": 2,
    "successful_questions": 2,
    "failed_questions": 0,
    "total_processing_time": 4.4,
    "average_time_per_question": 2.2,
    "concurrent_processing": true
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Issue: Out of memory during embedding generation
# Solution: Reduce batch size in config
# Edit config.py: embedding.batch_size = 16 (default: 32)
```

#### 2. CUDA Errors
```bash
# Issue: CUDA out of memory
# Solution: Force CPU usage
python ingest.py --url https://www.transfi.com --device cpu
```

### Performance Tuning

#### For Speed
```python
# config.py adjustments
crawler.concurrent_requests = 16
crawler.delay_between_requests = 0.5
embedding.batch_size = 64
answer_generation.max_concurrent_requests = 4
```

#### For Accuracy
```python
# config.py adjustments
crawler.max_pages = 300
processing.chunk_size = 800
processing.chunk_overlap = 300
retrieval.top_k = 50
retrieval.rerank_top_k = 30
```

#### For Resource Conservation
```python
# config.py adjustments
crawler.concurrent_requests = 4
crawler.delay_between_requests = 1.0
embedding.batch_size = 16
answer_generation.max_concurrent_requests = 1
```

### Logging and Debugging

#### Enable Debug Logging
```bash
# Set environment variable
export RAWE_LOG_LEVEL=DEBUG
python ingest.py --url https://www.transfi.com

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check System Status
```bash
# CLI stats
python rag_cli.py stats

# API stats
curl -X GET "http://localhost:8000/api/stats"
```
