#!/usr/bin/env python3
"""
RAWE System API Launcher

This script launches the FastAPI service for the RAWE system.

Usage:
    python api.py
    uvicorn api:app --port 8000
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the FastAPI app
from api.main import app
from rag_system.config import get_config

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.api.default_port,
        log_level="info"
    )