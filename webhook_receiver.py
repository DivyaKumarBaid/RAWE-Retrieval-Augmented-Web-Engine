#!/usr/bin/env python3
"""
Webhook Receiver for RAWE System

This script receives webhook notifications from the RAWE API service
and logs the ingestion and query completion metrics.

Usage:
    python webhook_receiver.py --port 8001
    python webhook_receiver.py --port 8001 --log-file webhooks.log
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, Request
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text
import uvicorn

from rag_system.config import get_config


console = Console()
app = FastAPI(title="RAWE System Webhook Receiver", version="1.0.0")

# Global configuration
config = {
    "log_file": None,
    "show_json": True
}


def log_webhook(payload: Dict[str, Any], source_ip: str):
    """
    Log webhook payload.
    
    Args:
        payload: Webhook payload
        source_ip: Source IP address
    """
    timestamp = datetime.now().isoformat()
    
    # Console output
    console.print(f"\n[bold blue]Webhook Received at {timestamp}[/bold blue]")
    console.print(f"[dim]Source IP: {source_ip}[/dim]")
    
    # Extract key information
    operation = payload.get('operation', 'unknown')
    status = payload.get('status', 'unknown')
    
    # Status color
    status_color = "green" if status == "success" else "red" if status == "failed" else "yellow"
    
    console.print(f"[bold]Operation:[/bold] {operation}")
    console.print(f"[bold]Status:[/bold] [{status_color}]{status}[/{status_color}]")
    
    # Show error if present
    if 'error_message' in payload:
        console.print(f"[bold red]Error:[/bold red] {payload['error_message']}")
    
    # Show metrics if present
    if 'metrics' in payload and payload['metrics']:
        metrics = payload['metrics']
        
        if operation == "ingestion":
            console.print("\n[bold yellow]Ingestion Metrics:[/bold yellow]")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if key == 'errors' and isinstance(value, list):
                        console.print(f"  {key}: {len(value)} error(s)")
                    else:
                        console.print(f"  {key}: {value}")
        
        elif operation == "batch_query":
            console.print("\n[bold yellow]Query Metrics:[/bold yellow]")
            if isinstance(metrics, dict) and 'summary' in metrics:
                summary = metrics['summary']
                for key, value in summary.items():
                    if isinstance(value, float):
                        console.print(f"  {key}: {value:.2f}")
                    else:
                        console.print(f"  {key}: {value}")
                
                # Show number of results
                if 'results' in metrics:
                    results_count = len(metrics['results'])
                    console.print(f"  results_returned: {results_count}")
    
    # Show full JSON if requested
    if config["show_json"]:
        console.print("\n[bold]Full Payload:[/bold]")
        console.print(Panel(JSON(json.dumps(payload, indent=2)), border_style="dim"))
    
    # Log to file if configured
    if config["log_file"]:
        log_entry = {
            "timestamp": timestamp,
            "source_ip": source_ip,
            "payload": payload
        }
        
        try:
            with open(config["log_file"], "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            console.print(f"[red]Error writing to log file: {e}[/red]")


@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Receive webhook notifications.
    """
    try:
        # Get payload
        payload = await request.json()
        
        # Get source IP
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        source_ip = forwarded_for.split(",")[0] if forwarded_for else client_ip
        
        # Log the webhook
        log_webhook(payload, source_ip)
        
        return {"status": "received", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        console.print(f"[red]Error processing webhook: {e}[/red]")
        return {"status": "error", "message": str(e)}, 400


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAWE System Webhook Receiver",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "/webhook"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def print_startup_info(port: int):
    """Print startup information."""
    console.print(Panel.fit(
        f"[bold green]RAWE System Webhook Receiver[/bold green]\n"
        f"[dim]Listening for webhook notifications[/dim]\n\n"
        f"[bold]Webhook URL:[/bold] http://localhost:{port}/webhook\n"
        f"[bold]Health Check:[/bold] http://localhost:{port}/health\n"
        f"[bold]Log File:[/bold] {config['log_file'] or 'Console only'}",
        title="Server Started",
        border_style="green"
    ))
    
    console.print("\n[dim]Waiting for webhooks... (Press Ctrl+C to stop)[/dim]\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Webhook receiver for RAWE system notifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webhook_receiver.py --port 8001
  python webhook_receiver.py --port 8001 --log-file webhooks.log
  python webhook_receiver.py --port 8001 --no-json
        """
    )
    
    rag_config = get_config()
    parser.add_argument(
        '--port',
        type=int,
        default=rag_config.webhook.default_port,
        help=f'Port to listen on (default: {rag_config.webhook.default_port})'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='File to log webhook payloads (optional)'
    )
    
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Hide full JSON payload in console output'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    args = parser.parse_args()
    
    # Update global config
    config["log_file"] = args.log_file
    config["show_json"] = not args.no_json
    
    # Validate port
    rag_config = get_config()
    if not (rag_config.webhook.min_port <= args.port <= rag_config.webhook.max_port):
        console.print(f"[red]Error: Port must be between {rag_config.webhook.min_port} and {rag_config.webhook.max_port}[/red]")
        sys.exit(1)
    
    # Print startup info
    print_startup_info(args.port)
    
    try:
        # Start server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",  # Reduce uvicorn logging
            access_log=False      # Disable access logs
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Webhook receiver stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error starting webhook receiver: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()