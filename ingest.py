#!/usr/bin/env python3
"""
RAWE System Ingestion Script

This script crawls the TransFi website, extracts content from Products and Solutions pages,
processes the content into chunks, generates embeddings, and builds a vector index.

Usage:
    python ingest.py --url 'https://www.transfi.com'
    python ingest.py --url 'https://www.transfi.com' --max-depth 2 --max-pages 50
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional
import argparse
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from rag_system import RAWESystem, get_config, update_config
from rag_system.models import IngestionMetrics


console = Console()


def setup_logging(log_level: str = "INFO"):
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


async def run_ingestion(url: str, 
                       max_depth: Optional[int] = None,
                       max_pages: Optional[int] = None,
                       concurrent_requests: Optional[int] = None) -> IngestionMetrics:
    """
    Run the ingestion process.
    
    Args:
        url: Starting URL for crawling
        max_depth: Maximum crawling depth
        max_pages: Maximum number of pages to crawl
        concurrent_requests: Number of concurrent requests
        
    Returns:
        Ingestion metrics
    """
    # Update configuration if provided
    config_updates = {}
    if max_depth is not None:
        config_updates['crawler'] = {'max_depth': max_depth}
    if max_pages is not None:
        if 'crawler' not in config_updates:
            config_updates['crawler'] = {}
        config_updates['crawler']['max_pages'] = max_pages
    if concurrent_requests is not None:
        if 'crawler' not in config_updates:
            config_updates['crawler'] = {}
        config_updates['crawler']['concurrent_requests'] = concurrent_requests
    
    if config_updates:
        update_config(**config_updates)
    
    # Initialize RAWE system
    rawe_system = RAWESystem()
    
    console.print(Panel.fit(
        "[bold blue]RAWE System Ingestion[/bold blue]\n"
        f"[green]Target URL:[/green] {url}\n"
        f"[green]Max Depth:[/green] {get_config().crawler.max_depth}\n"
        f"[green]Max Pages:[/green] {get_config().crawler.max_pages}\n"
        f"[green]Concurrent Requests:[/green] {get_config().crawler.concurrent_requests}",
        title="Configuration"
    ))
    
    # Start ingestion with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Add tasks
        main_task = progress.add_task("[cyan]Initializing RAWE system...", total=100)
        
        try:
            # Initialize system
            await rawe_system.initialize()
            progress.update(main_task, advance=10, description="[cyan]System initialized")
            
            # Run ingestion
            progress.update(main_task, description="[yellow]Starting website crawling...")
            metrics = await rawe_system.ingest_website(url)
            progress.update(main_task, advance=90, description="[green]Ingestion completed!")
            
            return metrics
            
        except Exception as e:
            progress.update(main_task, description=f"[red]Error: {str(e)}")
            logging.error(f"Ingestion failed: {e}", exc_info=True)
            raise


def print_metrics(metrics: IngestionMetrics):
    """Print comprehensive ingestion metrics."""
    console.print("\n")
    console.print(Panel.fit(
        Text(metrics.print_summary(), style="bold green"),
        title="[bold blue]Ingestion Metrics[/bold blue]",
        border_style="green"
    ))
    
    # Additional detailed metrics
    if metrics.errors:
        config = get_config()
        max_errors = config.cli.max_error_display
        console.print("\n[bold red]Errors Encountered:[/bold red]")
        for i, error in enumerate(metrics.errors[:max_errors], 1):  # Show first N errors from config
            console.print(f"  {i}. {error}")
        if len(metrics.errors) > max_errors:
            console.print(f"  ... and {len(metrics.errors) - max_errors} more errors")


def create_enhanced_parser():
    """Create an enhanced argument parser with better help formatting."""
    parser = argparse.ArgumentParser(
        description="🔥 RAWE System Ingestion Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 Examples:
  Basic ingestion:
    python ingest.py --url 'https://www.transfi.com'
  
  Custom depth and page limits:
    python ingest.py --url 'https://www.transfi.com' --max-depth 2 --max-pages 50
  
  Performance tuning:
    python ingest.py --url 'https://www.transfi.com' --concurrent-requests 3 --log-level DEBUG
  
  Large website ingestion:
    python ingest.py --url 'https://www.transfi.com' --max-depth 3 --max-pages 200 --concurrent-requests 5

📝 Notes:
  • The script will crawl the website, extract content, generate embeddings, and build a vector index
  • Crawling is async-first with configurable concurrency for optimal performance
  • All data is stored locally and can be used for subsequent queries
  • Use --log-level DEBUG to see detailed progress information
        """
    )
    return parser


def main():
    """Main function."""
    parser = create_enhanced_parser()
    
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        metavar='URL',
        help='🌐 Starting URL for crawling (e.g., https://www.transfi.com)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        metavar='N',
        help=f'📊 Maximum crawling depth (default: {get_config().crawler.max_depth})'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        metavar='N',
        help=f'📄 Maximum number of pages to crawl (default: {get_config().crawler.max_pages})'
    )
    
    parser.add_argument(
        '--concurrent-requests',
        type=int,
        metavar='N',
        help=f'⚡ Number of concurrent requests (default: {get_config().crawler.concurrent_requests})'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        metavar='LEVEL',
        help='📝 Logging level - DEBUG for detailed output (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        console.print("[red]Error: URL must start with http:// or https://[/red]")
        sys.exit(1)
    
    # Print startup banner
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔥 RAWE System Ingestion[/bold cyan]\n"
        "[dim]Async-first document ingestion with comprehensive metrics[/dim]\n"
        "[dim]Powered by FAISS vector search, sentence transformers, and LLM generation[/dim]",
        style="cyan",
        title="[bold white]RAWE Ingestion Pipeline[/bold white]"
    ))
    
    start_time = time.time()
    
    try:
        # Run ingestion
        metrics = asyncio.run(run_ingestion(
            url=args.url,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            concurrent_requests=args.concurrent_requests
        ))
        
        # Print results
        print_metrics(metrics)
        
        # Success message
        total_time = time.time() - start_time
        console.print(f"\n[bold green]✓ Ingestion completed successfully in {total_time:.1f}s![/bold green]")
        
        # Print storage info
        config = get_config()
        console.print(f"\n[dim]Data stored in:[/dim]")
        console.print(f"  [dim]Raw HTML:[/dim] {config.raw_data_dir}")
        console.print(f"  [dim]Processed data:[/dim] {config.processed_data_dir}")
        console.print(f"  [dim]Vector store:[/dim] {config.vector_store.storage_path}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Ingestion interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Ingestion failed: {e}[/red]")
        logging.error("Ingestion failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()