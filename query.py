#!/usr/bin/env python3
"""
RAWE System Query Script

This script answers questions using the RAWE system with support for single questions
and batch processing with concurrent execution.

Usage:
    # Single question
    python query.py --question "What is BizPay?"
    
    # Multiple questions from file
    python query.py --questions questions.txt
    
    # Batch processing (concurrent)
    python query.py --questions questions.txt --concurrent
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import argparse
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from rag_system import RAWESystem, get_config
from rag_system.models import QueryResult


console = Console()


def setup_logging(log_level: str = "INFO"):
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def load_questions_from_file(file_path: str) -> List[str]:
    """
    Load questions from a text file.
    
    Args:
        file_path: Path to the questions file
        
    Returns:
        List of questions
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    questions.append(line)
            return questions
    except FileNotFoundError:
        console.print(f"[red]Error: Questions file '{file_path}' not found[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading questions file: {e}[/red]")
        sys.exit(1)


def format_sources(sources, max_sources: Optional[int] = None) -> str:
    """
    Format sources for display.
    
    Args:
        sources: List of RetrievalResult objects
        max_sources: Maximum number of sources to display
        
    Returns:
        Formatted sources string
    """
    if not sources:
        return "No sources found."
    
    # Get max_sources from config if not provided
    if max_sources is None:
        config = get_config()
        max_sources = config.cli.max_display_sources
    
    formatted_sources = []
    
    for i, source in enumerate(sources[:max_sources], 1):
        # Get document info
        doc_title = "Unknown"
        doc_url = "Unknown"
        
        if source.document:
            doc_title = source.document.title or "Untitled"
            doc_url = source.document.url
        elif hasattr(source.chunk, 'metadata'):
            doc_title = source.chunk.metadata.get('document_title', 'Unknown')
            doc_url = source.chunk.metadata.get('document_url', 'Unknown')
        
        # Get content snippet
        content = source.chunk.content
        config = get_config()
        max_length = config.cli.content_snippet_length
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        source_text = f"  {i}. {doc_title} - {doc_url}\n     Snippet: \"{content}\""
        formatted_sources.append(source_text)
    
    return "\n".join(formatted_sources)


def print_query_result(result: QueryResult, show_metrics: bool = True):
    """
    Print a formatted query result.
    
    Args:
        result: Query result to print
        show_metrics: Whether to show detailed metrics
    """
    # Question header
    console.print(f"\n[bold cyan]Question:[/bold cyan] {result.query}")
    
    # Answer
    console.print(f"\n[bold green]Answer:[/bold green]")
    console.print(Panel.fit(
        Text(result.answer, style="white"),
        border_style="green"
    ))
    
    # Sources
    if result.sources:
        console.print(f"\n[bold yellow]Sources:[/bold yellow]")
        sources_text = format_sources(result.sources)
        console.print(sources_text)
    else:
        console.print("\n[dim]No sources available[/dim]")
    
    # Metrics
    if show_metrics and hasattr(result, 'metadata'):
        metrics_text = []
        
        # Basic metrics
        metrics_text.append(f"Total Latency: {result.processing_time:.1f}s")
        
        if 'retrieval_time' in result.metadata:
            metrics_text.append(f"Retrieval Time: {result.metadata['retrieval_time']:.1f}s")
        
        # Document counts
        if hasattr(result, 'metrics') and result.metrics:
            metrics_text.append(f"Documents Retrieved: {result.metrics.documents_retrieved}")
            metrics_text.append(f"Documents Used in Answer: {result.metrics.documents_used_in_answer}")
        else:
            metrics_text.append(f"Documents Retrieved: {len(result.sources)}")
        
        # Model info
        if 'model_name' in result.metadata:
            metrics_text.append(f"LLM Model: {result.metadata['model_name']}")
        
        console.print(f"\n[dim]Metrics:[/dim]")
        for metric in metrics_text:
            console.print(f"  [dim]{metric}[/dim]")


async def process_single_question(rawe_system: RAWESystem, question: str, top_k: Optional[int] = None) -> QueryResult:
    """
    Process a single question.
    
    Args:
        rawe_system: RAWE system instance
        question: Question to process
        top_k: Number of documents to retrieve
        
    Returns:
        Query result
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing question...", total=None)
        
        try:
            result = await rawe_system.query(question, top_k)
            progress.update(task, description="[green]Question processed!")
            return result
        except Exception as e:
            progress.update(task, description=f"[red]Error: {str(e)}")
            raise


async def process_batch_questions(rawe_system: RAWESystem, 
                                questions: List[str], 
                                concurrent: bool = False,
                                top_k: Optional[int] = None) -> List[QueryResult]:
    """
    Process multiple questions.
    
    Args:
        rawe_system: RAWE system instance
        questions: List of questions to process
        concurrent: Whether to process concurrently
        top_k: Number of documents to retrieve per question
        
    Returns:
        List of query results
    """
    if concurrent:
        # Process concurrently
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing {len(questions)} questions concurrently...", 
                total=len(questions)
            )
            
            try:
                results = await rawe_system.batch_query(questions, top_k)
                progress.update(task, completed=len(questions), description="[green]All questions processed!")
                return results
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                raise
    else:
        # Process sequentially
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing questions sequentially...", 
                total=len(questions)
            )
            
            for i, question in enumerate(questions, 1):
                try:
                    result = await rawe_system.query(question, top_k)
                    results.append(result)
                    progress.update(task, advance=1, description=f"[cyan]Processed {i}/{len(questions)} questions")
                except Exception as e:
                    # Create error result
                    error_result = QueryResult(
                        query=question,
                        answer=f"Error processing question: {str(e)}",
                        sources=[],
                        metadata={'error': str(e)},
                        processing_time=0.0
                    )
                    results.append(error_result)
                    progress.update(task, advance=1, description=f"[yellow]Error on question {i}")
            
            progress.update(task, description="[green]All questions processed!")
        
        return results


def print_batch_summary(results: List[QueryResult]):
    """Print a summary of batch processing results."""
    total_questions = len(results)
    successful = len([r for r in results if 'error' not in r.metadata])
    failed = total_questions - successful
    
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / total_questions if total_questions > 0 else 0
    
    console.print(f"\n[bold blue]Batch Processing Summary:[/bold blue]")
    console.print(f"  Total Questions: {total_questions}")
    console.print(f"  Successful: {successful}")
    console.print(f"  Failed: {failed}")
    console.print(f"  Total Processing Time: {total_time:.1f}s")
    console.print(f"  Average Time per Question: {avg_time:.1f}s")


async def main_async(args):
    """Main async function."""
    # Initialize RAWE system
    rawe_system = RAWESystem()
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]🤖 RAWE System Query Interface[/bold blue]\n"
        "[dim]Retrieval-Augmented Generation for intelligent Q&A[/dim]\n"
        "[dim]Powered by vector search, cross-encoder reranking, and LLM generation[/dim]",
        style="blue",
        title="[bold white]Question Answering System[/bold white]"
    ))
    
    # Initialize system
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initializing RAWE system...", total=None)
        
        try:
            await rawe_system.initialize()
            progress.update(task, description="[green]RAWE system initialized!")
        except Exception as e:
            progress.update(task, description=f"[red]Initialization failed: {str(e)}")
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Make sure you have run the ingestion script first:[/dim]")
            console.print("[dim]  python ingest.py --url 'https://www.transfi.com'[/dim]")
            sys.exit(1)
    
    # Check if vector store is empty
    stats = await rawe_system.get_system_stats()
    if stats['vector_store']['num_vectors'] == 0:
        console.print("\n[yellow]Warning: No documents found in the vector store.[/yellow]")
        console.print("[dim]Please run the ingestion script first:[/dim]")
        console.print("[dim]  python ingest.py --url 'https://www.transfi.com'[/dim]")
        sys.exit(1)
    
    console.print(f"\n[green]✓ Loaded {stats['vector_store']['num_vectors']} document chunks from {stats['vector_store']['num_documents']} documents[/green]")
    
    try:
        if args.question:
            # Single question mode
            console.print(f"\n[bold]Processing single question...[/bold]")
            result = await process_single_question(rawe_system, args.question, args.top_k)
            print_query_result(result, show_metrics=True)
            
        elif args.questions:
            # Batch mode
            questions = load_questions_from_file(args.questions)
            console.print(f"\n[bold]Processing {len(questions)} questions from file...[/bold]")
            
            if args.concurrent:
                console.print("[dim]Using concurrent processing[/dim]")
            else:
                console.print("[dim]Using sequential processing[/dim]")
            
            results = await process_batch_questions(
                rawe_system, 
                questions, 
                concurrent=args.concurrent,
                top_k=args.top_k
            )
            
            # Print results
            for i, result in enumerate(results, 1):
                console.print(f"\n[bold]{'='*60}[/bold]")
                console.print(f"[bold]Question {i} of {len(results)}[/bold]")
                print_query_result(result, show_metrics=not args.no_metrics)
            
            # Print summary
            print_batch_summary(results)
        
        else:
            console.print("[red]Error: Either --question or --questions must be provided[/red]")
            sys.exit(1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Query processing interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Query processing failed: {e}[/red]")
        logging.error("Query processing failed", exc_info=True)
        sys.exit(1)


def create_enhanced_parser():
    """Create an enhanced argument parser with better help formatting."""
    parser = argparse.ArgumentParser(
        description="🤖 RAWE System Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 Examples:
  Single question:
    python query.py --question "What is BizPay and its key features?"
    python query.py --question "In what countries is stripe available?"
  
  Multiple questions from file:
    python query.py --questions questions.txt
    python query.py --questions sample_questions.txt --no-metrics
  
  Concurrent batch processing:
    python query.py --questions questions.txt --concurrent
  
  Fine-tuning retrieval:
    python query.py --question "What services does TransFi offer?" --top-k 3
    python query.py --question "How does cross-border payment work?" --top-k 10

📄 Question File Format:
  Create a text file with one question per line:
    What is BizPay?
    How do I integrate TransFi APIs?
    # This is a comment and will be ignored
    What countries are supported?

🚀 Performance Tips:
  • Use --concurrent for faster batch processing
  • Increase --top-k for more comprehensive answers
  • Use --no-metrics to reduce output in batch mode
  • Use DEBUG log level to see retrieval details
        """
    )
    return parser


def main():
    """Main function."""
    parser = create_enhanced_parser()
    
    # Question input options (mutually exclusive)
    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument(
        '--question',
        type=str,
        metavar='TEXT',
        help='❓ Single question to process'
    )
    question_group.add_argument(
        '--questions',
        type=str,
        metavar='FILE',
        help='📄 Path to file containing questions (one per line, # for comments)'
    )
    
    # Processing options
    parser.add_argument(
        '--concurrent',
        action='store_true',
        help='🚀 Process multiple questions concurrently (only with --questions)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        metavar='N',
        help=f'🔍 Number of documents to retrieve per question (default: {get_config().retrieval.top_k})'
    )
    
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='📊 Hide detailed metrics in batch mode for cleaner output'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        metavar='LEVEL',
        help='📝 Logging level - DEBUG shows retrieval details (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate arguments
    if args.concurrent and not args.questions:
        console.print("[red]Error: --concurrent can only be used with --questions[/red]")
        sys.exit(1)
    
    if args.top_k is not None and args.top_k <= 0:
        console.print("[red]Error: --top-k must be a positive integer[/red]")
        sys.exit(1)
    
    # Run main async function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()