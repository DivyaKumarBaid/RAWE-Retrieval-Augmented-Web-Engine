#!/usr/bin/env python3
"""
Interactive RAWE System CLI
A user-friendly launcher for the RAWE system with guided workflows.
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.align import Align

console = Console()

def show_welcome():
    """Show welcome screen with ASCII art and introduction."""
    console.clear()

    welcome_text = """
[bold cyan]
 ██████╗  █████╗ ██╗    ██╗███████╗
 ██╔══██╗██╔══██╗██║    ██║██╔════╝
 ██████╔╝███████║██║ █╗ ██║█████╗
 ██╔══██╗██╔══██║██║███╗██║██╔══╝
 ██║  ██║██║  ██║╚███╔███╔╝███████╗
 ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝
[/bold cyan]

[bold white]Retrieval-Augmented Web Engine[/bold white]
[dim]Intelligent document ingestion and question answering[/dim]
"""
    
    console.print(Panel(
        Align.center(welcome_text),
        style="cyan",
        title="[bold white]Welcome to RAWE CLI[/bold white]"
    ))

def show_main_menu():
    """Show the main menu options."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="bold cyan")
    table.add_column("Description", style="white")
    
    table.add_row("1", "🔥 Ingest Website - Crawl and process documents")
    table.add_row("2", "🤖 Ask Questions - Interactive Q&A mode")
    table.add_row("3", "📄 Batch Questions - Process questions from file")
    table.add_row("4", "⚙️  System Status - Check current system state")
    table.add_row("5", "📖 Show Help - Display detailed help information")
    table.add_row("6", "🚪 Exit - Quit the application")
    
    console.print()
    console.print(Panel(
        table,
        title="[bold white]Main Menu[/bold white]",
        style="blue"
    ))

def run_ingestion():
    """Handle website ingestion workflow."""
    console.print("\n[bold cyan]🔥 Website Ingestion Workflow[/bold cyan]")
    
    # Get URL
    url = Prompt.ask("\n[yellow]Enter website URL[/yellow]", default="https://www.transfi.com")
    
    # Advanced options
    if Confirm.ask("\n[yellow]Configure advanced options?[/yellow]", default=False):
        max_depth = Prompt.ask("[yellow]Maximum crawling depth[/yellow]", default="5")
        max_pages = Prompt.ask("[yellow]Maximum pages to crawl[/yellow]", default="150")
        concurrent = Prompt.ask("[yellow]Concurrent requests[/yellow]", default="8")
        log_level = Prompt.ask("[yellow]Log level[/yellow]", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
        
        cmd = [
            sys.executable, "ingest.py",
            "--url", url,
            "--max-depth", max_depth,
            "--max-pages", max_pages,
            "--concurrent-requests", concurrent,
            "--log-level", log_level
        ]
    else:
        cmd = [sys.executable, "ingest.py", "--url", url]
    
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Ingestion failed with exit code {e.returncode}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Ingestion interrupted by user[/yellow]")

def run_interactive_qa():
    """Handle interactive Q&A workflow."""
    console.print("\n[bold cyan]🤖 Interactive Q&A Mode[/bold cyan]")
    console.print("[dim]Type 'quit' or 'exit' to return to main menu[/dim]")
    
    while True:
        console.print()
        question = Prompt.ask("[yellow]Enter your question[/yellow]")
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        # Advanced options
        if Confirm.ask("[yellow]Configure retrieval options?[/yellow]", default=False):
            top_k = Prompt.ask("[yellow]Number of documents to retrieve[/yellow]", default="30")
            cmd = [sys.executable, "query.py", "--question", question, "--top-k", top_k]
        else:
            cmd = [sys.executable, "query.py", "--question", question]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"\n[red]Query failed with exit code {e.returncode}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Query interrupted by user[/yellow]")

def run_batch_questions():
    """Handle batch question processing workflow."""
    console.print("\n[bold cyan]📄 Batch Question Processing[/bold cyan]")
    
    # Check if sample file exists
    sample_file = Path("sample_questions.txt")
    if sample_file.exists():
        use_sample = Confirm.ask(f"\n[yellow]Use sample questions file ({sample_file})?[/yellow]", default=True)
        if use_sample:
            questions_file = str(sample_file)
        else:
            questions_file = Prompt.ask("[yellow]Enter path to questions file[/yellow]")
    else:
        questions_file = Prompt.ask("[yellow]Enter path to questions file[/yellow]")
    
    # Processing options
    concurrent = Confirm.ask("[yellow]Use concurrent processing (faster)?[/yellow]", default=True)
    show_metrics = Confirm.ask("[yellow]Show detailed metrics?[/yellow]", default=False)
    
    cmd = [sys.executable, "query.py", "--questions", questions_file]
    
    if concurrent:
        cmd.append("--concurrent")
    
    if not show_metrics:
        cmd.append("--no-metrics")
    
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Batch processing failed with exit code {e.returncode}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing interrupted by user[/yellow]")

def show_system_status():
    """Show system status and configuration."""
    console.print("\n[bold cyan]⚙️ System Status[/bold cyan]")
    
    # Check if key files exist
    status_table = Table(show_header=True, header_style="bold yellow")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", justify="center")
    status_table.add_column("Details", style="dim")
    
    # Check ingestion script
    ingest_exists = Path("ingest.py").exists()
    status_table.add_row(
        "Ingestion Script",
        "✅ Available" if ingest_exists else "❌ Missing",
        "ingest.py" if ingest_exists else "File not found"
    )
    
    # Check query script
    query_exists = Path("query.py").exists()
    status_table.add_row(
        "Query Script",
        "✅ Available" if query_exists else "❌ Missing",
        "query.py" if query_exists else "File not found"
    )
    
    # Check sample questions
    sample_exists = Path("sample_questions.txt").exists()
    status_table.add_row(
        "Sample Questions",
        "✅ Available" if sample_exists else "ℹ️  Optional",
        "sample_questions.txt" if sample_exists else "Not found (optional)"
    )
    
    # Check data directories
    data_dir = Path("data")
    status_table.add_row(
        "Data Directory",
        "✅ Available" if data_dir.exists() else "⚠️  Missing",
        f"{data_dir} ({'exists' if data_dir.exists() else 'will be created'})"
    )
    
    # Check vector store
    vector_store = Path("data/vector_store")
    status_table.add_row(
        "Vector Store",
        "✅ Ready" if vector_store.exists() else "⚠️  Empty",
        f"{vector_store} ({'populated' if vector_store.exists() else 'run ingestion first'})"
    )
    
    console.print(status_table)
    
    # Show configuration hint
    console.print(f"\n[dim]Configuration file: src/rag_system/config.py[/dim]")
    console.print(f"[dim]Logs directory: logs/[/dim]")

def show_help():
    """Show detailed help information."""
    console.print("\n[bold cyan]📖 Help Information[/bold cyan]")
    
    help_content = """
[bold white]Getting Started:[/bold white]
1. First, run [cyan]Ingest Website[/cyan] to crawl and process documents
2. Then use [cyan]Ask Questions[/cyan] for interactive Q&A
3. Or use [cyan]Batch Questions[/cyan] to process multiple questions

[bold white]Workflow:[/bold white]
[cyan]Ingestion[/cyan] → [cyan]Vector Store[/cyan] → [cyan]Question Answering[/cyan]

[bold white]Advanced Usage:[/bold white]
• Use DEBUG log level to see detailed processing information
• Increase concurrent requests for faster crawling (be respectful)
• Adjust top-k parameter for more/fewer retrieved documents
• Create custom question files (one question per line)

[bold white]Files and Directories:[/bold white]
• [cyan]ingest.py[/cyan] - Website crawling and ingestion
• [cyan]query.py[/cyan] - Question answering interface
• [cyan]data/[/cyan] - Processed documents and vector store
• [cyan]logs/[/cyan] - System logs and debug information
• [cyan]src/rag_system/config.py[/cyan] - System configuration

[bold white]Sample Question File Format:[/bold white]
```
# This is a comment
What is BizPay?
How does cross-border payment work?
# Another comment
What countries are supported?
```

[bold white]Performance Tips:[/bold white]
• Use concurrent processing for batch questions
• Increase max-pages for comprehensive ingestion
• Use higher top-k values for complex questions
• Monitor logs/ directory for debugging
"""
    
    console.print(Panel(
        help_content,
        title="[bold white]RAWE System Help[/bold white]",
        style="blue"
    ))

def main():
    """Main interactive CLI function."""
    show_welcome()
    
    while True:
        show_main_menu()
        
        try:
            choice = Prompt.ask("\n[bold yellow]Select an option[/bold yellow]", choices=["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                run_ingestion()
            elif choice == "2":
                run_interactive_qa()
            elif choice == "3":
                run_batch_questions()
            elif choice == "4":
                show_system_status()
            elif choice == "5":
                show_help()
            elif choice == "6":
                console.print("\n[green]Thank you for using RAWE CLI! 👋[/green]")
                break
            
            if choice in ["1", "2", "3"]:
                input("\n[dim]Press Enter to continue...[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]An error occurred: {e}[/red]")

if __name__ == "__main__":
    main()