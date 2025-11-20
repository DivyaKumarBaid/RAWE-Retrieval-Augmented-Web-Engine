"""Async web crawler for scraping website content."""

import asyncio
import time
from typing import List, Set, Dict, Optional, Tuple
from urllib.parse import urlparse
import aiohttp
import aiofiles
from pathlib import Path
import json
from datetime import datetime

from .models import Document, CrawlStatus
from .document_processor import DocumentProcessor
from .config import get_config


class AsyncWebCrawler:
    """Async web crawler with domain limiting and depth control."""
    
    def __init__(self):
        self.config = get_config()
        self.document_processor = DocumentProcessor()
        self.session: Optional[aiohttp.ClientSession] = None
        self.crawled_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.crawl_stats: List[CrawlStatus] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start the aiohttp session."""
        timeout = aiohttp.ClientTimeout(total=self.config.crawler.timeout)
        headers = {
            'User-Agent': self.config.crawler.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Create SSL context that's more permissive with broader protocol support
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        # Support older TLS versions
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1
        ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(
                limit=self.config.crawler.concurrent_requests,
                ssl=ssl_context
            )
        )
    
    async def close_session(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> Tuple[Optional[str], CrawlStatus]:
        """
        Fetch a single page.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (HTML content, crawl status)
        """
        start_time = time.time()
        status = CrawlStatus(url=url, status="pending")
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    status.status = "success"
                    status.response_time = time.time() - start_time
                    status.content_length = len(content)
                    
                    # Save raw HTML
                    await self._save_raw_html(url, content)
                    
                    return content, status
                else:
                    status.status = "failed"
                    status.error_message = f"HTTP {response.status}"
                    status.response_time = time.time() - start_time
                    print(f"HTTP error {response.status} for {url}")
                    return None, status
                    
        except Exception as e:
            status.status = "failed"
            status.error_message = str(e)
            status.response_time = time.time() - start_time
            print(f"Error fetching {url}: {e}")
            return None, status
    
    async def _save_raw_html(self, url: str, content: str):
        """Save raw HTML content to disk."""
        # Create filename from URL
        parsed = urlparse(url)
        filename = f"{parsed.netloc}{parsed.path}".replace('/', '_').replace('\\', '_')
        if not filename.endswith('.html'):
            filename += '.html'
        
        # Save to raw data directory
        file_path = Path(self.config.raw_data_dir) / filename
        
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
        except Exception as e:
            print(f"Warning: Could not save raw HTML for {url}: {e}")
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling."""
        parsed = urlparse(url)
        
        # Check domain - more permissive matching
        # If no allowed domains specified, allow all domains
        if self.config.crawler.allowed_domains:
            domain_match = False
            for domain in self.config.crawler.allowed_domains:
                if domain in parsed.netloc or parsed.netloc.endswith(domain):
                    domain_match = True
                    break
            
            if not domain_match:
                print(f"Domain not allowed: {parsed.netloc} not in {self.config.crawler.allowed_domains}")
                return False
        
        # Skip certain file types
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico']
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip mailto and other non-http protocols
        if not parsed.scheme.startswith('http'):
            print(f"Invalid scheme: {parsed.scheme} for {url}")
            return False
        
        return True
    
    async def crawl_url(self, url: str, current_depth: int = 0) -> List[Document]:
        """
        Crawl a single URL and extract documents.
        
        Args:
            url: URL to crawl
            current_depth: Current crawling depth
            
        Returns:
            List of extracted documents
        """
        if url in self.crawled_urls or not self._is_valid_url(url):
            return []
        
        self.crawled_urls.add(url)
        
        # Fetch page
        html, status = await self.fetch_page(url)
        self.crawl_stats.append(status)
        
        if html is None:
            self.failed_urls.add(url)
            return []
        
        documents = []
        
        # Create document from page
        try:
            document = self.document_processor.create_document(
                url=url,
                html=html,
                metadata={
                    'crawl_depth': current_depth,
                    'crawl_timestamp': datetime.now().isoformat(),
                    'content_length': len(html),
                    'response_time': status.response_time
                }
            )
            
            # Only include if it's relevant to products/solutions
            if self.document_processor.is_product_or_solution_page(url, document.content):
                documents.append(document)
            
        except Exception as e:
            print(f"Error processing document from {url}: {e}")
            self.failed_urls.add(url)
        
        # Extract links for further crawling if within depth limit
        if current_depth < self.config.crawler.max_depth:
            try:
                links = self.document_processor.extract_page_links(
                    html, url, self.config.crawler.allowed_domains
                )
                
                # Crawl child pages with rate limiting
                child_tasks = []
                for link in links[:15]:  # Increased limit for better coverage
                    if link not in self.crawled_urls and len(self.crawled_urls) < self.config.crawler.max_pages:
                        child_tasks.append(self.crawl_url(link, current_depth + 1))
                
                if child_tasks:
                    # Add delay between requests
                    await asyncio.sleep(self.config.crawler.delay_between_requests)
                    
                    # Run child crawls concurrently
                    child_results = await asyncio.gather(*child_tasks, return_exceptions=True)
                    
                    # Collect results
                    for result in child_results:
                        if isinstance(result, list):
                            documents.extend(result)
                        elif isinstance(result, Exception):
                            print(f"Error in child crawl: {result}")
                            
            except Exception as e:
                print(f"Error extracting links from {url}: {e}")
        
        return documents
    
    async def crawl_website(self, start_url: str) -> List[Document]:
        """
        Crawl a website starting from a given URL.
        
        Args:
            start_url: Starting URL for crawling
            
        Returns:
            List of all extracted documents
        """
        if self.session is None:
            await self.start_session()
        
        print(f"Starting crawl from: {start_url}")
        print(f"Max depth: {self.config.crawler.max_depth}")
        print(f"Max pages: {self.config.crawler.max_pages}")
        print(f"Allowed domains: {self.config.crawler.allowed_domains}")
        
        # Reset state
        self.crawled_urls.clear()
        self.failed_urls.clear()
        self.crawl_stats.clear()
        
        # Get initial seed URLs (homepage)
        seed_urls = self._get_transfi_seed_urls(start_url)
        print(f"Starting with {len(seed_urls)} initial seed URLs...")
        
        # Dynamically discover product and solution URLs
        print("Discovering product and solution URLs from homepage navigation...")
        discovered_urls = await self._discover_product_solution_urls(start_url)
        
        # Combine initial seeds with discovered URLs
        all_seed_urls = seed_urls + discovered_urls
        print(f"Total seed URLs for comprehensive coverage: {len(all_seed_urls)}")
        for i, url in enumerate(all_seed_urls):
            print(f"  Seed URL {i+1}: {url}")
        
        all_documents = []
        
        # Crawl all seed URLs
        for i, seed_url in enumerate(all_seed_urls):
            if len(self.crawled_urls) < self.config.crawler.max_pages:
                print(f"Crawling seed URL {i+1}/{len(all_seed_urls)}: {seed_url}")
                documents = await self.crawl_url(seed_url, 0)
                all_documents.extend(documents)
                
                # Add small delay between seed URLs
                await asyncio.sleep(0.3)
        
        # Save crawl statistics
        await self._save_crawl_stats()
        
        print(f"Crawling completed. Found {len(all_documents)} relevant documents from {len(self.crawled_urls)} pages.")
        print(f"Failed to crawl {len(self.failed_urls)} pages.")
        
        return all_documents
    
    async def _discover_product_solution_urls(self, base_url: str) -> List[str]:
        """
        Dynamically discover all product and solution URLs by parsing the homepage navigation.
        This follows the requirement to scrape all pages under Products and Solutions sections,
        including subpages linked via "Learn More" or relevant internal links.
        """
        discovered_urls = []
        
        try:
            # Fetch the homepage to extract navigation links
            html, status = await self.fetch_page(base_url)
            if html is None:
                print(f"Failed to fetch homepage for link discovery: {status.error_message}")
                return discovered_urls
            
            # Extract all links from the homepage (prioritizes "Learn More" links)
            all_links = self.document_processor.extract_page_links(
                html, base_url, self.config.crawler.allowed_domains
            )
            
            # Filter for primary pages using configurable patterns
            primary_urls = []
            for link in all_links:
                link_lower = link.lower()
                # If no filters specified, consider all links as primary
                if not self.config.crawler.primary_path_filters:
                    primary_urls.append(link)
                    discovered_urls.append(link)
                    print(f"  Discovered primary (no filters): {link}")
                # Otherwise, check if any primary path filter matches
                elif any(path_filter in link_lower for path_filter in self.config.crawler.primary_path_filters):
                    primary_urls.append(link)
                    discovered_urls.append(link)
                    print(f"  Discovered primary: {link}")
            
            # For each primary product/solution page, discover child pages
            print("Discovering child pages and 'Learn More' links...")
            for primary_url in primary_urls[:self.config.crawler.max_primary_pages_to_crawl]:  # Limit to prevent excessive crawling
                try:
                    child_html, _ = await self.fetch_page(primary_url)
                    if child_html:
                        child_links = self.document_processor.extract_page_links(
                            child_html, primary_url, self.config.crawler.allowed_domains
                        )
                        
                        # Add child pages that are under the same product/solution path
                        for child_link in child_links[:self.config.crawler.max_child_links_per_page]:  # Limit child discovery
                            child_lower = child_link.lower()
                            primary_lower = primary_url.lower()
                            
                            # Check if child is under the same primary section using configurable patterns
                            same_section = False
                            # If no filters specified, consider all child links as same section
                            if not self.config.crawler.primary_path_filters:
                                same_section = True
                            else:
                                for path_filter in self.config.crawler.primary_path_filters:
                                    if path_filter in child_lower and path_filter in primary_lower:
                                        same_section = True
                                        break
                            
                            if child_link not in discovered_urls and same_section:
                                discovered_urls.append(child_link)
                                print(f"  Discovered child: {child_link}")
                        
                        # Small delay between child page fetches
                        await asyncio.sleep(0.2)
                
                except Exception as e:
                    print(f"Error discovering children for {primary_url}: {e}")
            
            print(f"Total discovered URLs: {len(discovered_urls)} (including child pages)")
            
        except Exception as e:
            print(f"Error during link discovery: {e}")
        
        return discovered_urls
    
    def _get_transfi_seed_urls(self, base_url: str) -> List[str]:
        """Generate seed URLs focusing on Products and Solutions sections as per requirements."""
        # Convert to working www variant for TransFi
        if base_url.startswith('https://transfi.com'):
            base_url = base_url.replace('https://transfi.com', 'https://www.transfi.com')
            print(f"Using working www variant: {base_url}")
        
        seed_urls = []
        
        # Add the base URL first
        if self._is_valid_url(base_url):
            seed_urls.append(base_url)
        
        return seed_urls
    
    async def _save_crawl_stats(self):
        """Save crawling statistics to file."""
        stats = {
            'crawl_timestamp': datetime.now().isoformat(),
            'total_urls_attempted': len(self.crawl_stats),
            'successful_crawls': len([s for s in self.crawl_stats if s.status == "success"]),
            'failed_crawls': len([s for s in self.crawl_stats if s.status == "failed"]),
            'crawled_urls': list(self.crawled_urls),
            'failed_urls': list(self.failed_urls),
            'detailed_stats': [status.dict() for status in self.crawl_stats],
            'config': {
                'max_depth': self.config.crawler.max_depth,
                'max_pages': self.config.crawler.max_pages,
                'allowed_domains': self.config.crawler.allowed_domains
            }
        }
        
        stats_file = Path(self.config.processed_data_dir) / f"crawl_stats_{int(time.time())}.json"
        
        try:
            async with aiofiles.open(stats_file, 'w') as f:
                await f.write(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"Warning: Could not save crawl stats: {e}")
    
    def get_crawl_metrics(self) -> Dict[str, any]:
        """Get crawling metrics."""
        successful_crawls = [s for s in self.crawl_stats if s.status == "success"]
        failed_crawls = [s for s in self.crawl_stats if s.status == "failed"]
        
        metrics = {
            'pages_scraped': len(successful_crawls),
            'pages_failed': len(failed_crawls),
            'total_pages_attempted': len(self.crawl_stats),
            'success_rate': len(successful_crawls) / len(self.crawl_stats) if self.crawl_stats else 0,
            'average_response_time': sum(s.response_time for s in successful_crawls) / len(successful_crawls) if successful_crawls else 0,
            'total_content_length': sum(s.content_length for s in successful_crawls if s.content_length),
            'errors': [s.error_message for s in failed_crawls if s.error_message]
        }
        
        return metrics