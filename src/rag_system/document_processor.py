"""Document processing module for content extraction and cleaning."""

import hashlib
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Document, TextChunk
from .config import get_config


class DocumentProcessor:
    """Handles document extraction, cleaning, and chunking."""
    
    def __init__(self):
        self.config = get_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_content_from_html(self, html: str, url: str) -> Dict[str, str]:
        """
        Extract clean content from HTML using readability-lxml and BeautifulSoup.
        
        Args:
            html: Raw HTML content
            url: Source URL
            
        Returns:
            Dictionary with extracted content
        """
        try:
            # Use readability to extract main content
            doc = ReadabilityDocument(html)
            main_content = doc.content()
            title = doc.title() or ""
            
            # Parse with BeautifulSoup for additional cleaning
            soup = BeautifulSoup(main_content, 'lxml')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'form', 'button', 'input']):
                element.decompose()
            
            # Extract text content
            content = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Extract short description (first paragraph or sentence)
            short_desc = self._extract_short_description(content)
            
            return {
                'title': self._clean_text(title),
                'content': content,
                'short_description': short_desc,
                'cleaned_html': str(soup)
            }
            
        except Exception as e:
            # Fallback to basic BeautifulSoup parsing
            soup = BeautifulSoup(html, 'lxml')
            title = soup.title.string if soup.title else ""
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
                
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'title': self._clean_text(title),
                'content': content,
                'short_description': content[:self.config.processing.short_description_length] + "..." if len(content) > self.config.processing.short_description_length else content,
                'cleaned_html': str(soup)
            }
    
    def _extract_short_description(self, content: str) -> str:
        """Extract a short description from content."""
        sentences = re.split(r'[.!?]+', content)
        max_length = self.config.processing.short_description_length
        if sentences:
            # Use first sentence up to configurable length
            first_sentence = sentences[0].strip()
            if len(first_sentence) > max_length:
                return first_sentence[:max_length] + "..."
            return first_sentence + "."
        return content[:max_length] + "..." if len(content) > max_length else content
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\']', '', text)
        
        return text
    
    def create_document(self, url: str, html: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Create a Document object from HTML content.
        
        Args:
            url: Source URL
            html: Raw HTML content
            metadata: Additional metadata
            
        Returns:
            Document object
        """
        # Generate unique ID based on URL
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Extract content
        extracted = self.extract_content_from_html(html, url)
        
        # Create document
        doc = Document(
            id=doc_id,
            url=url,
            title=extracted['title'],
            content=extracted['content'],
            short_description=extracted['short_description'],
            raw_html=html,
            metadata=metadata or {}
        )
        
        return doc
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        """
        Split document content into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of text chunks
        """
        if not document.content.strip():
            return []
        
        # Split content into chunks
        texts = self.text_splitter.split_text(document.content)
        
        chunks = []
        start_char = 0
        
        for i, text in enumerate(texts):
            # Skip empty or too short chunks
            if len(text.strip()) < self.config.processing.min_chunk_size:
                continue
            
            # Find the actual position in the original text
            end_char = start_char + len(text)
            
            # Create chunk ID
            chunk_id = f"{document.id}_chunk_{i}"
            
            chunk = TextChunk(
                id=chunk_id,
                document_id=document.id,
                content=text.strip(),
                start_char=start_char,
                end_char=end_char,
                metadata={
                    'document_title': document.title,
                    'document_url': document.url,
                    'chunk_index': i,
                    'total_chunks': len(texts)
                }
            )
            
            chunks.append(chunk)
            start_char = end_char
        
        return chunks
    
    async def process_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Process multiple documents into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of all text chunks
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def extract_page_links(self, html: str, base_url: str, 
                          allowed_domains: Optional[List[str]] = None) -> List[str]:
        """
        Extract links from a page that match the allowed domains.
        Prioritizes "Learn More" and other relevant internal links as per requirements.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            allowed_domains: List of allowed domains
            
        Returns:
            List of absolute URLs
        """
        if allowed_domains is None:
            allowed_domains = self.config.crawler.allowed_domains
        
        soup = BeautifulSoup(html, 'lxml')
        links = []
        priority_links = []  # For "Learn More" and high-priority links
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True).lower()
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Parse URL
            parsed = urlparse(absolute_url)
            
            # Filter by domain - if no allowed domains specified, allow all domains
            if not allowed_domains or any(domain in parsed.netloc for domain in allowed_domains):
                # Remove fragment and query parameters for cleaner URLs
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                if clean_url not in links and clean_url != base_url:
                    # Prioritize "Learn More" and relevant internal links
                    if any(keyword in link_text for keyword in [
                        'learn more', 'read more', 'discover', 'explore',
                        'get started', 'view details', 'find out more'
                    ]):
                        priority_links.append(clean_url)
                    else:
                        links.append(clean_url)
        
        # Return priority links first, then regular links
        return priority_links + links
    
    def is_product_or_solution_page(self, url: str, content: str) -> bool:
        """
        Check if a page is related to products or solutions.
        
        Args:
            url: Page URL
            content: Page content
            
        Returns:
            True if page is relevant
        """
        url_lower = url.lower()
        content_lower = content.lower()
        
        # Use configurable patterns from config
        url_indicators = self.config.crawler.url_indicators
        content_indicators = self.config.crawler.content_indicators
        
        # If no indicators specified, consider all pages as relevant
        if not url_indicators and not content_indicators:
            return True
        
        # Check URL
        if url_indicators and any(indicator in url_lower for indicator in url_indicators):
            return True
        
        # Check content (more lenient for comprehensive coverage)
        if content_indicators:
            indicator_count = sum(1 for indicator in content_indicators 
                                if indicator in content_lower)
        else:
            indicator_count = 0
        
        # Also check for TransFi-specific content
        transfi_specific = [
            'transfi', 'bizpay', 'stablecoin', 'crypto payment',
            'blockchain payment', 'digital asset', 'web3 payment'
        ]
        
        transfi_count = sum(1 for term in transfi_specific 
                          if term in content_lower)
        
        # Accept if either condition is met, or if no content indicators are specified
        return (not content_indicators) or indicator_count >= self.config.processing.min_content_indicators or transfi_count >= self.config.processing.min_transfi_indicators