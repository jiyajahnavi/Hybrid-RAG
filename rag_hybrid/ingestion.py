import re
import requests
from bs4 import BeautifulSoup
import pdfplumber
from typing import List, Dict, Any, Optional

from rag_hybrid.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag_hybrid.utils import get_logger, normalize_text
from rag_hybrid.storage import storage

logger = get_logger(__name__)

def split_into_chunks(text: str, source: str, doc_name: str, page_number: Optional[int] = None) -> List[int]:
    """
    Splits text into paragraph-aware chunks and stores them.
    Approximates tokens by word count.
    
    Args:
        text (str): Raw text to chunk.
        source (str): Source type (e.g., 'PDF', 'URL', 'TEXT').
        doc_name (str): Name or path of the document.
        page_number (int, optional): Page number reference if applicable.
        
    Returns:
        List[int]: List of chunk_ids added to storage.
    """
    # Paragraph-aware split
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    current_chunk_words = []
    current_length = 0
    
    for para in paragraphs:
        para = normalize_text(para)
        if not para:
            continue
            
        words = para.split()
        para_len = len(words)
        
        # If adding this paragraph exceeds CHUNK_SIZE
        if current_length + para_len > CHUNK_SIZE and current_chunk_words:
            # Save the current chunk
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            # Determine overlap size in words
            overlap_words = current_chunk_words[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 and len(current_chunk_words) > CHUNK_OVERLAP else current_chunk_words
            current_chunk_words = overlap_words + words
            current_length = len(current_chunk_words)
        else:
            current_chunk_words.extend(words)
            current_length += para_len
            
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append(chunk_text)
        
    # Store chunks
    chunk_ids = []
    for chunk in chunks:
        metadata = {
            "source": source,
            "document_name": doc_name,
        }
        if page_number is not None:
            metadata["page_number"] = page_number
            
        chunk_id = storage.add_chunk(chunk, metadata)
        chunk_ids.append(chunk_id)
        
    logger.info(f"Chunked {doc_name} into {len(chunk_ids)} chunks.")
    return chunk_ids

def ingest_pdf(file_path: str, doc_name: str) -> List[int]:
    """
    Extracts text from a PDF, chunks it, and adds to storage.
    """
    logger.info(f"Ingesting PDF: {doc_name}")
    chunk_ids = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Treat each page's text as a segment to chunk to maintain page metadata
                    page_chunk_ids = split_into_chunks(text, source="PDF", doc_name=doc_name, page_number=i+1)
                    chunk_ids.extend(page_chunk_ids)
    except Exception as e:
        logger.error(f"Error ingesting PDF {file_path}: {e}")
    return chunk_ids

def ingest_url(url: str) -> List[int]:
    """
    Extracts content from a blog URL, cleans it, chunks it, and adds to storage.
    """
    logger.info(f"Ingesting URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove scripts, styles, nav, headers, footers
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
            
        # Get main text
        text = str(soup.get_text(separator="\n\n"))
        
        return split_into_chunks(text, source="URL", doc_name=url)
    except Exception as e:
        logger.error(f"Error ingesting URL {url}: {e}")
        return []

def ingest_text(text: str, doc_name: str = "Raw Text Input") -> List[int]:
    """
    Chunks raw text and adds to storage.
    """
    logger.info(f"Ingesting raw text: {doc_name}")
    return split_into_chunks(text, source="TEXT", doc_name=doc_name)
