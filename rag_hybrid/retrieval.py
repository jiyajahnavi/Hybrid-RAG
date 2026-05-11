import numpy as np
from typing import Dict

from rag_hybrid.indexing import indexer
from rag_hybrid.config import TOP_K_LEXICAL, TOP_K_SEMANTIC
from rag_hybrid.utils import get_logger

logger = get_logger(__name__)

def retrieve_lexical(query: str, top_k: int = TOP_K_LEXICAL) -> Dict[int, float]:
    """
    Performs BM25 lexical search.
    
    Args:
        query: User search query.
        top_k: Number of results to return.
        
    Returns:
        Dict mapping chunk_id to its BM25 score.
    """
    if not indexer.has_index() or indexer.bm25_index is None:
        logger.warning("Index not built yet.")
        return {}

    tokenized_query = query.split()
    scores = indexer.bm25_index.get_scores(tokenized_query)
    
    # Get top_k indices corresponding to highest scores
    top_n = np.argsort(scores)[::-1][:top_k]
    
    results = {}
    for idx in top_n:
        if scores[idx] > 0: # Only include if there is some match
            chunk_id = indexer.index_to_chunk_id[idx]
            results[chunk_id] = float(scores[idx])
            
    return results

def retrieve_semantic(query: str, top_k: int = TOP_K_SEMANTIC) -> Dict[int, float]:
    """
    Performs FAISS semantic vector search.
    
    Args:
        query: User search query.
        top_k: Number of results to return.
        
    Returns:
        Dict mapping chunk_id to its cosine similarity score.
    """
    if not indexer.has_index() or indexer.vector_index is None:
        logger.warning("Index not built yet.")
        return {}

    # Encode query, convert to numpy, normalize
    query_embedding = indexer.encoder.encode([query], convert_to_numpy=True)
    import faiss
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = indexer.vector_index.search(query_embedding, top_k)
    
    results = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1: # FAISS returns -1 if not enough results
            chunk_id = indexer.index_to_chunk_id[idx]
            results[chunk_id] = float(score)
            
    return results
