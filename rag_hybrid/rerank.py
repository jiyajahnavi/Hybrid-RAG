from typing import List, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

from rag_hybrid.config import CROSS_ENCODER_MODEL_NAME, TOP_K_RERANK
from rag_hybrid.utils import get_logger
from rag_hybrid.storage import storage

logger = get_logger(__name__)

class Reranker:
    def __init__(self):
        logger.info(f"Loading Cross-Encoder Model: {CROSS_ENCODER_MODEL_NAME}")
        self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512)
        
    def rerank(self, query: str, fused_results: List[Tuple[int, float]], top_k: int = TOP_K_RERANK) -> List[Tuple[int, float]]:
        """
        Reranks the fused results using a cross-encoder model.
        
        Args:
            query: The user search query.
            fused_results: List of (chunk_id, fused_score) tuples from previous step.
            top_k: Number of final results to return.
            
        Returns:
            List of (chunk_id, cross_encoder_score) tuples, sorted highest to lowest.
        """
        if not fused_results:
            return []
            
        query_chunk_pairs = []
        chunk_ids = []
        
        for chunk_id, _ in fused_results:
            chunk = storage.get_chunk(chunk_id)
            if chunk:
                query_chunk_pairs.append([query, chunk["text"]])
                chunk_ids.append(chunk_id)
                
        if not query_chunk_pairs:
            return []
            
        # Predict scores
        cross_scores = self.model.predict(query_chunk_pairs)
        
        # Combine ids with new scores
        scored_results = list(zip(chunk_ids, cross_scores))
        
        # Sort by cross encoder score decreasing
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:top_k]

# Global reranker instance
reranker = Reranker()
