from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

from rag_hybrid.config import RRF_K
from rag_hybrid.utils import get_logger

logger = get_logger(__name__)

def reciprocal_rank_fusion(lexical_results: Dict[int, float], semantic_results: Dict[int, float], k: int = RRF_K) -> Dict[int, float]:
    """
    Applies Reciprocal Rank Fusion on lexical and semantic results.
    RRF score = sum(1 / (k + rank_i))
    
    Args:
        lexical_results: Dict of chunk_id to raw bm25 score
        semantic_results: Dict of chunk_id to cosine sim score
        k: RRF constant
        
    Returns:
        Dict of chunk_id to fused RRF score
    """
    # Sort results to get ranks
    lexical_sorted = sorted(lexical_results.items(), key=lambda x: x[1], reverse=True)
    semantic_sorted = sorted(semantic_results.items(), key=lambda x: x[1], reverse=True)
    
    fused_scores = defaultdict(float)
    
    # Add lexical scores
    for rank, (chunk_id, _) in enumerate(lexical_sorted, 1):
        fused_scores[chunk_id] += 1.0 / (k + rank)
        
    # Add semantic scores
    for rank, (chunk_id, _) in enumerate(semantic_sorted, 1):
        fused_scores[chunk_id] += 1.0 / (k + rank)
        
    return dict(fused_scores)

def min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalizes a dictionary of scores to [0, 1].
    """
    if not scores:
        return {}
        
    min_val = min(scores.values())
    max_val = max(scores.values())
    
    if max_val == min_val:
        # Avoid division by zero, all scores set to 1.0 if they are identical
        return {k: 1.0 for k in scores}
        
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

def weighted_fusion(lexical_results: Dict[int, float], semantic_results: Dict[int, float], alpha: float) -> Dict[int, float]:
    """
    Combines normalized scores using a weighted sum.
    Final_score = alpha * normalized_lexical + (1 - alpha) * normalized_semantic
    
    Args:
        lexical_results: Dict of chunk_id to raw bm25 score
        semantic_results: Dict of chunk_id to cosine sim score
        alpha: Weight for lexical score (0.0 to 1.0)
        
    Returns:
        Dict of chunk_id to fused weighted score
    """
    norm_lexical = min_max_normalize(lexical_results)
    norm_semantic = min_max_normalize(semantic_results)
    
    all_chunk_ids = set(norm_lexical.keys()).union(set(norm_semantic.keys()))
    
    fused_scores = {}
    for chunk_id in all_chunk_ids:
        lex_score = norm_lexical.get(chunk_id, 0.0)
        sem_score = norm_semantic.get(chunk_id, 0.0)
        
        fused_scores[chunk_id] = alpha * lex_score + (1.0 - alpha) * sem_score
        
    return fused_scores

def sort_fused_results(fused_results: Dict[int, float], top_k: int) -> List[Tuple[int, float]]:
    """
    Sorts fused results and returns the top k.
    """
    return sorted(fused_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
