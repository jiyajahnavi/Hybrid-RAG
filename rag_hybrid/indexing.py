import faiss
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

from rag_hybrid.config import EMBEDDING_MODEL_NAME
from rag_hybrid.utils import get_logger
from rag_hybrid.storage import storage

logger = get_logger(__name__)

class Indexer:
    """
    Handles BM25 and FAISS vector indexing.
    Caches embeddings to avoid recomputation if chunks are unchanged.
    """
    def __init__(self):
        logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # BM25 Lexical Index
        self.bm25_index: BM25Okapi = None
        
        # FAISS Vector Index (Inner Product)
        # We need to normalize embeddings before adding to IndexFlatIP to get Cosine Similarity
        self.vector_index: faiss.IndexFlatIP = None
        
        # Mapping from index row to global chunk_id
        self.index_to_chunk_id: List[int] = []
        
        # Local cache for embeddings: chunk_id -> np.ndarray
        self.embedding_cache: Dict[int, np.ndarray] = {}
        
        # Flag to indicate if we need to rebuild the index
        self.is_built = False

    def build_index(self):
        """
        Builds the BM25 and FAISS indices from the document chunks in storage.
        """
        chunks = storage.get_all_chunks()
        if not chunks:
            logger.warning("No chunks found in storage to index.")
            return

        logger.info("Rebuilding indices...")
        
        texts = []
        tokenized_texts = []
        self.index_to_chunk_id = []
        
        embeddings_to_compute = []
        compute_chunk_ids = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            
            self.index_to_chunk_id.append(chunk_id)
            texts.append(text)
            tokenized_texts.append(text.split()) # Basic whitespace tokenization for BM25
            
            if chunk_id not in self.embedding_cache:
                embeddings_to_compute.append(text)
                compute_chunk_ids.append(chunk_id)

        # 1. Build BM25
        logger.info(f"Building BM25 with {len(tokenized_texts)} documents.")
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        # 2. Compute missing embeddings and cache them
        if embeddings_to_compute:
            logger.info(f"Computing embeddings for {len(embeddings_to_compute)} new chunks.")
            new_embeddings = self.encoder.encode(embeddings_to_compute, convert_to_numpy=True, normalize_embeddings=True)
            for cid, emb in zip(compute_chunk_ids, new_embeddings):
                self.embedding_cache[cid] = emb
        
        # 3. Build FAISS
        logger.info("Building FAISS index.")
        self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Collect all embeddings in order
        all_embeddings = np.array([self.embedding_cache[cid] for cid in self.index_to_chunk_id]).astype('float32')
        # Ensure normalization (though the encode method normalizes them, doing it here to be safe)
        faiss.normalize_L2(all_embeddings)
        self.vector_index.add(all_embeddings)
        
        self.is_built = True
        logger.info(f"Indexing complete. Indexed {len(self.index_to_chunk_id)} chunks.")

    def has_index(self) -> bool:
        return self.is_built

# Global indexer instance
indexer = Indexer()
