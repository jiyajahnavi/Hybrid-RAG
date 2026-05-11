from typing import Dict, Any, List

class DocumentStorage:
    """
    In-memory storage for document chunks.
    Maps chunk_id to its text and metadata.
    """
    def __init__(self):
        self.chunks: Dict[int, Dict[str, Any]] = {}
        self.next_chunk_id = 0

    def add_chunk(self, text: str, metadata: Dict[str, Any]) -> int:
        """
        Adds a new chunk to storage.
        
        Args:
            text: The chunk text.
            metadata: Metadata associated with the chunk.
            
        Returns:
            The chunk ID.
        """
        chunk_id = self.next_chunk_id
        self.chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata
        }
        self.next_chunk_id += 1
        return chunk_id

    def get_chunk(self, chunk_id: int) -> Dict[str, Any]:
        """
        Retrieves a chunk by ID.
        """
        return self.chunks.get(chunk_id)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Returns all chunks as a list.
        """
        return list(self.chunks.values())
    
    def clear(self):
        """
        Clears all chunks from storage.
        """
        self.chunks.clear()
        self.next_chunk_id = 0

# Global storage instance
storage = DocumentStorage()
