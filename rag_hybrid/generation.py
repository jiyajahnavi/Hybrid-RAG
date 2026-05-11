import os
import google.generativeai as genai
from typing import List

from rag_hybrid.config import GEMINI_MODEL_NAME, GENERATION_TEMPERATURE, MAX_OUTPUT_TOKENS
from rag_hybrid.utils import get_logger
from rag_hybrid.storage import storage

logger = get_logger(__name__)

def generate_answer(api_key: str, query: str, retrieved_chunk_ids: List[int]) -> str:
    """
    Generates an answer using Gemini based solely on the provided chunk IDs.
    
    Args:
        api_key: The Gemini API key.
        query: The user's question.
        retrieved_chunk_ids: List of chunk IDs returned by the reranker.
        
    Returns:
        The generated answer string.
    """
    if not api_key or not api_key.strip():
        logger.error("API Key not provided.")
        return "Error: Gemini API Key is missing. Please provide it in the UI."
        
    genai.configure(api_key=api_key.strip())
    
    context_str = ""
    for chunk_id in retrieved_chunk_ids:
        chunk = storage.get_chunk(chunk_id)
        if chunk:
            context_str += f"[Chunk ID {chunk_id}]\n{chunk['text']}\n\n"
            
    if not context_str.strip():
        return "No context retrieved to answer the question."
        
    sys_instruction = "You are a precise question answering assistant. Use only the provided context."
    prompt = f"System:\n{sys_instruction}\n\nContext:\n{context_str}Question:\n{query}\n\nInstructions:\nCite chunk IDs. If insufficient information, say so."
    
    generation_config = genai.types.GenerationConfig(
        temperature=GENERATION_TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"Error during generation: {str(e)}"
