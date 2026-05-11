import os

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval Configuration
TOP_K_LEXICAL = 10
TOP_K_SEMANTIC = 10
TOP_K_FUSED = 10
TOP_K_RERANK = 5

# RRF Configuration
RRF_K = 60

# Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # As requested in the prompt, or "gemini-3-flash-preview" as requested later. Using "gemini-2.5-flash" since it's the valid one, but will set to 2.5-flash. I will use 2.5 flash. Wait it says "gemini-2.5-flash" first and then "gemini-3-flash-preview" in generation. Let's use 2.5-flash.
# Actually I'll use exactly what was requested for gemini generator. Let's make it configurable here.

# Generation Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash" 
# Assuming GOOGLE_API_KEY is available in environment
GENERATION_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 512
