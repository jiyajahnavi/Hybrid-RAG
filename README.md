# Hybrid RAG Lab

A production-quality Hybrid Retrieval-Augmented Generation (RAG) system using a modular architecture. This system allows you to build a powerful knowledge base from PDFs, raw text, and web URLs, and query it using a combination of lexical (BM25) and semantic (FAISS vector) search, followed by cross-encoder reranking and Google Gemini-2.5-flash generation.

## Features

- **Multi-Modal Ingestion**: Upload PDFs, paste raw text, or input blog URLs. The system automatically extracts, cleans, and chunks the content with intelligent overlap.
- **Hybrid Retrieval Strategy**: 
  - **Lexical Search (BM25)**: Exact keyword matching.
  - **Semantic Vector Search (FAISS)**: Deep contextual embedding matching using `sentence-transformers/all-MiniLM-L6-v2`.
- **Intelligent Fusion**:
  - **Reciprocal Rank Fusion (RRF)**: A state-of-the-art technique for mathematically combining rankings without score tuning.
  - **Weighted Fusion Mode**: Adjust the "Alpha" slider to dynamically shift the weight between lexical and semantic focus.
- **Cross-Encoder Reranking**: Re-scores the top fused results using `cross-encoder/ms-marco-MiniLM-L-6-v2` to ensure maximal contextual relevance.
- **Visualizer Table**: Gain perfect insight into *why* chunks were generation candidates, seeing exact score breakdowns for BM25, FAISS, RRF, and the Reranker entirely natively in the UI.
- **Secure Gemini Generation**: Connects to `gemini-3-flash-preview` for answer generation with strict prompting parameters to minimize hallucination. The API key is entered at runtime and never saved to disk.


## Hybrid RAG Architecture

![Hybrid RAG Architecture](Hybrid_RAG%20architecture.png)

## System Architecture

The project maintains a strict, modular separation of concerns designed for future extensibility:

```text
rag_hybrid/
│
├── app.py          # Gradio Web UI and main application entry point
├── ingestion.py    # Document parsing (pdfplumber, bs4) and chunking logic
├── indexing.py     # Index building for BM25 and FAISS FlatIP (cosine similarity)
├── retrieval.py    # Independent lexical and semantic search executors
├── fusion.py       # RRF and Weighted fusion combination equations
├── rerank.py       # ms-marco cross-encoder deep contextual rescoring
├── generation.py   # Connection to Google Generative AI (Gemini)
├── storage.py      # In-memory document storage and chunk retrieval
├── config.py       # Centralized hyperparameters (Chunk dimensions, top_k values)
└── utils.py        # Centralized logging and helper functions
```

## Setup & Installation

**Prerequisites:** Python 3.8+ 

1. **Clone the Repository** and open a terminal in the project root.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Get a Gemini API Key**:
   Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to generate an API key. 

## Usage

Start the system by running the main module:

```bash
python -m rag_hybrid.app
```

The Gradio web interface will boot up locally (usually on `http://127.0.0.1:7860`).


## Demo

![Hybrid RAG Demo](hybrid_RAG_demo.mp4)

### 1. Indexing Documents
Navigate to the **Indexing** tab. You can add as many PDFs, text snippets, and URLs as you like. Hitting "Index" will execute extraction, chunking, and dual-index (BM25 + FAISS) construction. 

### 2. Querying and Visualizing
Navigate to the **Query** tab. 
1. **Input API Key**: Enter your Gemini API Key securely.
2. **Select Fusion Mode**: Choose between RRF or Weighted Fusion (and tune the alpha slider).
3. **Ask**: Submit your question. 
4. **Insights**: The system will display the generated answer, the exact document chunks it grounded the answer on, and a **Retrieval Visualizer** data table showing the scoring lifecycle of the extracted text.

## Configuration Tuning
All system hyperparameters can be adjusted within `rag_hybrid/config.py`.

- `CHUNK_SIZE` and `CHUNK_OVERLAP` (Default: 500, 100)
- `TOP_K_LEXICAL` and `TOP_K_SEMANTIC` (Default: 10, 10)
- `TOP_K_FUSED` to send to reranker (Default: 10)
- `TOP_K_RERANK` to send to Gemini (Default: 5)
- `RRF_K` constant (Default: 60)