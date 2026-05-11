import gradio as gr
import pandas as pd
from typing import List, Tuple

from rag_hybrid.ingestion import ingest_pdf, ingest_url, ingest_text
from rag_hybrid.indexing import indexer
from rag_hybrid.retrieval import retrieve_lexical, retrieve_semantic
from rag_hybrid.fusion import reciprocal_rank_fusion, weighted_fusion, sort_fused_results
from rag_hybrid.rerank import reranker
from rag_hybrid.generation import generate_answer
from rag_hybrid.storage import storage
from rag_hybrid.config import TOP_K_FUSED
from rag_hybrid.utils import get_logger

logger = get_logger(__name__)

# --- Indexing Callbacks ---

def handle_index_pdf(pdf_files):
    if not pdf_files:
        return "No PDFs provided."
    count = 0
    for pdf in pdf_files:
        # pdf is a temp path in Gradio, but we can extract name
        name = getattr(pdf, "name", "uploaded_pdf")
        ids = ingest_pdf(pdf.name, name)
        count += len(ids)
    indexer.build_index()
    return f"Indexed {len(pdf_files)} PDFs, resulting in {count} chunks. Index built."

def handle_index_text(text: str):
    if not text.strip():
        return "Empty text."
    ids = ingest_text(text, "Raw Text")
    indexer.build_index()
    return f"Indexed text into {len(ids)} chunks. Index built."

def handle_index_url(url: str):
    if not url.strip():
        return "Empty URL."
    ids = ingest_url(url)
    indexer.build_index()
    return f"Indexed URL into {len(ids)} chunks. Index built."

# --- Query Callback ---

def handle_query(api_key: str, query: str, fusion_mode: str, alpha: float):
    if not api_key.strip():
        return "Error: Please provide a Gemini API Key.", None, ""
        
    if not query.strip():
        return "Please enter a query.", None, ""
        
    if not indexer.has_index():
        return "Error: Index not built. Please index some documents first.", None, ""
        
    logger.info(f"Processing query: {query} with {fusion_mode} (alpha={alpha})")
    
    # 1. Retrieval
    lex_results = retrieve_lexical(query)
    sem_results = retrieve_semantic(query)
    
    # 2. Fusion
    rrf_scores = reciprocal_rank_fusion(lex_results, sem_results)
    weight_scores = weighted_fusion(lex_results, sem_results, alpha)
    
    if fusion_mode == "RRF Mode":
        fused_to_sort = rrf_scores
    else:
        fused_to_sort = weight_scores
        
    top_fused = sort_fused_results(fused_to_sort, TOP_K_FUSED)
    
    # 3. Reranking
    top_reranked = reranker.rerank(query, top_fused)
    final_chunk_ids = [cid for cid, score in top_reranked]
    
    # 4. Generation
    answer = generate_answer(api_key, query, final_chunk_ids)
    
    # 5. Visualizer formatting
    # Collect all candidate chunks to show in the table (all chunks that made it to top_fused)
    table_data = []
    
    # We want to show the final rank for the ones that got reranked
    final_rank_map = {cid: rank + 1 for rank, (cid, score) in enumerate(top_reranked)}
    rerank_score_map = {cid: score for cid, score in top_reranked}
    
    for cid, fused_score in top_fused:
        chunk = storage.get_chunk(cid)
        preview = chunk["text"][:150].replace("\n", " ") + "..." if chunk else "N/A"
        
        row = {
            "Chunk ID": cid,
            "Preview Text": preview,
            "BM25 Score": round(lex_results.get(cid, 0.0), 4),
            "Vector Score": round(sem_results.get(cid, 0.0), 4),
            "RRF Score": round(rrf_scores.get(cid, 0.0), 4),
            "Weighted Score": round(weight_scores.get(cid, 0.0), 4),
            "Cross-Encoder Score": round(rerank_score_map.get(cid, "N/A"), 4) if cid in rerank_score_map else "N/A",
            "Final Rank": final_rank_map.get(cid, "Dropped")
        }
        table_data.append(row)
        
    df = pd.DataFrame(table_data)
    
    # Format retrieved chunks text to show in UI
    chunks_display = ""
    for cid in final_chunk_ids:
        chunk = storage.get_chunk(cid)
        doc = chunk['metadata'].get('document_name', 'Unknown')
        page = chunk['metadata'].get('page_number', 'N/A')
        chunks_display += f"--- Chunk {cid} (Doc: {doc}, Page: {page}) ---\n{chunk['text']}\n\n"
    
    return answer, df, chunks_display

# --- UI Setup ---

with gr.Blocks(title="Hybrid RAG Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Hybrid RAG Lab")
    gr.Markdown("A production-quality RAG system using BM25, FAISS, Reciprocal Rank Fusion, Cross-Encoder reranking, and Gemini-2.5-flash.")
    
    with gr.Tabs():
        # TAB 1: Indexing
        with gr.Tab("Indexing"):
            gr.Markdown("### Add Documents to the Knowledge Base")
            
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
                    pdf_btn = gr.Button("Index PDF(s)")
                    
                with gr.Column():
                    text_input = gr.Textbox(label="Raw Text", lines=5)
                    text_btn = gr.Button("Index Text")
                    
                with gr.Column():
                    url_input = gr.Textbox(label="Blog URL")
                    url_btn = gr.Button("Index URL")
                    
            status_output = gr.Textbox(label="Indexing Status", interactive=False)
            
            pdf_btn.click(handle_index_pdf, inputs=[pdf_input], outputs=[status_output])
            text_btn.click(handle_index_text, inputs=[text_input], outputs=[status_output])
            url_btn.click(handle_index_url, inputs=[url_input], outputs=[status_output])
            
        # TAB 2: Query
        with gr.Tab("Query"):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Gemini API Key", 
                    type="password", 
                    placeholder="Enter API key",
                    info="Get your Gemini API key from https://aistudio.google.com/app/apikey"
                )
                
            query_input = gr.Textbox(label="Query", placeholder="What would you like to know?", lines=2)
            
            with gr.Row():
                fusion_mode = gr.Radio(
                    choices=["RRF Mode", "Weighted Fusion Mode"], 
                    value="RRF Mode", 
                    label="Fusion Mode"
                )
                alpha_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.1, 
                    label="Hybrid Weight (Alpha) - Lexical Weight",
                    info="1.0 = BM25 Only, 0.0 = Vector Only. Only applies if Weighted Fusion Mode is selected."
                )
                
            ask_btn = gr.Button("Ask", variant="primary")
            
            gr.Markdown("### Answer")
            answer_output = gr.Textbox(label="Generated Answer", lines=6)
            
            gr.Markdown("### Retrieved Chunks (Top 5)")
            chunks_output = gr.Textbox(label="Source Context", lines=10)
            
            gr.Markdown("### Retrieval Visualizer")
            visualizer_output = gr.Dataframe(
                headers=["Chunk ID", "Preview Text", "BM25 Score", "Vector Score", "RRF Score", "Weighted Score", "Cross-Encoder Score", "Final Rank"],
                datatype=["number", "str", "number", "number", "number", "number", "number", "str"],
                label="Scoring Breakdown"
            )
            
            ask_btn.click(
                handle_query,
                inputs=[api_key_input, query_input, fusion_mode, alpha_slider],
                outputs=[answer_output, visualizer_output, chunks_output]
            )

if __name__ == "__main__":
    demo.launch()
