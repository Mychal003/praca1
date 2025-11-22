# backend/evaluation/inspect_all_chunks.py
"""
Wyświetla top-1 chunk dla każdego chunk_size.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()
from src.rag_pipeline import RAGPipeline

pdf_path = "uploads/Archer_D7UN_V1_UG.pdf"
question = "What are the default login credentials for the router?"

for chunk_size in [500, 800, 1200]:
    print(f"\n{'='*80}")
    print(f"CHUNK_SIZE = {chunk_size}")
    print(f"{'='*80}\n")
    
    pipeline = RAGPipeline(chunk_size=chunk_size, chunk_overlap=100, k=5)
    pipeline.process_document(pdf_path)
    
    sources = pipeline.get_sources(question, k=5)
    
    print(f"TOP-1 Chunk (ID: {sources[0]['chunk_id']}):")
    print(f"Score: {sources[0]['similarity_score']:.4f}\n")
    print(sources[0]['text'])
    print("\n" + "─"*80 + "\n")