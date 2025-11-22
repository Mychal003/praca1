# backend/evaluation/inspect_chunk.py
"""
Wyświetla konkretny chunk po ID.
"""
    
import sys
import os

from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()
from src.rag_pipeline import RAGPipeline

pdf_path = "uploads/Archer_D7UN_V1_UG.pdf"

# Test dla chunk_size=800
pipeline = RAGPipeline(chunk_size=800, chunk_overlap=100, k=5)
pipeline.process_document(pdf_path)

print(f"Total chunks: {pipeline.num_chunks}")

# Pobierz chunk #151 (top-1 dla chunk_size=800)
sources = pipeline.get_sources("What are the default login credentials?", k=5)

print("\n" + "="*80)
print("TOP-1 CHUNK (ID: 151):")
print("="*80)
print(sources[0]['text'])
print("="*80)