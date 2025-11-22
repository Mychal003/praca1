# backend/evaluation/debug_llm_context.py
"""
Pokazuje DOKŁADNIE jaki kontekst LLM otrzymuje.
"""

from json import load
import sys
import os

import dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dotenv.load_dotenv()
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

pdf_path = "uploads/Archer_D7UN_V1_UG.pdf"
question = "What are the default login credentials for the router?"

for chunk_size in [500, 800, 1200]:
    print(f"\n{'='*80}")
    print(f"CHUNK_SIZE = {chunk_size}")
    print(f"{'='*80}\n")
    
    pipeline = RAGPipeline(chunk_size=chunk_size, chunk_overlap=100, k=5)
    pipeline.process_document(pdf_path)
    
    # Pobierz k=5 chunków (to co LLM dostanie)
    sources = pipeline.get_sources(question, k=5)
    
    # Złącz chunki (tak jak robi to RAG)
    combined_context = "\n\n---\n\n".join([s['text'] for s in sources])
    
    print(f"COMBINED CONTEXT LENGTH: {len(combined_context)} chars\n")
    print("FULL CONTEXT SENT TO LLM:")
    print("─"*80)
    print(combined_context)
    print("─"*80)
    
    # Generuj odpowiedź
    answer = pipeline.query(question)
    print(f"\n🤖 Generated: {answer}\n")
    
    # Sprawdź czy "default" jest w kontekście
    has_default = "default" in combined_context.lower()
    has_first_login = "first login" in combined_context.lower()
    has_set_password = "set" in combined_context.lower() and "password" in combined_context.lower()
    
    print(f"📊 Context analysis:")
    print(f"   Contains 'default': {has_default}")
    print(f"   Contains 'first login': {has_first_login}")
    print(f"   Contains 'set password': {has_set_password}")
    print()