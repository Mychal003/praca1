# backend/evaluation/debug_retrieval.py
"""
Debug script - sprawdza co retrieval pobiera dla konkretnego pytania.
"""
from dotenv import load_dotenv

load_dotenv()


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline

def debug_question(pdf_path: str, question: str, chunk_sizes: list = [500, 800, 1200]):
    """
    Debuguje retrieval dla jednego pytania z różnymi chunk_size.
    """
    print(f"\n{'='*80}")
    print(f"🔍 DEBUG RETRIEVAL")
    print(f"{'='*80}")
    print(f"\nPytanie: {question}")
    print(f"{'='*80}\n")
    
    for chunk_size in chunk_sizes:
        print(f"\n{'─'*80}")
        print(f"📦 CHUNK_SIZE = {chunk_size}")
        print(f"{'─'*80}\n")
        
        # Stwórz pipeline
        pipeline = RAGPipeline(
            chunk_size=chunk_size,
            chunk_overlap=100,
            k=5
        )
        
        # Przetwórz dokument
        pipeline.process_document(pdf_path)
        
        print(f"   Total chunks created: {pipeline.num_chunks}")
        
        # Pobierz retrieved sources
        sources = pipeline.get_sources(question, k=5)
        
        print(f"   Retrieved {len(sources)} chunks:\n")
        
        for i, source in enumerate(sources, 1):
            chunk_id = source['chunk_id']
            text = source['text']
            score = source.get('similarity_score', 'N/A')
            
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"   [{i}] Chunk ID: {chunk_id} | Score: {score_str}")
        
        # Generuj odpowiedź
        try:
            answer = pipeline.query(question)
            print(f"   🤖 Generated answer: {answer}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
        
        print(f"{'─'*80}\n")


if __name__ == "__main__":
    pdf_path = "uploads/Archer_D7UN_V1_UG.pdf"
    
    # Problematyczne pytanie
    question = "What are the default login credentials for the router?"
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║              DEBUG RETRIEVAL - Credentials Question          ║
╚══════════════════════════════════════════════════════════════╝

Sprawdzamy co retrieval pobiera dla pytania o credentials
dla trzech chunk_size: 500, 800, 1200

Oczekiwania:
- chunk_size=800 powinien retrieval chunki BEZ informacji o credentials
- chunk_size=500/1200 powinien retrieval chunki Z informacją o credentials

Jeśli tak jest - LLM Judge ocenił poprawnie!
Jeśli nie - mamy problem z oceną LLM Judge.
    """)
    
    debug_question(pdf_path, question)