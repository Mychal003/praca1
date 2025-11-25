"""
Przygotowanie datasetu do ewaluacji.
Łączy: ekstrakcję ground truth + annotację relevant chunks.

Użycie:
    python prepare_dataset.py <pdf_path>
    python prepare_dataset.py uploads/Archer_D7UN_V1_UG.pdf
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from src.rag_pipeline import RAGPipeline

# ============================================================================
# KONFIGURACJA
# ============================================================================

# Pytania do datasetu (bez odpowiedzi - będą wyekstrahowane)
QUESTIONS = [
    "What is the full model name and type of this router?",
    "What is the default web address to access the router interface?",
    "What are the default login credentials for the router?",
    "What port is used to connect the router to the Internet?",
    "How many operation modes does the Archer D7 support?",
    "What USB features does the router support?",
    "What is the WPS button used for on the Archer D7?",
    "What wireless security functions does the router provide?",
    "How do you perform a factory reset on the Archer D7?",
    "How do you access the router's web interface for the first time?",
    "How do you set up the router using Quick Setup Wizard?",
    "How do you change the wireless network name and password?",
    "How do you turn on or off the WiFi function on the router?",
    "How do you access a USB disk connected to the router via network?",
    "How do you set up parental controls on the router?",
    "How do you customize the USB disk server name?",
    "How do you enable MAC Filtering to control wireless access?",
    "What should you do if you cannot access the router's web interface?",
    "How do you recover access if you forgot the router's login password?",
    "What does it mean if the ADSL LED is not lit on the router?",
    "What are the possible causes if wireless devices cannot connect to the network?",
    "Why might bandwidth control not work as expected?",
    "What is the purpose of IP & MAC Binding feature?",
    "What is Access Control and how does it differ from MAC Filtering?",
    "How can you remotely access USB storage connected to the router?",
]

# Baseline config (MUSI być taka sama jak w evaluate_retrieval.py!)
BASELINE_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "k": 20  # Więcej dla annotacji
}


# ============================================================================
# GROUND TRUTH EXTRACTOR
# ============================================================================

class GroundTruthExtractor:
    """Ekstrahuje odpowiedzi bezpośrednio z dokumentu."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def extract_answer(self, question: str, chunks: list) -> str:
        """Ekstrahuje odpowiedź z chunków."""
        context = "\n\n---\n\n".join(chunks)
        
        prompt = f"""You are extracting the EXACT answer from a technical document.

TASK: Find the answer to the question in the context below and extract it WORD-FOR-WORD from the document.

RULES:
1. Use ONLY text that appears in the context
2. Extract complete sentences, don't paraphrase
3. Include all relevant details from the document
4. If answer spans multiple sentences, include them all
5. Don't add information not in the context
6. Don't use phrases like "According to the document" - just the raw answer

Context from document:
{context}

Question: {question}

Extracted answer (word-for-word from document):"""

        return self.llm.predict(prompt).strip()


# ============================================================================
# RELEVANCE ANNOTATOR
# ============================================================================

class RelevanceAnnotator:
    """Annotuje które chunki są relevant dla pytania."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def is_relevant(self, question: str, chunk: str) -> bool:
        """Sprawdza czy chunk jest relevant."""
        prompt = f"""You are evaluating if a text chunk contains information relevant to answering a question.

Question: {question}

Text Chunk:
{chunk}

Does this chunk contain information that would help answer the question?
Consider a chunk relevant if it contains:
- Direct answers to the question
- Information needed to formulate the answer
- Context that supports the answer

Respond with ONLY "YES" or "NO"."""

        response = self.llm.predict(prompt).strip().upper()
        return response == "YES"


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def prepare_dataset(pdf_path: str, output_path: str = "dataset_ready.json"):
    """
    Przygotowuje kompletny dataset do ewaluacji.
    
    Kroki:
    1. Ładuje dokument
    2. Ekstrahuje ground truth answers
    3. Annotuje relevant chunks
    4. Zapisuje gotowy dataset
    """
    
    print("\n" + "="*70)
    print("📦 PRZYGOTOWANIE DATASETU DO EWALUACJI")
    print("="*70)
    
    # 1. Załaduj dokument
    print(f"\n[1/4] Ładowanie dokumentu: {pdf_path}")
    print(f"      Config: chunk_size={BASELINE_CONFIG['chunk_size']}, overlap={BASELINE_CONFIG['chunk_overlap']}")
    
    pipeline = RAGPipeline(
        chunk_size=BASELINE_CONFIG['chunk_size'],
        chunk_overlap=BASELINE_CONFIG['chunk_overlap'],
        k=BASELINE_CONFIG['k']
    )
    pipeline.process_document(pdf_path)
    print(f"      ✓ Utworzono {pipeline.num_chunks} chunków")
    
    # 2. Inicjalizacja
    print(f"\n[2/4] Inicjalizacja ekstraktorów...")
    extractor = GroundTruthExtractor()
    annotator = RelevanceAnnotator()
    print(f"      ✓ Gotowe")
    
    # 3. Przetwarzanie pytań
    print(f"\n[3/4] Przetwarzanie {len(QUESTIONS)} pytań...")
    
    dataset = []
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n   [{i}/{len(QUESTIONS)}] {question[:50]}...")
        
        # Pobierz chunki
        sources = pipeline.get_sources(question, k=BASELINE_CONFIG['k'])
        chunks = [s['text'] for s in sources]
        
        # Ekstrahuj ground truth
        ground_truth = extractor.extract_answer(question, chunks)
        print(f"      ✓ GT: {ground_truth[:60]}...")
        
        # Annotuj relevant chunks
        relevant_ids = []
        for source in sources:
            if annotator.is_relevant(question, source['text']):
                relevant_ids.append(source['chunk_id'])
        
        print(f"      ✓ Relevant chunks: {relevant_ids}")
        
        # Kategoryzuj pytanie
        q_lower = question.lower()
        if any(w in q_lower for w in ['how', 'step', 'configure', 'set up', 'enable']):
            category = "procedural"
        elif any(w in q_lower for w in ['why', 'cannot', 'problem', 'error', 'troubleshoot']):
            category = "troubleshooting"
        else:
            category = "factual"
        
        dataset.append({
            "question": question,
            "expected_answer": ground_truth,
            "relevant_chunk_indices": relevant_ids,
            "category": category
        })
    
    # 4. Zapisz
    print(f"\n[4/4] Zapisywanie datasetu...")
    
    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "pdf_path": pdf_path,
            "num_questions": len(dataset),
            "baseline_config": BASELINE_CONFIG,
            "total_chunks": pipeline.num_chunks
        },
        "data": dataset
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Statystyki
    total_relevant = sum(len(item['relevant_chunk_indices']) for item in dataset)
    avg_relevant = total_relevant / len(dataset)
    
    print(f"\n" + "="*70)
    print("✅ DATASET GOTOWY!")
    print("="*70)
    print(f"   Plik: {output_path}")
    print(f"   Pytań: {len(dataset)}")
    print(f"   Chunków w dokumencie: {pipeline.num_chunks}")
    print(f"   Średnia relevant chunks/pytanie: {avg_relevant:.1f}")
    print("="*70 + "\n")
    
    return output


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              PREPARE DATASET                                 ║
╚══════════════════════════════════════════════════════════════╝

Użycie:
    python prepare_dataset.py <pdf_path> [output_path]

Przykład:
    python prepare_dataset.py uploads/Archer_D7UN_V1_UG.pdf
    python prepare_dataset.py uploads/Archer_D7UN_V1_UG.pdf my_dataset.json
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_ready.json"
    
    prepare_dataset(pdf_path, output_path)