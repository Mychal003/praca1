"""
Automatyczna ekstrakcja "ground truth" odpowiedzi bezpośrednio z dokumentu.
Używa LLM do znalezienia najlepszej odpowiedzi w dokumencie.
"""

from langchain_openai import ChatOpenAI
import json
from typing import List, Dict
#from src.rag_pipeline import RAGPipeline
#from evaluation.evaluate_simple import evaluate_system, TEST_DATASET

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from rag_pipeline import RAGPipeline

class GroundTruthExtractor:
    """
    Ekstrahuje 'prawdziwe' odpowiedzi bezpośrednio z dokumentu
    zamiast pisać je manualnie.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def extract_answer_from_chunks(self, question: str, chunks: List[str]) -> str:
        """
        Ekstrahuje najlepszą odpowiedź z chunków dokumentu.
        
        Args:
            question: Pytanie
            chunks: Lista relevantnych chunków z dokumentu
            
        Returns:
            Odpowiedź wyekstrahowana bezpośrednio z tekstu
        """
        
        # Połącz chunki
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

        response = self.llm.predict(prompt)
        return response.strip()
    
    def create_ground_truth_dataset(
        self, 
        pipeline: RAGPipeline, 
        questions: List[str],
        k: int = 5
    ) -> List[Dict]:
        """
        Tworzy dataset z ground truth odpowiedziami.
        
        Args:
            pipeline: RAGPipeline z załadowanym dokumentem
            questions: Lista pytań
            k: Liczba chunków do retrieval
            
        Returns:
            Lista dict z pytaniami i ground truth answers
        """
        
        dataset = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Extracting: {question[:60]}...")
            
            # Pobierz relevantne chunki
            sources = pipeline.get_sources(question, k=k)
            chunks = [s['text'] for s in sources]
            
            # Ekstrahuj odpowiedź
            ground_truth = self.extract_answer_from_chunks(question, chunks)
            
            dataset.append({
                "question": question,
                "ground_truth_answer": ground_truth,
                "source_chunks": chunks[:3],  # Zapisz top-3 dla referencji
                "category": self._infer_category(question)
            })
            
            print(f"   ✓ Extracted: {ground_truth[:80]}...")
        
        return dataset
    
    def _infer_category(self, question: str) -> str:
        """Prosta heurystyka kategoryzacji"""
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['how', 'how to', 'step', 'configure', 'set up', 'enable']):
            return "procedural"
        elif any(word in q_lower for word in ['why', 'cannot', 'not working', 'problem', 'error', 'troubleshoot']):
            return "troubleshooting"
        else:
            return "factual"


def regenerate_test_dataset(pdf_path: str, questions: List[str], output_file: str = "test_dataset_ground_truth.json"):
    """
    Główna funkcja: regeneruje dataset z ground truth.
    
    Args:
        pdf_path: Ścieżka do PDF
        questions: Lista pytań (bez expected answers!)
        output_file: Gdzie zapisać nowy dataset
    """
    
    print("\n" + "="*70)
    print("🔬 GROUND TRUTH EXTRACTION")
    print("="*70)
    
    # 1. Załaduj dokument
    print("\n1️⃣  Loading document...")
    pipeline = RAGPipeline(chunk_size=800, chunk_overlap=150, k=7)
    pipeline.process_document(pdf_path)
    
    # 2. Ekstrahuj ground truth
    print("\n2️⃣  Extracting ground truth answers...")
    extractor = GroundTruthExtractor()
    dataset = extractor.create_ground_truth_dataset(pipeline, questions, k=7)
    
    # 3. Zapisz
    print("\n3️⃣  Saving dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Ground truth dataset saved: {output_file}")
    print(f"   Total questions: {len(dataset)}")
    
    # 4. Pokaż przykłady
    print("\n" + "="*70)
    print("📝 PRZYKŁADY (pierwsze 3)")
    print("="*70)
    
    for i, item in enumerate(dataset[:3], 1):
        print(f"\n{i}. Q: {item['question']}")
        print(f"   A: {item['ground_truth_answer'][:150]}...")
    
    return dataset


# ============================================================================
# UŻYCIE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Tylko pytania, bez expected answers
    QUESTIONS_ONLY = [
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
    
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════╗
║         GROUND TRUTH EXTRACTION                             ║
╚══════════════════════════════════════════════════════════════╝

Użycie:
    python evaluation/extract_ground_truth.py <pdf_path>

Przykład:
    python evaluation/extract_ground_truth.py uploads/Archer_D7UN_V1_UG.pdf

Co robi:
    1. Ładuje dokument
    2. Dla każdego pytania znajduje relevantne chunki
    3. Używa GPT-4 do ekstrakcji DOKŁADNEJ odpowiedzi z tekstu
    4. Zapisuje do test_dataset_ground_truth.json
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    dataset = regenerate_test_dataset(pdf_path, QUESTIONS_ONLY)