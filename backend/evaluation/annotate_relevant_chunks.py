"""
Narzędzie do annotacji relevant chunks dla retrieval evaluation.
Używa LLM do automatycznego oznaczenia które chunki są relevant.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from typing import List, Dict
from src.rag_pipeline import RAGPipeline
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


class RelevanceAnnotator:
    """
    Automatycznie annotuje które chunki są relevant dla danego pytania.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def is_chunk_relevant(self, question: str, chunk: str, threshold: float = 0.7) -> bool:
        """
        Używa LLM do określenia czy chunk jest relevant dla pytania.
        
        Args:
            question: Pytanie użytkownika
            chunk: Text chunk z dokumentu
            threshold: Próg relevance (0-1)
            
        Returns:
            True jeśli chunk jest relevant
        """
        prompt = f"""You are evaluating if a text chunk contains information relevant to answering a question.

Question: {question}

Text Chunk:
{chunk}

Does this chunk contain information that would help answer the question?
Consider a chunk relevant if it contains:
- Direct answers to the question
- Information needed to formulate the answer
- Context that supports the answer

Respond with ONLY "YES" or "NO".
"""
        
        try:
            response = self.llm.predict(prompt).strip().upper()
            return response == "YES"
        except Exception as e:
            print(f"⚠️  Error checking relevance: {e}")
            return False
    
    def annotate_dataset(
        self, 
        pipeline: RAGPipeline, 
        questions: List[Dict],
        k: int = 20  # Pobierz więcej chunków do annotacji
    ) -> List[Dict]:
        """
        Annotuje dataset z relevant chunks dla każdego pytania.
        
        Args:
            pipeline: RAGPipeline z załadowanym dokumentem
            questions: Lista pytań z datasetu
            k: Liczba chunków do rozważenia
            
        Returns:
            Dataset z dodanymi relevant_chunk_indices
        """
        annotated_dataset = []
        
        print(f"\n{'='*70}")
        print("🏷️  ANNOTACJA RELEVANT CHUNKS")
        print(f"{'='*70}\n")
        
        for i, item in enumerate(questions, 1):
            question = item['question']
            print(f"[{i}/{len(questions)}] {question[:60]}...")
            
            try:
                # Pobierz top-k chunków
                all_chunks = pipeline.get_sources(question, k=k)
                
                # Sprawdź relevance każdego chunku
                relevant_indices = []
                for j, chunk_data in enumerate(all_chunks):
                    chunk_text = chunk_data['text']
                    is_relevant = self.is_chunk_relevant(question, chunk_text)
                    
                    if is_relevant:
                        relevant_indices.append(j)
                        print(f"   ✓ Chunk {j} is relevant")
                
                # Dodaj annotacje do datasetu
                annotated_item = item.copy()
                annotated_item['relevant_chunk_indices'] = relevant_indices
                annotated_item['total_chunks_evaluated'] = k
                annotated_dataset.append(annotated_item)
                
                print(f"   → Found {len(relevant_indices)} relevant chunks\n")
                
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
                # Dodaj bez annotacji
                annotated_item = item.copy()
                annotated_item['relevant_chunk_indices'] = []
                annotated_item['total_chunks_evaluated'] = 0
                annotated_dataset.append(annotated_item)
        
        return annotated_dataset


def annotate_test_dataset(pdf_path: str, input_dataset_path: str, output_path: str):
    """
    Główna funkcja: annotuje istniejący dataset.
    """
    # 1. Wczytaj dataset
    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} questions from {input_dataset_path}")
    
    # 2. Załaduj dokument
    print("\n🔧 Initializing RAG pipeline...")
    pipeline = RAGPipeline(chunk_size=800, chunk_overlap=100, k=20)
    pipeline.process_document(pdf_path)
    
    # 3. Annotuj
    annotator = RelevanceAnnotator()
    annotated_dataset = annotator.annotate_dataset(pipeline, dataset, k=20)
    
    # 4. Zapisz
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotated_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Annotated dataset saved: {output_path}")
    
    # 5. Statystyki
    total_relevant = sum(len(item['relevant_chunk_indices']) for item in annotated_dataset)
    avg_relevant = total_relevant / len(annotated_dataset)
    
    print(f"\n{'='*70}")
    print("📊 STATYSTYKI ANNOTACJI")
    print(f"{'='*70}")
    print(f"Pytania: {len(annotated_dataset)}")
    print(f"Średnia liczba relevant chunks na pytanie: {avg_relevant:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              ANNOTACJA RELEVANT CHUNKS                       ║
╚══════════════════════════════════════════════════════════════╝

Użycie:
    python evaluation/annotate_relevant_chunks.py <pdf_path> <input_dataset.json>

Przykład:
    python evaluation/annotate_relevant_chunks.py \\
        uploads/Archer_D7UN_V1_UG.pdf \\
        test_dataset_ground_truth.json

Wyjście:
    test_dataset_with_relevance.json
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    input_dataset = sys.argv[2]
    output_dataset = input_dataset.replace('.json', '_with_relevance.json')
    
    annotate_test_dataset(pdf_path, input_dataset, output_dataset)