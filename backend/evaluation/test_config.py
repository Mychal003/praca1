"""
Test konkretnej konfiguracji RAG.
Pozwala przetestować dowolną kombinację chunk_size, k, overlap.

Użycie:
    python test_config.py <pdf_path> <dataset_path> --chunk-size 1200 --k 7 --overlap 150
    python test_config.py <pdf_path> <dataset_path> --chunk-size 1200 --k 7 --llm-judge

Przykład:
    python evaluation/test_config.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json --chunk-size 1200 --k 7 --overlap 150 --llm-judge
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline

# Semantic model
print("🤖 Ładuję model Semantic Similarity...")
from sentence_transformers import SentenceTransformer, util
SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')
print("✅ Model załadowany!\n")


# ============================================================================
# METRYKI
# ============================================================================

def rouge_1_f1(prediction: str, reference: str) -> float:
    pred_words = Counter(prediction.lower().split())
    ref_words = Counter(reference.lower().split())
    if not pred_words or not ref_words:
        return 0.0
    overlap = sum((pred_words & ref_words).values())
    if sum(pred_words.values()) == 0 or sum(ref_words.values()) == 0:
        return 0.0
    p = overlap / sum(pred_words.values())
    r = overlap / sum(ref_words.values())
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def semantic_similarity(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    emb1 = SEMANTIC_MODEL.encode(prediction, convert_to_tensor=True)
    emb2 = SEMANTIC_MODEL.encode(reference, convert_to_tensor=True)
    return max(0.0, min(1.0, util.pytorch_cos_sim(emb1, emb2).item()))


# ============================================================================
# LLM JUDGE (uproszczony)
# ============================================================================

class LLMJudge:
    def __init__(self, model: str = "gpt-4o"):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def evaluate(self, question: str, generated: str, reference: str, context: str = None):
        scores = {
            'correctness': self._score(question, generated, reference, "correctness"),
            'completeness': self._score(question, generated, reference, "completeness"),
            'relevance': self._score_relevance(question, generated),
            'groundedness': self._score_groundedness(question, generated, context) if context else None,
            'overall': self._score(question, generated, reference, "overall")
        }
        valid = [v for v in scores.values() if v is not None]
        scores['average'] = sum(valid) / len(valid) if valid else 0.0
        return scores
    
    def _score(self, q, gen, ref, metric):
        prompt = f"""Rate the {metric.upper()} of the generated answer (0.0-1.0).

Question: {q}
Reference: {ref}
Generated: {gen}

Respond with ONLY a number (e.g., 0.85):"""
        try:
            response = self.llm.predict(prompt).strip()
            return max(0.0, min(1.0, float(response)))
        except:
            return 0.5
    
    def _score_relevance(self, q, gen):
        prompt = f"""Rate the RELEVANCE of the answer to the question (0.0-1.0).

Question: {q}
Generated: {gen}

1.0 = directly answers the question
0.0 = completely irrelevant

Respond with ONLY a number:"""
        try:
            response = self.llm.predict(prompt).strip()
            return max(0.0, min(1.0, float(response)))
        except:
            return 0.5
    
    def _score_groundedness(self, q, gen, ctx):
        prompt = f"""Rate if the answer is GROUNDED in the context (0.0-1.0).

Question: {q}
Context: {ctx[:2000]}
Generated: {gen}

1.0 = all facts from context (or correctly says "no info")
0.0 = contains hallucinations

Respond with ONLY a number:"""
        try:
            response = self.llm.predict(prompt).strip()
            return max(0.0, min(1.0, float(response)))
        except:
            return 0.5


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def test_config(pdf_path: str, dataset_path: str, 
                chunk_size: int, k: int, overlap: int,
                use_llm_judge: bool = False):
    """
    Testuje konkretną konfigurację RAG.
    """
    
    print(f"\n{'='*70}")
    print(f"🧪 TEST KONFIGURACJI")
    print(f"{'='*70}")
    print(f"   chunk_size: {chunk_size}")
    print(f"   k: {k}")
    print(f"   overlap: {overlap}")
    print(f"   LLM Judge: {'TAK' if use_llm_judge else 'NIE'}")
    print(f"{'='*70}\n")
    
    # Wczytaj dataset
    print(f"📂 Ładowanie datasetu: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'data' in data:
        dataset = data['data']
    else:
        dataset = data
    
    print(f"   ✓ Załadowano {len(dataset)} pytań")
    
    # Stwórz pipeline
    print(f"\n🔧 Tworzenie pipeline...")
    pipeline = RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        k=k
    )
    pipeline.process_document(pdf_path)
    print(f"   ✓ Utworzono {pipeline.num_chunks} chunków")
    
    # LLM Judge
    judge = LLMJudge() if use_llm_judge else None
    
    # Ewaluacja
    print(f"\n📝 Ewaluacja...\n")
    
    results = []
    failures = []  # Pytania z "I don't have info"
    
    for i, item in enumerate(dataset, 1):
        question = item['question']
        expected = item['expected_answer']
        
        print(f"[{i}/{len(dataset)}] {question[:50]}...")
        
        # Generuj odpowiedź
        start = time.time()
        try:
            generated = pipeline.query(question)
        except Exception as e:
            print(f"   ❌ Błąd: {e}")
            generated = "ERROR"
        latency = time.time() - start
        
        # Sprawdź czy to "I don't have info"
        is_failure = "don't have" in generated.lower() or "do not have" in generated.lower()
        
        # Metryki
        rouge = rouge_1_f1(generated, expected)
        semantic = semantic_similarity(generated, expected)
        
        result = {
            'question': question,
            'expected': expected,
            'generated': generated,
            'category': item.get('category', 'unknown'),
            'is_failure': is_failure,
            'metrics': {
                'rouge1_f1': rouge,
                'semantic_similarity': semantic,
                'latency': latency
            }
        }
        
        # LLM Judge
        if judge and generated != "ERROR":
            sources = pipeline.get_sources(question, k=k)
            context = "\n\n".join([s['text'] for s in sources])
            llm_scores = judge.evaluate(question, generated, expected, context)
            result['metrics']['llm_judge'] = llm_scores
            
            status = "❌ FAILURE" if is_failure else "✅"
            print(f"   {status} ROUGE: {rouge:.3f} | Semantic: {semantic:.3f} | LLM: {llm_scores['overall']:.2f}")
        else:
            status = "❌ FAILURE" if is_failure else "✅"
            print(f"   {status} ROUGE: {rouge:.3f} | Semantic: {semantic:.3f}")
        
        if is_failure:
            failures.append(question)
        
        results.append(result)
    
    # Podsumowanie
    print(f"\n{'='*70}")
    print(f"📊 PODSUMOWANIE")
    print(f"{'='*70}\n")
    
    # Podstawowe metryki
    avg_rouge = sum(r['metrics']['rouge1_f1'] for r in results) / len(results)
    avg_semantic = sum(r['metrics']['semantic_similarity'] for r in results) / len(results)
    avg_latency = sum(r['metrics']['latency'] for r in results) / len(results)
    
    # Success rate
    success_count = len([r for r in results if not r['is_failure']])
    success_rate = success_count / len(results) * 100
    
    print(f"{'Metryka':<25} {'Wartość'}")
    print("-" * 40)
    print(f"{'ROUGE-1 F1':<25} {avg_rouge:.3f}")
    print(f"{'Semantic Similarity':<25} {avg_semantic:.3f}")
    print(f"{'Latencja (s)':<25} {avg_latency:.2f}")
    print(f"{'Success Rate':<25} {success_rate:.1f}% ({success_count}/{len(results)})")
    print(f"{'Failures':<25} {len(failures)}")
    
    # Metryki tylko dla sukcesów
    successes = [r for r in results if not r['is_failure']]
    if successes:
        avg_rouge_success = sum(r['metrics']['rouge1_f1'] for r in successes) / len(successes)
        avg_semantic_success = sum(r['metrics']['semantic_similarity'] for r in successes) / len(successes)
        print(f"\n{'Metryki (tylko sukcesy):'}")
        print(f"{'  ROUGE-1 F1':<25} {avg_rouge_success:.3f}")
        print(f"{'  Semantic Similarity':<25} {avg_semantic_success:.3f}")
    
    # LLM Judge
    if use_llm_judge:
        print()
        llm_results = [r for r in results if 'llm_judge' in r['metrics']]
        for key in ['correctness', 'completeness', 'relevance', 'groundedness', 'overall']:
            scores = [r['metrics']['llm_judge'][key] for r in llm_results 
                     if r['metrics']['llm_judge'].get(key) is not None]
            if scores:
                print(f"{'LLM ' + key.title():<25} {sum(scores)/len(scores):.3f}")
    
    # Lista failures
    if failures:
        print(f"\n{'='*70}")
        print(f"❌ PYTANIA Z 'I DON'T HAVE INFO' ({len(failures)}):")
        print(f"{'='*70}")
        for i, q in enumerate(failures, 1):
            print(f"   {i}. {q[:65]}...")
    
    # Per category
    print(f"\n{'='*70}")
    print(f"📊 PER KATEGORIA")
    print(f"{'='*70}")
    
    categories = set(r['category'] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r['category'] == cat]
        cat_successes = [r for r in cat_results if not r['is_failure']]
        cat_rouge = sum(r['metrics']['rouge1_f1'] for r in cat_results) / len(cat_results)
        cat_success_rate = len(cat_successes) / len(cat_results) * 100
        print(f"   {cat.upper():<15} ROUGE: {cat_rouge:.3f} | Success: {cat_success_rate:.0f}% | n={len(cat_results)}")
    
    print(f"\n{'='*70}\n")
    
    # Zapisz wyniki
    summary = {
        'config': {
            'chunk_size': chunk_size,
            'k': k,
            'overlap': overlap,
            'num_chunks': pipeline.num_chunks
        },
        'metrics': {
            'avg_rouge1_f1': avg_rouge,
            'avg_semantic_similarity': avg_semantic,
            'avg_latency': avg_latency,
            'success_rate': success_rate,
            'num_failures': len(failures)
        },
        'failures': failures,
        'detailed': results
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"config_test_{chunk_size}_{k}_{overlap}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Wyniki zapisane: {output_file}")
    
    return summary


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test konkretnej konfiguracji RAG')
    parser.add_argument('pdf_path', help='Ścieżka do PDF')
    parser.add_argument('dataset_path', help='Ścieżka do datasetu')
    parser.add_argument('--chunk-size', type=int, default=1200, help='Rozmiar chunka (default: 1200)')
    parser.add_argument('--k', type=int, default=7, help='Liczba dokumentów k (default: 7)')
    parser.add_argument('--overlap', type=int, default=150, help='Overlap (default: 150)')
    parser.add_argument('--llm-judge', action='store_true', help='Użyj LLM Judge')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    TEST KONFIGURACJI RAG                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  chunk_size: {args.chunk_size:<10}                                          ║
║  k:          {args.k:<10}                                          ║
║  overlap:    {args.overlap:<10}                                          ║
║  LLM Judge:  {'TAK' if args.llm_judge else 'NIE':<10}                                          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    test_config(
        args.pdf_path,
        args.dataset_path,
        chunk_size=args.chunk_size,
        k=args.k,
        overlap=args.overlap,
        use_llm_judge=args.llm_judge
    )