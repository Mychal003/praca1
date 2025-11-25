"""
Ewaluacja GENERATION - tylko metryki generacji.
ROUGE-1 F1, Semantic Similarity, LLM Judge

Użycie:
    python evaluate_generation.py <pdf_path> <dataset_path>
    python evaluate_generation.py <pdf_path> <dataset_path> --llm-judge
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline

# Semantic similarity model
print("🤖 Ładuję model Semantic Similarity...")
from sentence_transformers import SentenceTransformer, util
SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')
print("✅ Model załadowany!\n")


# ============================================================================
# GENERATION METRICS
# ============================================================================

def rouge_1_f1(prediction: str, reference: str) -> float:
    """ROUGE-1 F1: overlap słów"""
    pred_words = Counter(prediction.lower().split())
    ref_words = Counter(reference.lower().split())
    
    if not pred_words or not ref_words:
        return 0.0
    
    overlap = sum((pred_words & ref_words).values())
    
    if sum(pred_words.values()) == 0 or sum(ref_words.values()) == 0:
        return 0.0
    
    precision = overlap / sum(pred_words.values())
    recall = overlap / sum(ref_words.values())
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(prediction: str, reference: str) -> float:
    """Semantic similarity: cosine similarity embeddingów"""
    if not prediction or not reference:
        return 0.0
    
    emb1 = SEMANTIC_MODEL.encode(prediction, convert_to_tensor=True)
    emb2 = SEMANTIC_MODEL.encode(reference, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return max(0.0, min(1.0, similarity))


def token_overlap(prediction: str, reference: str) -> float:
    """Token Overlap: prosty % wspólnych słów"""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    overlap = len(pred_words & ref_words)
    return overlap / len(ref_words)


# ============================================================================
# LLM JUDGE
# ============================================================================

class LLMJudge:
    """LLM as Judge - ocena jakości odpowiedzi."""
    
    def __init__(self, model: str = "gpt-4o"):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def evaluate(
        self, 
        question: str, 
        generated: str, 
        reference: str,
        context: str = None
    ) -> Dict[str, float]:
        """Ocenia odpowiedź w 5 wymiarach."""
        
        scores = {
            'correctness': self._score_correctness(question, generated, reference),
            'completeness': self._score_completeness(question, generated, reference),
            'relevance': self._score_relevance(question, generated),
            'groundedness': self._score_groundedness(question, generated, context) if context else None,
            'overall': self._score_overall(question, generated, reference)
        }
        
        # Average (bez None)
        valid = [v for v in scores.values() if v is not None]
        scores['average'] = sum(valid) / len(valid) if valid else 0.0
        
        return scores
    
    def _get_score(self, prompt: str) -> float:
        """Helper: wywołuje LLM i parsuje score."""
        try:
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _score_correctness(self, q: str, gen: str, ref: str) -> float:
        prompt = f"""Rate the CORRECTNESS of the generated answer (0.0-1.0).

Question: {q}
Reference: {ref}
Generated: {gen}

1.0 = completely correct
0.0 = completely wrong

Respond with ONLY a number (e.g., 0.85):"""
        return self._get_score(prompt)
    
    def _score_completeness(self, q: str, gen: str, ref: str) -> float:
        prompt = f"""Rate the COMPLETENESS of the generated answer (0.0-1.0).

Question: {q}
Reference: {ref}
Generated: {gen}

1.0 = contains all key information
0.0 = missing all key information

Respond with ONLY a number (e.g., 0.75):"""
        return self._get_score(prompt)
    
    def _score_relevance(self, q: str, gen: str) -> float:
        prompt = f"""Rate the RELEVANCE of the answer to the question (0.0-1.0).

Question: {q}
Generated: {gen}

1.0 = directly answers the question
0.0 = completely irrelevant

Respond with ONLY a number (e.g., 0.90):"""
        return self._get_score(prompt)
    
    def _score_groundedness(self, q: str, gen: str, ctx: str) -> float:
        prompt = f"""Rate if the answer is GROUNDED in the context (0.0-1.0).

Question: {q}
Context: {ctx[:2000]}
Generated: {gen}

1.0 = all facts from context (or correctly says "no info")
0.0 = contains hallucinations

Respond with ONLY a number (e.g., 0.85):"""
        return self._get_score(prompt)
    
    def _score_overall(self, q: str, gen: str, ref: str) -> float:
        prompt = f"""Rate the OVERALL QUALITY of the answer (0.0-1.0).

Question: {q}
Reference: {ref}
Generated: {gen}

Consider: accuracy, completeness, clarity, usefulness.

Respond with ONLY a number (e.g., 0.80):"""
        return self._get_score(prompt)


# ============================================================================
# EWALUACJA
# ============================================================================

def evaluate_generation(
    pipeline: RAGPipeline,
    dataset: List[Dict],
    use_llm_judge: bool = False
) -> Dict:
    """
    Ewaluacja generation metrics.
    
    Args:
        pipeline: RAGPipeline z załadowanym dokumentem
        dataset: Dataset z 'question' i 'expected_answer'
        use_llm_judge: Czy użyć LLM Judge (droższe!)
    
    Returns:
        Dict z wynikami
    """
    
    print(f"\n{'='*70}")
    print("📝 EWALUACJA GENERATION")
    print(f"{'='*70}")
    print(f"   Pytań: {len(dataset)}")
    print(f"   LLM Judge: {'TAK' if use_llm_judge else 'NIE'}")
    print(f"{'='*70}\n")
    
    # Inicjalizacja LLM Judge
    judge = LLMJudge() if use_llm_judge else None
    
    results = []
    
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
        
        # Oblicz metryki
        rouge = rouge_1_f1(generated, expected)
        semantic = semantic_similarity(generated, expected)
        overlap = token_overlap(generated, expected)
        
        result = {
            'question': question,
            'expected': expected,
            'generated': generated,
            'category': item.get('category', 'unknown'),
            'metrics': {
                'rouge1_f1': rouge,
                'semantic_similarity': semantic,
                'token_overlap': overlap,
                'latency': latency
            }
        }
        
        # LLM Judge (opcjonalnie)
        if judge and generated != "ERROR":
            # Pobierz kontekst dla groundedness
            sources = pipeline.get_sources(question, k=5)
            context = "\n\n".join([s['text'] for s in sources])
            
            llm_scores = judge.evaluate(question, generated, expected, context)
            result['metrics']['llm_judge'] = llm_scores
            
            print(f"   ROUGE: {rouge:.3f} | Semantic: {semantic:.3f} | LLM: {llm_scores['overall']:.2f} | {latency:.2f}s")
        else:
            print(f"   ROUGE: {rouge:.3f} | Semantic: {semantic:.3f} | {latency:.2f}s")
        
        results.append(result)
    
    # Agregacja
    print(f"\n{'='*70}")
    print("📊 PODSUMOWANIE GENERATION METRICS")
    print(f"{'='*70}\n")
    
    # Podstawowe metryki
    avg_rouge = sum(r['metrics']['rouge1_f1'] for r in results) / len(results)
    avg_semantic = sum(r['metrics']['semantic_similarity'] for r in results) / len(results)
    avg_overlap = sum(r['metrics']['token_overlap'] for r in results) / len(results)
    avg_latency = sum(r['metrics']['latency'] for r in results) / len(results)
    
    summary = {
        'avg_rouge1_f1': avg_rouge,
        'avg_semantic_similarity': avg_semantic,
        'avg_token_overlap': avg_overlap,
        'avg_latency': avg_latency
    }
    
    print(f"{'Metryka':<25} {'Wartość':<12}")
    print("-" * 37)
    print(f"{'ROUGE-1 F1':<25} {avg_rouge:<12.3f}")
    print(f"{'Semantic Similarity':<25} {avg_semantic:<12.3f}")
    print(f"{'Token Overlap':<25} {avg_overlap:<12.3f}")
    print(f"{'Latencja (s)':<25} {avg_latency:<12.2f}")
    
    # LLM Judge metryki
    if use_llm_judge:
        print()
        llm_results = [r for r in results if 'llm_judge' in r['metrics']]
        
        if llm_results:
            for key in ['correctness', 'completeness', 'relevance', 'groundedness', 'overall']:
                scores = [r['metrics']['llm_judge'][key] for r in llm_results 
                         if r['metrics']['llm_judge'].get(key) is not None]
                if scores:
                    avg = sum(scores) / len(scores)
                    summary[f'avg_llm_{key}'] = avg
                    print(f"{'LLM ' + key.title():<25} {avg:<12.3f}")
    
    # Metryki per kategoria
    print(f"\n{'='*70}")
    print("📊 METRYKI PER KATEGORIA")
    print(f"{'='*70}\n")
    
    categories = set(r['category'] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r['category'] == cat]
        cat_rouge = sum(r['metrics']['rouge1_f1'] for r in cat_results) / len(cat_results)
        cat_semantic = sum(r['metrics']['semantic_similarity'] for r in cat_results) / len(cat_results)
        
        print(f"{cat.upper():<20} ROUGE: {cat_rouge:.3f} | Semantic: {cat_semantic:.3f} | n={len(cat_results)}")
        summary[f'{cat}_rouge1_f1'] = cat_rouge
        summary[f'{cat}_semantic'] = cat_semantic
    
    print(f"\n{'='*70}\n")
    
    return {
        'summary': summary,
        'detailed': results,
        'config': {
            'use_llm_judge': use_llm_judge,
            'pipeline_k': pipeline.k,
            'chunk_size': pipeline.chunk_size,
            'model': 'gpt-4o'
        }
    }


# ============================================================================
# ANALIZA KORELACJI METRYK
# ============================================================================

def compare_metrics_correlation(results: Dict):
    """
    Analizuje korelację między ROUGE/Semantic a LLM Judge.
    Pokazuje która metryka lepiej przewiduje jakość.
    """
    detailed = results.get('detailed', [])
    
    # Sprawdź czy mamy LLM Judge
    llm_results = [r for r in detailed if 'llm_judge' in r.get('metrics', {})]
    
    if len(llm_results) < 3:
        print("⚠️  Za mało wyników z LLM Judge do analizy korelacji")
        return None
    
    print(f"\n{'='*70}")
    print("📊 ANALIZA KORELACJI METRYK")
    print(f"{'='*70}\n")
    
    # Zbierz dane
    rouge = [r['metrics']['rouge1_f1'] for r in llm_results]
    semantic = [r['metrics']['semantic_similarity'] for r in llm_results]
    llm_overall = [r['metrics']['llm_judge']['overall'] for r in llm_results]
    llm_correctness = [r['metrics']['llm_judge']['correctness'] for r in llm_results]
    
    try:
        from scipy.stats import pearsonr
        
        corr_rouge_overall, p1 = pearsonr(rouge, llm_overall)
        corr_semantic_overall, p2 = pearsonr(semantic, llm_overall)
        corr_rouge_correct, p3 = pearsonr(rouge, llm_correctness)
        corr_semantic_correct, p4 = pearsonr(semantic, llm_correctness)
        
        print("Korelacja z LLM Judge (Overall):")
        print(f"   • ROUGE-1 F1:          r = {corr_rouge_overall:+.3f} (p={p1:.4f})")
        print(f"   • Semantic Similarity: r = {corr_semantic_overall:+.3f} (p={p2:.4f})")
        
        print("\nKorelacja z LLM Judge (Correctness):")
        print(f"   • ROUGE-1 F1:          r = {corr_rouge_correct:+.3f} (p={p3:.4f})")
        print(f"   • Semantic Similarity: r = {corr_semantic_correct:+.3f} (p={p4:.4f})")
        
        # Interpretacja
        diff = corr_semantic_overall - corr_rouge_overall
        print(f"\n💡 INTERPRETACJA:")
        if diff > 0.05:
            print(f"   ✅ Semantic Similarity lepiej koreluje z LLM Judge (+{diff:.3f})")
            print(f"      → Semantic lepiej przewiduje rzeczywistą jakość odpowiedzi!")
        elif diff < -0.05:
            print(f"   ℹ️  ROUGE-1 lepiej koreluje z LLM Judge (+{-diff:.3f})")
        else:
            print(f"   ➡️  Obie metryki podobnie korelują z LLM Judge (różnica: {diff:.3f})")
        
        print(f"\n{'='*70}\n")
        
        return {
            'rouge_vs_llm_overall': corr_rouge_overall,
            'semantic_vs_llm_overall': corr_semantic_overall,
            'rouge_vs_llm_correctness': corr_rouge_correct,
            'semantic_vs_llm_correctness': corr_semantic_correct
        }
        
    except ImportError:
        print("⚠️  Zainstaluj scipy: pip install scipy")
        return None


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              EVALUATE GENERATION                             ║
╚══════════════════════════════════════════════════════════════╝

Użycie:
    python evaluate_generation.py <pdf_path> <dataset_path>
    python evaluate_generation.py <pdf_path> <dataset_path> --llm-judge

Przykład:
    python evaluate_generation.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json
    python evaluate_generation.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json --llm-judge
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    dataset_path = sys.argv[2]
    use_llm_judge = '--llm-judge' in sys.argv
    
    # Wczytaj dataset
    print(f"\n📂 Ładowanie datasetu: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Obsłuż oba formaty
    if 'data' in data:
        dataset = data['data']
        config = data.get('metadata', {}).get('baseline_config', {})
    else:
        dataset = data
        config = {"chunk_size": 800, "chunk_overlap": 100}
    
    print(f"   ✓ Załadowano {len(dataset)} pytań")
    
    # Stwórz pipeline
    print(f"\n🔧 Tworzenie pipeline...")
    
    pipeline = RAGPipeline(
        chunk_size=config.get('chunk_size', 800),
        chunk_overlap=config.get('chunk_overlap', 100),
        k=5
    )
    pipeline.process_document(pdf_path)
    print(f"   ✓ Utworzono {pipeline.num_chunks} chunków")
    
    # Ewaluacja
    results = evaluate_generation(pipeline, dataset, use_llm_judge=use_llm_judge)
    
    # Analiza korelacji (jeśli LLM Judge)
    if use_llm_judge:
        correlations = compare_metrics_correlation(results)
        if correlations:
            results['correlations'] = correlations
    
    # Zapisz wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_with_llm_judge" if use_llm_judge else ""
    output_file = f"generation_results{suffix}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Wyniki zapisane: {output_file}")