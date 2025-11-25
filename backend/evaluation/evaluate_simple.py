import json
import sys
import time
from datetime import datetime
from typing import List, Dict
import re
from collections import Counter
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from evaluation.retrieval_metrics import evaluate_retrieval
from collections import Counter
from evaluation.llm_judge import LLMJudge
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
load_dotenv()
print("🤖 Ładuję model Semantic Similarity (to może potrwać chwilę przy pierwszym uruchomieniu)...")
SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')
print("✅ Model załadowany!\n")
# ============================================================================
# METRYKI
# ============================================================================

from collections import Counter

def rouge_1_f1(prediction: str, reference: str) -> float:
    pred_words = Counter(prediction.lower().split())
    ref_words = Counter(reference.lower().split())
    
    if not pred_words or not ref_words:
        return 0.0
    
    # Overlap = suma minimum counts dla każdego słowa
    overlap = sum((pred_words & ref_words).values())
    
    precision = overlap / sum(pred_words.values())
    recall = overlap / sum(ref_words.values())
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def token_overlap(prediction: str, reference: str) -> float:
    """
    Prosty % wspólnych słów - łatwa do zrozumienia metryka.
    """
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    overlap = len(pred_words & ref_words)
    return overlap / len(ref_words)

def semantic_similarity(prediction: str, reference: str) -> float:
    """
    Semantic Similarity: Miara podobieństwa semantycznego (0-1).
    
    Używa sentence-transformers do obliczenia jak bardzo dwa teksty
    są semantycznie podobne, niezależnie od użytych słów.
    
    1.0 = semantycznie identyczne
    0.0 = całkowicie różne
    
    Przykład:
        "I don't have this info" vs "Document does not provide this information"
        ROUGE-1: ~0.1 (słabe)
        Semantic: ~0.85 (świetne!)
    """
    if not prediction or not reference:
        return 0.0
    
    try:
        # Embeddingi
        emb1 = SEMANTIC_MODEL.encode(prediction, convert_to_tensor=True)
        emb2 = SEMANTIC_MODEL.encode(reference, convert_to_tensor=True)
        
        # Cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        return max(0.0, min(1.0, similarity))  # Clamp do [0, 1]
    except Exception as e:
        print(f"⚠️  Błąd semantic similarity: {e}")
        return 0.0


# ============================================================================
# EVALUATOR (ZMODYFIKOWANA WERSJA Z LLM JUDGE)
# ============================================================================

def evaluate_system(
    rag_pipeline, 
    test_dataset: List[Dict] = None, 
    evaluate_retrieval_metrics: bool = True,
    use_llm_judge: bool = False,
    llm_judge_model: str = "chatgpt-4o-latest"
):
    """
    Główna funkcja ewaluacji - z POPRAWNYMI retrieval metrics i OPCJONALNYM LLM Judge!
    
    Args:
        rag_pipeline: Instancja RAGPipeline
        test_dataset: Dataset z pytaniami i expected answers
        evaluate_retrieval_metrics: Czy ewaluować retrieval metrics
        use_llm_judge: Czy użyć LLM as Judge (DROŻSZE, ale dokładniejsze!)
        llm_judge_model: Model LLM do użycia jako judge
    """
    if test_dataset is None:
        test_dataset = TEST_DATASET
    
    print(f"\n{'='*60}")
    print(f"🚀 EWALUACJA - {len(test_dataset)} pytań")
    if evaluate_retrieval_metrics:
        print("   (Including RETRIEVAL METRICS)")
    if use_llm_judge:
        print(f"   (Including LLM JUDGE - model: {llm_judge_model})")
    print(f"{'='*60}\n")
    
    # 🆕 Inicjalizuj LLM Judge jeśli potrzebny
    llm_judge = None
    if use_llm_judge:
        print("🤖 Initializing LLM Judge...")
        llm_judge = LLMJudge(model=llm_judge_model)
        print("✅ LLM Judge ready!\n")
    
    results = []
    all_retrieval_metrics = []
    
    for i, item in enumerate(test_dataset, 1):
        print(f"[{i}/{len(test_dataset)}] {item['question'][:50]}...")
        
        start = time.time()
        
        # Generuj odpowiedź
        try:
            generated = rag_pipeline.query(item['question'])
        except Exception as e:
            print(f"  ❌ Błąd: {e}")
            generated = "ERROR"
        
        latency = time.time() - start
        
        # GENERATION METRICS
        rouge1 = rouge_1_f1(generated, item['expected_answer'])
        overlap = token_overlap(generated, item['expected_answer'])
        semantic_sim = semantic_similarity(generated, item['expected_answer'])
        
        result = {
            "question": item['question'],
            "expected": item['expected_answer'],
            "generated": generated,
            "rouge1_f1": rouge1,
            "token_overlap": overlap,
            "semantic_similarity": semantic_sim,
            "latency": latency
        }
        
        # 🆕 LLM JUDGE EVALUATION
        if use_llm_judge and llm_judge and generated != "ERROR":
            # Pobierz kontekst dla groundedness
            sources = rag_pipeline.get_sources(item['question'], k=5)
            context = "\n\n".join([s['text'] for s in sources])
            
            # Oceń odpowiedź
            llm_scores = llm_judge.evaluate_answer(
                question=item['question'],
                generated_answer=generated,
                reference_answer=item['expected_answer'],
                context=context
            )
            
            result['llm_judge_scores'] = llm_scores
        
        # RETRIEVAL METRICS (POPRAWIONE!)
        if evaluate_retrieval_metrics and 'relevant_chunk_indices' in item:
            # Pobierz retrieved chunks z ich chunk_id
            retrieved_sources = rag_pipeline.get_sources(item['question'], k=20)
            
            # PRAWIDŁOWE: wyciągnij chunk_id z retrieved sources
            retrieved_chunk_ids = [src['chunk_id'] for src in retrieved_sources]
            
            # Ground truth relevant chunk IDs
            relevant_chunk_ids = item['relevant_chunk_indices']
            
            # 🔍 DEBUG: Sprawdź czy chunk_ids mają sens
            if -1 in retrieved_chunk_ids:
                print(f"  ⚠️  WARNING: Some chunks have chunk_id=-1!")
            
            # Sprawdź czy chunk_ids są w sensownym zakresie
            if retrieved_chunk_ids and max(retrieved_chunk_ids) >= rag_pipeline.num_chunks:
                print(f"  ⚠️  WARNING: chunk_id out of range!")
            
            ret_metrics = evaluate_retrieval(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k_values=[1, 3, 5, 10]
            )
            
            result['retrieval_metrics'] = ret_metrics
            all_retrieval_metrics.append(ret_metrics)
            
            # Print z wszystkimi metrykami
            if use_llm_judge and 'llm_judge_scores' in result:
                print(f"  ✓ ROUGE: {rouge1:.3f} | Semantic: {semantic_sim:.3f} | "
                      f"LLM: {result['llm_judge_scores']['overall']:.2f} | "
                      f"P@5: {ret_metrics['precision@5']:.3f} | {latency:.2f}s\n")
            else:
                print(f"  ✓ ROUGE: {rouge1:.3f} | Semantic: {semantic_sim:.3f} | "
                      f"P@5: {ret_metrics['precision@5']:.3f} | R@5: {ret_metrics['recall@5']:.3f} | {latency:.2f}s\n")
        else:
            # Print bez retrieval metrics
            if use_llm_judge and 'llm_judge_scores' in result:
                print(f"  ✓ ROUGE: {rouge1:.3f} | Semantic: {semantic_sim:.3f} | "
                      f"LLM Overall: {result['llm_judge_scores']['overall']:.2f} | {latency:.2f}s\n")
            else:
                print(f"  ✓ ROUGE: {rouge1:.3f} | Semantic: {semantic_sim:.3f} | {latency:.2f}s\n")
        
        results.append(result)
    
    # =========================================================================
    # OBLICZ ŚREDNIE - Generation Metrics
    # =========================================================================
    avg_rouge1 = sum(r['rouge1_f1'] for r in results) / len(results)
    avg_overlap = sum(r['token_overlap'] for r in results) / len(results)
    avg_semantic = sum(r.get('semantic_similarity', 0) for r in results) / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    summary = {
        "avg_rouge1_f1": avg_rouge1,
        "avg_token_overlap": avg_overlap,
        "avg_semantic_similarity": avg_semantic,
        "avg_latency": avg_latency,
        "num_questions": len(results)
    }
    
    # =========================================================================
    # OBLICZ ŚREDNIE - Retrieval Metrics
    # =========================================================================
    if all_retrieval_metrics:
        metric_keys = all_retrieval_metrics[0].keys()
        for key in metric_keys:
            avg_value = sum(m[key] for m in all_retrieval_metrics) / len(all_retrieval_metrics)
            summary[f'avg_{key}'] = avg_value
    
    # =========================================================================
    # 🆕 OBLICZ ŚREDNIE - LLM Judge Metrics
    # =========================================================================
    if use_llm_judge:
        llm_results = [r for r in results if 'llm_judge_scores' in r]
        
        if llm_results:
            # Zbierz wszystkie score keys
            score_keys = llm_results[0]['llm_judge_scores'].keys()
            
            for key in score_keys:
                scores = [r['llm_judge_scores'][key] for r in llm_results 
                         if r['llm_judge_scores'][key] is not None]
                if scores:
                    summary[f'avg_llm_{key}'] = sum(scores) / len(scores)
    
    # =========================================================================
    # PODSUMOWANIE WYNIKÓW
    # =========================================================================
    print(f"\n{'='*60}")
    print("📊 PODSUMOWANIE")
    print(f"{'='*60}")
    
    print("\n🎯 GENERATION METRICS:")
    print(f"  ROUGE-1 F1:          {avg_rouge1:.3f}")
    print(f"  Token Overlap:       {avg_overlap:.3f}")
    print(f"  Semantic Similarity: {avg_semantic:.3f}")
    print(f"  Latencja:            {avg_latency:.2f}s")
    
    if all_retrieval_metrics:
        print("\n🔍 RETRIEVAL METRICS:")
        for key, value in summary.items():
            if key.startswith('avg_') and 'retrieval' not in key.lower() and key not in ['avg_rouge1_f1', 'avg_token_overlap', 'avg_semantic_similarity', 'avg_latency'] and not key.startswith('avg_llm_'):
                metric_name = key.replace('avg_', '')
                print(f"  {metric_name:<20} {value:.3f}")
    
    # 🆕 LLM Judge Results
    if use_llm_judge:
        print("\n🤖 LLM JUDGE METRICS:")
        llm_metrics = {k: v for k, v in summary.items() if k.startswith('avg_llm_')}
        for key, value in llm_metrics.items():
            metric_name = key.replace('avg_llm_', '').title()
            print(f"  {metric_name:<20} {value:.3f}")
    
    print(f"{'='*60}\n")
    
    return {
        "summary": summary,
        "detailed_results": results
    }


# ============================================================================
# 🆕 NOWA FUNKCJA: Porównanie metryk
# ============================================================================

def compare_metrics_correlation(results: Dict):
    """
    Analizuje korelację między różnymi metrykami.
    Pokazuje która metryka najlepiej przewiduje jakość odpowiedzi.
    """
    detailed = results['detailed_results']
    
    # Sprawdź czy mamy LLM Judge scores
    has_llm = any('llm_judge_scores' in r for r in detailed)
    
    if not has_llm:
        print("⚠️  Brak wyników LLM Judge - nie można obliczyć korelacji")
        return
    
    print(f"\n{'='*60}")
    print("📊 ANALIZA KORELACJI METRYK")
    print(f"{'='*60}\n")
    
    # Zbierz metryki
    rouge_scores = [r['rouge1_f1'] for r in detailed if 'llm_judge_scores' in r]
    semantic_scores = [r.get('semantic_similarity', 0) for r in detailed if 'llm_judge_scores' in r]
    llm_overall = [r['llm_judge_scores']['overall'] for r in detailed if 'llm_judge_scores' in r]
    llm_correctness = [r['llm_judge_scores']['correctness'] for r in detailed if 'llm_judge_scores' in r]
    
    # Oblicz korelacje (Pearson)
    try:
        from scipy.stats import pearsonr
        
        corr_rouge_overall, _ = pearsonr(rouge_scores, llm_overall)
        corr_semantic_overall, _ = pearsonr(semantic_scores, llm_overall)
        corr_rouge_correctness, _ = pearsonr(rouge_scores, llm_correctness)
        corr_semantic_correctness, _ = pearsonr(semantic_scores, llm_correctness)
        
        print("Korelacja z LLM Judge (Overall Quality):")
        print(f"  • ROUGE-1 F1:          {corr_rouge_overall:+.3f}")
        print(f"  • Semantic Similarity: {corr_semantic_overall:+.3f}")
        
        print("\nKorelacja z LLM Judge (Correctness):")
        print(f"  • ROUGE-1 F1:          {corr_rouge_correctness:+.3f}")
        print(f"  • Semantic Similarity: {corr_semantic_correctness:+.3f}")
        
        # Interpretacja
        print("\n💡 INTERPRETACJA:")
        if corr_semantic_overall > corr_rouge_overall:
            diff = corr_semantic_overall - corr_rouge_overall
            print(f"  ✅ Semantic Similarity lepiej koreluje z LLM Judge (+{diff:.3f})")
            print(f"     → Semantic lepiej przewiduje jakość odpowiedzi!")
        else:
            diff = corr_rouge_overall - corr_semantic_overall
            print(f"  ℹ️  ROUGE-1 lepiej koreluje z LLM Judge (+{diff:.3f})")
        
        print(f"\n{'='*60}\n")
        
    except ImportError:
        print("⚠️  Zainstaluj scipy aby obliczyć korelacje: pip install scipy")


# ============================================================================
# EKSPERYMENTY (ZMODYFIKOWANA WERSJA)
# ============================================================================

def run_experiments(
    pdf_path: str, 
    use_llm_judge: bool = False,  # 🆕 NOWY PARAMETR
    llm_judge_sample_size: int = None  # 🆕 Opcjonalnie: użyj tylko N pytań dla LLM Judge
):
    """
    Uruchamia 3 podstawowe eksperymenty z różnymi konfiguracjami.
    
    Args:
        pdf_path: Ścieżka do PDF
        use_llm_judge: Czy użyć LLM Judge (droższe!)
        llm_judge_sample_size: Ile pytań użyć dla LLM Judge (None = wszystkie)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from src.rag_pipeline import RAGPipeline
    
    configs = [
        {"name": "Small chunks", "chunk_size": 500, "k": 3},
        {"name": "Medium chunks", "chunk_size": 800, "k": 3},
        {"name": "Large chunks", "chunk_size": 1200, "k": 3},
    ]
    
    print(f"\n{'='*60}")
    print("🔬 EKSPERYMENTY - 3 konfiguracje")
    if use_llm_judge:
        if llm_judge_sample_size:
            print(f"   (Z LLM Judge - sample: {llm_judge_sample_size} pytań)")
        else:
            print(f"   (Z LLM Judge - wszystkie pytania)")
    print(f"{'='*60}\n")
    
    # 🆕 Przygotuj dataset (opcjonalnie sample dla LLM Judge)
    test_dataset = TEST_DATASET
    if use_llm_judge and llm_judge_sample_size and llm_judge_sample_size < len(TEST_DATASET):
        print(f"💡 Używam sampla {llm_judge_sample_size} pytań dla LLM Judge (oszczędność kosztów)\n")
        test_dataset = TEST_DATASET[:llm_judge_sample_size]
    
    all_results = []
    
    for config in configs:
        print(f"\n📦 {config['name']}")
        print(f"   chunk_size={config['chunk_size']}, k={config['k']}\n")
        
        # Stwórz pipeline
        pipeline = RAGPipeline(
            chunk_size=config['chunk_size'],
            chunk_overlap=100,
            k=config['k']
        )
        
        # Przetwórz dokument
        pipeline.process_document(pdf_path)
        
        # 🆕 Ewaluacja (z opcjonalnym LLM Judge)
        results = evaluate_system(
            pipeline, 
            test_dataset=test_dataset,
            evaluate_retrieval_metrics=False,  # Wyłącz dla szybkości
            use_llm_judge=use_llm_judge
        )
        
        # 🆕 Analiza korelacji (jeśli LLM Judge)
        if use_llm_judge:
            compare_metrics_correlation(results)
        
        # Zapisz
        results['config'] = config
        all_results.append(results)
    
    # Porównanie
    print(f"\n{'='*60}")
    print("📊 PORÓWNANIE")
    print(f"{'='*60}")
    
    if use_llm_judge:
        # Rozszerzona tabela z LLM Judge
        print(f"{'Konfiguracja':<20} {'ROUGE-1':<10} {'Semantic':<10} {'LLM':<10} {'Latency'}")
        print("-" * 60)
        
        for res in all_results:
            name = res['config']['name']
            rouge = res['summary']['avg_rouge1_f1']
            semantic = res['summary']['avg_semantic_similarity']
            llm_overall = res['summary'].get('avg_llm_overall', 0)
            latency = res['summary']['avg_latency']
            
            print(f"{name:<20} {rouge:<10.3f} {semantic:<10.3f} {llm_overall:<10.3f} {latency:.2f}s")
    else:
        # Standardowa tabela
        print(f"{'Konfiguracja':<20} {'ROUGE-1':<12} {'Overlap':<12} {'Latency'}")
        print("-" * 60)
        
        for res in all_results:
            name = res['config']['name']
            rouge = res['summary']['avg_rouge1_f1']
            overlap = res['summary']['avg_token_overlap']
            latency = res['summary']['avg_latency']
            
            print(f"{name:<20} {rouge:<12.3f} {overlap:<12.3f} {latency:.2f}s")
    
    # Znajdź najlepszy
    if use_llm_judge:
        best = max(all_results, key=lambda x: x['summary'].get('avg_llm_overall', 0))
        print(f"\n🏆 Najlepszy (według LLM Judge): {best['config']['name']}")
        print(f"   LLM Overall: {best['summary'].get('avg_llm_overall', 0):.3f}")
        print(f"   ROUGE-1 F1: {best['summary']['avg_rouge1_f1']:.3f}")
    else:
        best = max(all_results, key=lambda x: x['summary']['avg_rouge1_f1'])
        print(f"\n🏆 Najlepszy: {best['config']['name']}")
        print(f"   ROUGE-1 F1: {best['summary']['avg_rouge1_f1']:.3f}")
    
    # Zapisz do JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Wyniki zapisane: {filename}\n")
    
    return all_results


# ============================================================================
# UŻYCIE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         PROSTY SYSTEM EWALUACJI RAG                      ║
    ╚══════════════════════════════════════════════════════════╝
    
    Użycie:
    
    1. POJEDYNCZA EWALUACJA:
       from evaluate_simple import evaluate_system, TEST_DATASET
       
       pipeline = RAGPipeline()
       pipeline.process_document("doc.pdf")
       
       # Standardowa ewaluacja
       results = evaluate_system(pipeline, TEST_DATASET)
       
       # Z LLM Judge
       results = evaluate_system(pipeline, TEST_DATASET, use_llm_judge=True)
    
    2. EKSPERYMENTY (3 konfiguracje):
       from evaluate_simple import run_experiments
       
       # Bez LLM Judge
       run_experiments("doc.pdf")
       
       # Z LLM Judge (tylko 10 pytań)
       run_experiments("doc.pdf", use_llm_judge=True, llm_judge_sample_size=10)
    
    3. CUSTOM PYTANIA:
       custom_dataset = [
           {
               "question": "Twoje pytanie?",
               "expected_answer": "Oczekiwana odpowiedź",
               "category": "factual"
           }
       ]
       
       results = evaluate_system(pipeline, custom_dataset, use_llm_judge=True)
    
    ══════════════════════════════════════════════════════════
    """)
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # 🆕 Sprawdź flagę --llm-judge
        use_llm = '--llm-judge' in sys.argv
        
        if use_llm:
            print("🤖 Uruchamiam eksperymenty Z LLM Judge (sample 10 pytań)")
            run_experiments(pdf_path, use_llm_judge=True, llm_judge_sample_size=None)
        else:
            print("📊 Uruchamiam standardowe eksperymenty (bez LLM Judge)")
            run_experiments(pdf_path)
    else:
        print("📖 Aby uruchomić eksperymenty: python evaluate_simple.py <pdf_path>")
        print("📖 Z LLM Judge: python evaluate_simple.py <pdf_path> --llm-judge")
