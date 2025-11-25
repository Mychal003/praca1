"""
Rozszerzone eksperymenty dla pracy inżynierskiej.
Testuje różne konfiguracje RAG pipeline.
"""

import json
import time
from datetime import datetime
from typing import List, Dict
import sys
import os

# Dodaj parent directory do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline

# Wczytaj ground truth dataset
try:
    with open('test_dataset_ground_truth.json', 'r', encoding='utf-8') as f:
        GROUND_TRUTH_DATASET = json.load(f)
    GROUND_TRUTH_AVAILABLE = True
except FileNotFoundError:
    GROUND_TRUTH_DATASET = None
    GROUND_TRUTH_AVAILABLE = False
    print("⚠️  WARNING: test_dataset_ground_truth.json not found. Using manual dataset only.")


def run_comprehensive_experiments(pdf_path: str, use_ground_truth: bool = True):
    """
    Kompleksowe eksperymenty testujące różne aspekty systemu RAG.
    
    WAŻNE: Retrieval metrics są ewaluowane TYLKO dla konfiguracji bazowej,
    ponieważ różne chunk_size generują różne zbiory dokumentów.
    """
    # Wybierz dataset
    if use_ground_truth and GROUND_TRUTH_AVAILABLE:
        dataset = GROUND_TRUTH_DATASET
        dataset_name = "Ground Truth Dataset"
        print(f"\n{'='*70}")
        print("🔬 KOMPLEKSOWE EKSPERYMENTY RAG")
        print(f"📊 Dataset: {dataset_name} (wyekstrahowany z dokumentu)")
        print(f"{'='*70}\n")
    else:
        dataset = TEST_DATASET
        dataset_name = "Manual Dataset"
        print(f"\n{'='*70}")
        print("🔬 KOMPLEKSOWE EKSPERYMENTY RAG")
        print(f"📊 Dataset: {dataset_name} (ręczne expected answers)")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # 🆕 KONFIGURACJA BAZOWA (dla retrieval metrics)
    # =========================================================================
    BASE_CONFIG = {
        "name": "BASELINE (for retrieval metrics)",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "k": 5
    }
    
    print("\n" + "="*70)
    print("🎯 KONFIGURACJA BAZOWA (z retrieval metrics)")
    print("="*70)
    print(f"   chunk_size={BASE_CONFIG['chunk_size']}")
    print(f"   overlap={BASE_CONFIG['chunk_overlap']}")
    print(f"   k={BASE_CONFIG['k']}")
    print(f"\n   ℹ️  Retrieval metrics będą ewaluowane TYLKO dla tej konfiguracji,")
    print(f"      ponieważ różne chunk_size generują różne zbiory dokumentów.")
    print("="*70)
    
    baseline_result = run_single_config_with_retrieval(
        pdf_path, 
        BASE_CONFIG, 
        dataset
    )
    
    # =========================================================================
    # EKSPERYMENT 1: Wpływ rozmiaru chunków (BEZ retrieval metrics)
    # =========================================================================
    
    chunk_size_configs = [
        {"name": "Tiny chunks (300)", "chunk_size": 300, "chunk_overlap": 50, "k": 3},
        {"name": "Small chunks (500)", "chunk_size": 500, "chunk_overlap": 100, "k": 3},
        {"name": "Medium chunks (800)", "chunk_size": 800, "chunk_overlap": 100, "k": 3},
        {"name": "Large chunks (1200)", "chunk_size": 1200, "chunk_overlap": 150, "k": 3},
        {"name": "Extra large (1500)", "chunk_size": 1500, "chunk_overlap": 200, "k": 3},
    ]
    
    print("\n" + "="*70)
    print("📦 EKSPERYMENT 1: Wpływ rozmiaru chunków")
    print("   (GENERATION METRICS ONLY)")
    print("="*70)
    
    chunk_results = run_config_batch(
        pdf_path, 
        chunk_size_configs, 
        "chunk_size", 
        dataset,
        evaluate_retrieval=False  # 🆕 Wyłącz retrieval
    )
    
    # =========================================================================
    # EKSPERYMENT 2: Wpływ liczby retrievanych dokumentów (k)
    # =========================================================================
    
    k_configs = [
        {"name": "k=1", "chunk_size": 800, "chunk_overlap": 100, "k": 1},
        {"name": "k=2", "chunk_size": 800, "chunk_overlap": 100, "k": 2},
        {"name": "k=3", "chunk_size": 800, "chunk_overlap": 100, "k": 3},
        {"name": "k=5", "chunk_size": 800, "chunk_overlap": 100, "k": 5},
        {"name": "k=7", "chunk_size": 800, "chunk_overlap": 100, "k": 7},
    ]
    
    print("\n" + "="*70)
    print("🔍 EKSPERYMENT 2: Wpływ liczby retrievanych dokumentów (k)")
    print("   (GENERATION METRICS ONLY)")
    print("="*70)
    
    k_results = run_config_batch(
        pdf_path, 
        k_configs, 
        "k", 
        dataset,
        evaluate_retrieval=False  # 🆕 Wyłącz retrieval
    )
    
    # =========================================================================
    # EKSPERYMENT 3: Wpływ overlap
    # =========================================================================
    
    overlap_configs = [
        {"name": "No overlap (0)", "chunk_size": 800, "chunk_overlap": 0, "k": 3},
        {"name": "Small overlap (50)", "chunk_size": 800, "chunk_overlap": 50, "k": 3},
        {"name": "Medium overlap (100)", "chunk_size": 800, "chunk_overlap": 100, "k": 3},
        {"name": "Large overlap (200)", "chunk_size": 800, "chunk_overlap": 200, "k": 3},
    ]
    
    print("\n" + "="*70)
    print("🔗 EKSPERYMENT 3: Wpływ overlap między chunkami")
    print("   (GENERATION METRICS ONLY)")
    print("="*70)
    
    overlap_results = run_config_batch(
        pdf_path, 
        overlap_configs, 
        "overlap", 
        dataset,
        evaluate_retrieval=False  # 🆕 Wyłącz retrieval
    )
    
    # =========================================================================
    # 🆕 EKSPERYMENT 4: Retrieval Quality (TYLKO baseline config)
    # =========================================================================
    
    print("\n" + "="*70)
    print("🔍 EKSPERYMENT 4: Retrieval Quality Analysis")
    print("   (BASELINE CONFIG WITH FULL RETRIEVAL METRICS)")
    print("="*70)
    
    # baseline_result już obliczony wcześniej
    
    # =========================================================================
    # PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW
    # =========================================================================
    
    all_results = {
        "baseline_with_retrieval": baseline_result,  # 🆕 Osobna sekcja
        "chunk_size_experiment": chunk_results,
        "k_experiment": k_results,
        "overlap_experiment": overlap_results,
        "metadata": {
            "pdf_path": pdf_path,
            "dataset_used": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(chunk_results) + len(k_results) + len(overlap_results) + 1,
            "baseline_config": BASE_CONFIG,  # 🆕 Zapisz baseline
            "note": "Retrieval metrics evaluated only for baseline config due to chunk incompatibility"
        }
    }
    
    # Zapisz wszystkie wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"advanced_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("💾 WYNIKI ZAPISANE")
    print(f"{'='*70}")
    print(f"Plik: {filename}")
    print(f"Całkowita liczba eksperymentów: {all_results['metadata']['total_experiments']}")
    
    # Najlepsze konfiguracje (GENERATION tylko)
    print(f"\n{'='*70}")
    print("🏆 NAJLEPSZE KONFIGURACJE (Generation Metrics)")
    print(f"{'='*70}\n")
    
    best_chunk = max(chunk_results, key=lambda x: x['summary']['avg_rouge1_f1'])
    print(f"📦 Najlepszy chunk_size: {best_chunk['config']['chunk_size']}")
    print(f"   ROUGE-1 F1: {best_chunk['summary']['avg_rouge1_f1']:.3f}")
    print(f"   Semantic Sim: {best_chunk['summary']['avg_semantic_similarity']:.3f}")
    print(f"   Latencja: {best_chunk['summary']['avg_latency']:.2f}s\n")
    
    best_k = max(k_results, key=lambda x: x['summary']['avg_rouge1_f1'])
    print(f"🔍 Najlepsze k: {best_k['config']['k']}")
    print(f"   ROUGE-1 F1: {best_k['summary']['avg_rouge1_f1']:.3f}")
    print(f"   Semantic Sim: {best_k['summary']['avg_semantic_similarity']:.3f}")
    print(f"   Latencja: {best_k['summary']['avg_latency']:.2f}s\n")
    
    best_overlap = max(overlap_results, key=lambda x: x['summary']['avg_rouge1_f1'])
    print(f"🔗 Najlepszy overlap: {best_overlap['config']['chunk_overlap']}")
    print(f"   ROUGE-1 F1: {best_overlap['summary']['avg_rouge1_f1']:.3f}")
    print(f"   Semantic Sim: {best_overlap['summary']['avg_semantic_similarity']:.3f}")
    print(f"   Latencja: {best_overlap['summary']['avg_latency']:.2f}s\n")
    
    # 🆕 Podsumowanie Retrieval (baseline)
    if 'avg_precision@5' in baseline_result.get('summary', {}):
        print(f"{'='*70}")
        print("🎯 RETRIEVAL METRICS (Baseline Config)")
        print(f"{'='*70}\n")
        print(f"   Precision@5:  {baseline_result['summary'].get('avg_precision@5', 0):.3f}")
        print(f"   Recall@5:     {baseline_result['summary'].get('avg_recall@5', 0):.3f}")
        print(f"   F1@5:         {baseline_result['summary'].get('avg_f1@5', 0):.3f}")
        print(f"   MRR:          {baseline_result['summary'].get('avg_mrr', 0):.3f}")
        print(f"   NDCG@5:       {baseline_result['summary'].get('avg_ndcg@5', 0):.3f}\n")
    
    print(f"{'='*70}\n")
    
    return all_results


def run_single_config_with_retrieval(
    pdf_path: str, 
    config: Dict, 
    dataset: List[Dict]
) -> Dict:
    """
    Uruchamia POJEDYNCZĄ konfigurację Z retrieval metrics.
    Używane tylko dla baseline config.
    
    Args:
        pdf_path: Ścieżka do PDF
        config: Konfiguracja (chunk_size, overlap, k)
        dataset: Dataset z annotacjami
    
    Returns:
        Wyniki ewaluacji z retrieval metrics
    """
    print(f"\n🔧 Tworzę pipeline: {config['name']}")
    print(f"   chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}, k={config['k']}")
    
    try:
        # Stwórz pipeline
        pipeline = RAGPipeline(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            k=config['k']
        )
        
        # Przetwórz dokument
        pipeline.process_document(pdf_path)
        
        print(f"   📊 Chunks created: {pipeline.num_chunks}")
        
        # 🆕 Ewaluacja Z retrieval metrics
        result = evaluate_system(
            pipeline, 
            dataset, 
            evaluate_retrieval_metrics=True  # ← WŁĄCZ retrieval
        )
        
        result['config'] = config
        
        print(f"\n   ✅ Ewaluacja zakończona:")
        print(f"      ROUGE-1: {result['summary']['avg_rouge1_f1']:.3f}")
        print(f"      Semantic: {result['summary']['avg_semantic_similarity']:.3f}")
        
        if 'avg_precision@5' in result['summary']:
            print(f"      P@5: {result['summary']['avg_precision@5']:.3f}")
            print(f"      R@5: {result['summary']['avg_recall@5']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_config_batch(
    pdf_path: str, 
    configs: List[Dict], 
    experiment_name: str, 
    dataset: List[Dict],
    evaluate_retrieval: bool = False  # 🆕 Parametr kontrolny
) -> List[Dict]:
    """
    Uruchamia batch eksperymentów dla danej listy konfiguracji.
    
    Args:
        pdf_path: Ścieżka do PDF
        configs: Lista konfiguracji do przetestowania
        experiment_name: Nazwa eksperymentu (do logowania)
        dataset: Dataset do użycia w ewaluacji
        evaluate_retrieval: Czy ewaluować retrieval metrics (domyślnie False)
    
    Returns:
        Lista wyników dla każdej konfiguracji
    """
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")
        print(f"   chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}, k={config['k']}")
        
        try:
            # Stwórz pipeline
            pipeline = RAGPipeline(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap'],
                k=config['k']
            )
            
            # Przetwórz dokument
            pipeline.process_document(pdf_path)
            
            # 🆕 Ewaluacja (z lub bez retrieval)
            result = evaluate_system(
                pipeline, 
                dataset,
                evaluate_retrieval_metrics=evaluate_retrieval  # ← Kontrola retrieval
            )
            result['config'] = config
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Błąd: {e}")
            continue
    
    # Mini podsumowanie dla tego batcha
    print(f"\n{'─'*70}")
    print(f"📊 Podsumowanie: {experiment_name}")
    print(f"{'─'*70}")
    print(f"{'Konfiguracja':<25} {'ROUGE-1':<12} {'Semantic':<12} {'Latency'}")
    print("─" * 70)
    
    for res in results:
        name = res['config']['name']
        rouge = res['summary']['avg_rouge1_f1']
        semantic = res['summary'].get('avg_semantic_similarity', 0)
        latency = res['summary']['avg_latency']
        print(f"{name:<25} {rouge:<12.3f} {semantic:<12.3f} {latency:.2f}s")
    
    return results



def analyze_error_patterns(results_file: str):
    """
    Analizuje wzorce błędów w odpowiedziach systemu.
    Identyfikuje pytania, na które system radzi sobie najgorzej.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print("🔍 ANALIZA BŁĘDÓW")
    print(f"{'='*70}\n")
    
    # 🆕 Zbierz wyniki z baseline (pojedynczy dict)
    all_detailed = []
    
    if 'baseline_with_retrieval' in data:
        baseline = data['baseline_with_retrieval']
        if 'detailed_results' in baseline:
            all_detailed.extend(baseline['detailed_results'])
    
    # Zbierz wyniki z eksperymentów (listy)
    for exp_name in ['chunk_size_experiment', 'k_experiment', 'overlap_experiment']:
        if exp_name not in data:
            continue
        
        for result in data[exp_name]:
            if 'detailed_results' in result:
                all_detailed.extend(result['detailed_results'])
    
    if not all_detailed:
        print("⚠️  Brak detailed_results w danych!")
        return
    
    # Grupuj po pytaniach
    questions_performance = {}
    
    for detail in all_detailed:
        q = detail['question']
        if q not in questions_performance:
            questions_performance[q] = []
        questions_performance[q].append(detail['rouge1_f1'])
    
    # Oblicz średnią dla każdego pytania
    avg_performance = {
        q: sum(scores) / len(scores) 
        for q, scores in questions_performance.items()
    }
    
    # Sortuj od najgorszych
    sorted_questions = sorted(avg_performance.items(), key=lambda x: x[1])
    
    print("🔴 TOP 5 NAJTRUDNIEJSZYCH PYTAŃ (najniższy średni ROUGE-1):\n")
    
    for i, (question, avg_score) in enumerate(sorted_questions[:5], 1):
        print(f"{i}. Średni ROUGE-1: {avg_score:.3f}")
        print(f"   Pytanie: {question}\n")
    
    print("\n🟢 TOP 5 NAJŁATWIEJSZYCH PYTAŃ (najwyższy średni ROUGE-1):\n")
    
    for i, (question, avg_score) in enumerate(reversed(sorted_questions[-5:]), 1):
        print(f"{i}. Średni ROUGE-1: {avg_score:.3f}")
        print(f"   Pytanie: {question}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
═══════════════════════════════════════════════════════════════
         ROZSZERZONE EKSPERYMENTY RAG
═══════════════════════════════════════════════════════════════
Użycie:

1. Uruchom wszystkie eksperymenty (GROUND TRUTH - domyślnie):
   python evaluation/advanced_experiments.py <pdf_path>

2. Uruchom z MANUAL dataset:
   python evaluation/advanced_experiments.py <pdf_path> --manual

3. Analizuj błędy z istniejącego pliku wyników:
   python evaluation/advanced_experiments.py --analyze <results.json>

Przykłady:
   python evaluation/advanced_experiments.py uploads/Archer_D7UN_V1_UG.pdf
   python evaluation/advanced_experiments.py uploads/Archer_D7UN_V1_UG.pdf --manual

═══════════════════════════════════════════════════════════════
        """)
        sys.exit(1)
    
    if sys.argv[1] == '--analyze':
        if len(sys.argv) < 3:
            print("❌ Podaj plik z wynikami do analizy")
            sys.exit(1)
        analyze_error_patterns(sys.argv[2])
    else:
        pdf_path = sys.argv[1]
        
        # Sprawdź czy użytkownik chce manual dataset
        use_ground_truth = True
        if len(sys.argv) > 2 and sys.argv[2] == '--manual':
            use_ground_truth = False
            print("📝 Używam MANUAL dataset (ręczne expected answers)")
        else:
            print("📊 Używam GROUND TRUTH dataset (domyślnie)")
        
        results = run_comprehensive_experiments(pdf_path, use_ground_truth=use_ground_truth)
        
        # Automatyczna analiza błędów
        print("\n🔄 Uruchamiam analizę błędów...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_results_{timestamp}.json"
        analyze_error_patterns(filename)