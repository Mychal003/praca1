"""
Eksperymenty z różnymi konfiguracjami RAG.
Testuje: chunk_size, k, overlap

UWAGA: Retrieval metrics tylko dla baseline config!
       (różne chunk_size = różne chunk_ids = nieporównywalne)

Użycie:
    python run_experiments.py <pdf_path> <dataset_path>
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

# Semantic model
print("🤖 Ładuję model Semantic Similarity...")
from sentence_transformers import SentenceTransformer, util
SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')
print("✅ Model załadowany!\n")


# ============================================================================
# METRYKI (tylko generation - szybkie)
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
# KONFIGURACJE EKSPERYMENTÓW
# ============================================================================

EXPERIMENTS = {
    "chunk_size": [
        {"name": "Tiny (300)", "chunk_size": 300, "chunk_overlap": 50, "k": 5},
        {"name": "Small (500)", "chunk_size": 500, "chunk_overlap": 100, "k": 5},
        {"name": "Medium (800)", "chunk_size": 800, "chunk_overlap": 100, "k": 5},
        {"name": "Large (1200)", "chunk_size": 1200, "chunk_overlap": 150, "k": 5},
        {"name": "XLarge (1500)", "chunk_size": 1500, "chunk_overlap": 200, "k": 5},
    ],
    "k_values": [
        {"name": "k=1", "chunk_size": 800, "chunk_overlap": 100, "k": 1},
        {"name": "k=3", "chunk_size": 800, "chunk_overlap": 100, "k": 3},
        {"name": "k=5", "chunk_size": 800, "chunk_overlap": 100, "k": 5},
        {"name": "k=7", "chunk_size": 800, "chunk_overlap": 100, "k": 7},
        {"name": "k=10", "chunk_size": 800, "chunk_overlap": 100, "k": 10},
    ],
    "overlap": [
        {"name": "No overlap (0)", "chunk_size": 800, "chunk_overlap": 0, "k": 5},
        {"name": "Small (50)", "chunk_size": 800, "chunk_overlap": 50, "k": 5},
        {"name": "Medium (100)", "chunk_size": 800, "chunk_overlap": 100, "k": 5},
        {"name": "Large (200)", "chunk_size": 800, "chunk_overlap": 200, "k": 5},
        {"name": "XLarge (300)", "chunk_size": 800, "chunk_overlap": 300, "k": 5},
    ]
}


# ============================================================================
# EWALUACJA POJEDYNCZEJ KONFIGURACJI
# ============================================================================

def evaluate_config(
    pdf_path: str,
    config: Dict,
    dataset: List[Dict],
    verbose: bool = True
) -> Dict:
    """Ewaluuje pojedynczą konfigurację."""
    
    # Stwórz pipeline
    pipeline = RAGPipeline(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        k=config['k']
    )
    pipeline.process_document(pdf_path)
    
    if verbose:
        print(f"   Chunks: {pipeline.num_chunks}")
    
    # Ewaluacja
    results = []
    total_latency = 0
    
    for item in dataset:
        start = time.time()
        try:
            generated = pipeline.query(item['question'])
        except:
            generated = "ERROR"
        latency = time.time() - start
        total_latency += latency
        
        results.append({
            'rouge1_f1': rouge_1_f1(generated, item['expected_answer']),
            'semantic_similarity': semantic_similarity(generated, item['expected_answer']),
            'latency': latency
        })
    
    # Agregacja
    avg_rouge = sum(r['rouge1_f1'] for r in results) / len(results)
    avg_semantic = sum(r['semantic_similarity'] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    
    return {
        'config': config,
        'num_chunks': pipeline.num_chunks,
        'avg_rouge1_f1': avg_rouge,
        'avg_semantic_similarity': avg_semantic,
        'avg_latency': avg_latency,
        'detailed': results
    }


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def run_experiments(pdf_path: str, dataset: List[Dict]) -> Dict:
    """Uruchamia wszystkie eksperymenty."""
    
    all_results = {}
    
    for exp_name, configs in EXPERIMENTS.items():
        print(f"\n{'='*70}")
        print(f"🧪 EKSPERYMENT: {exp_name.upper()}")
        print(f"{'='*70}")
        
        exp_results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] {config['name']}")
            print(f"   chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}, k={config['k']}")
            
            result = evaluate_config(pdf_path, config, dataset)
            exp_results.append(result)
            
            print(f"   ROUGE: {result['avg_rouge1_f1']:.3f} | Semantic: {result['avg_semantic_similarity']:.3f} | Latency: {result['avg_latency']:.2f}s")
        
        # Podsumowanie eksperymentu
        print(f"\n{'-'*70}")
        print(f"📊 Podsumowanie: {exp_name}")
        print(f"{'-'*70}")
        print(f"{'Konfiguracja':<20} {'ROUGE-1':<12} {'Semantic':<12} {'Latency':<12} {'Chunks'}")
        print("-" * 70)
        
        for res in exp_results:
            name = res['config']['name']
            print(f"{name:<20} {res['avg_rouge1_f1']:<12.3f} {res['avg_semantic_similarity']:<12.3f} {res['avg_latency']:<12.2f} {res['num_chunks']}")
        
        # Najlepsza konfiguracja
        best = max(exp_results, key=lambda x: x['avg_rouge1_f1'])
        print(f"\n🏆 Najlepsza: {best['config']['name']} (ROUGE: {best['avg_rouge1_f1']:.3f})")
        
        all_results[exp_name] = {
            'results': exp_results,
            'best': best['config']['name']
        }
    
    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              RUN EXPERIMENTS                                 ║
╚══════════════════════════════════════════════════════════════╝

Użycie:
    python run_experiments.py <pdf_path> <dataset_path>

Przykład:
    python run_experiments.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json

Testuje:
    - chunk_size: 300, 500, 800, 1200, 1500
    - k: 1, 3, 5, 7, 10
    - overlap: 0, 50, 100, 200, 300
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    dataset_path = sys.argv[2]
    
    # Wczytaj dataset
    print(f"\n📂 Ładowanie datasetu: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'data' in data:
        dataset = data['data']
    else:
        dataset = data
    
    print(f"   ✓ Załadowano {len(dataset)} pytań")
    
    # Uruchom eksperymenty
    print(f"\n🚀 Rozpoczynam eksperymenty...")
    print(f"   Łącznie konfiguracji: {sum(len(c) for c in EXPERIMENTS.values())}")
    
    start_time = time.time()
    results = run_experiments(pdf_path, dataset)
    total_time = time.time() - start_time
    
    # Podsumowanie końcowe
    print(f"\n{'='*70}")
    print("🏆 PODSUMOWANIE KOŃCOWE")
    print(f"{'='*70}")
    
    for exp_name, exp_data in results.items():
        print(f"\n{exp_name.upper()}:")
        print(f"   Najlepsza konfiguracja: {exp_data['best']}")
    
    print(f"\n⏱️  Całkowity czas: {total_time/60:.1f} minut")
    
    # Zapisz wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiments_results_{timestamp}.json"
    
    # Przygotuj do zapisu (usuń detailed dla czytelności)
    save_results = {}
    for exp_name, exp_data in results.items():
        save_results[exp_name] = {
            'best': exp_data['best'],
            'results': [
                {
                    'config': r['config'],
                    'num_chunks': r['num_chunks'],
                    'avg_rouge1_f1': r['avg_rouge1_f1'],
                    'avg_semantic_similarity': r['avg_semantic_similarity'],
                    'avg_latency': r['avg_latency']
                }
                for r in exp_data['results']
            ]
        }
    
    save_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'pdf_path': pdf_path,
        'dataset_path': dataset_path,
        'num_questions': len(dataset),
        'total_time_seconds': total_time
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Wyniki zapisane: {output_file}")
    
    # Automatyczna analiza błędów
    print("\n🔍 Uruchamiam analizę błędów...")
    analyze_error_patterns(save_results)


# ============================================================================
# ANALIZA BŁĘDÓW
# ============================================================================

def analyze_error_patterns(results: Dict):
    """
    Analizuje wzorce błędów - które pytania są najtrudniejsze/najłatwiejsze.
    """
    print(f"\n{'='*70}")
    print("🔍 ANALIZA BŁĘDÓW")
    print(f"{'='*70}")
    
    # Zbierz wszystkie detailed results
    all_questions = {}
    
    for exp_name in ['chunk_size', 'k_values', 'overlap']:
        if exp_name not in results:
            continue
        
        for config_result in results[exp_name]['results']:
            if 'detailed' not in config_result:
                continue
            
            for detail in config_result.get('detailed', []):
                q = detail.get('question', '')
                if not q:
                    continue
                
                if q not in all_questions:
                    all_questions[q] = []
                all_questions[q].append(detail.get('rouge1_f1', 0))
    
    if not all_questions:
        print("⚠️  Brak detailed results do analizy")
        return
    
    # Oblicz średnią dla każdego pytania
    avg_scores = {
        q: sum(scores) / len(scores) 
        for q, scores in all_questions.items()
    }
    
    # Sortuj
    sorted_questions = sorted(avg_scores.items(), key=lambda x: x[1])
    
    print("\n🔴 TOP 5 NAJTRUDNIEJSZYCH PYTAŃ (najniższy ROUGE-1):\n")
    for i, (question, score) in enumerate(sorted_questions[:5], 1):
        print(f"   {i}. ROUGE: {score:.3f}")
        print(f"      {question[:70]}...\n")
    
    print("🟢 TOP 5 NAJŁATWIEJSZYCH PYTAŃ (najwyższy ROUGE-1):\n")
    for i, (question, score) in enumerate(reversed(sorted_questions[-5:]), 1):
        print(f"   {i}. ROUGE: {score:.3f}")
        print(f"      {question[:70]}...\n")
    
    # Statystyki ogólne
    all_scores = list(avg_scores.values())
    print(f"{'='*70}")
    print("📊 STATYSTYKI OGÓLNE")
    print(f"{'='*70}")
    print(f"   Liczba pytań:     {len(all_scores)}")
    print(f"   Min ROUGE:        {min(all_scores):.3f}")
    print(f"   Max ROUGE:        {max(all_scores):.3f}")
    print(f"   Średnia:          {np.mean(all_scores):.3f}")
    print(f"   Mediana:          {np.median(all_scores):.3f}")
    print(f"   Odchylenie std:   {np.std(all_scores):.3f}")
    print(f"{'='*70}\n")