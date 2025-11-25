"""
Ewaluacja RETRIEVAL - tylko metryki retrieval.
Precision@k, Recall@k, F1@k, MRR, NDCG@k

Użycie:
    python evaluate_retrieval.py <pdf_path> <dataset_path>
    python evaluate_retrieval.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline

# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Precision@k: % relevant w top-k"""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / k


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Recall@k: % znalezionych relevant"""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def f1_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """F1@k: harmonic mean of P@k and R@k"""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: List[int], relevant: List[int]) -> float:
    """MRR: 1/rank pierwszego relevant"""
    relevant_set = set(relevant)
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """NDCG@k: normalized discounted cumulative gain"""
    relevant_set = set(relevant)
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(retrieved[:k], start=1):
        rel = 1.0 if idx in relevant_set else 0.0
        dcg += rel / np.log2(i + 1)
    
    # IDCG
    ideal = [1.0] * min(len(relevant_set), k)
    idcg = sum(r / np.log2(i + 1) for i, r in enumerate(ideal, start=1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved: List[int], relevant: List[int]) -> float:
    """AP: average precision"""
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    num_relevant = 0
    sum_precisions = 0.0
    
    for k, idx in enumerate(retrieved, start=1):
        if idx in relevant_set:
            num_relevant += 1
            sum_precisions += num_relevant / k
    
    return sum_precisions / len(relevant_set)


# ============================================================================
# EWALUACJA
# ============================================================================

def evaluate_retrieval(
    pipeline: RAGPipeline,
    dataset: List[Dict],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Ewaluacja retrieval metrics.
    
    Args:
        pipeline: RAGPipeline z załadowanym dokumentem
        dataset: Dataset z 'question' i 'relevant_chunk_indices'
        k_values: Wartości k do ewaluacji
    
    Returns:
        Dict z wynikami
    """
    
    print(f"\n{'='*70}")
    print("🔍 EWALUACJA RETRIEVAL")
    print(f"{'='*70}")
    print(f"   Pytań: {len(dataset)}")
    print(f"   k values: {k_values}")
    print(f"   Retrieval k: {pipeline.k}")
    print(f"{'='*70}\n")
    
    all_metrics = []
    
    for i, item in enumerate(dataset, 1):
        question = item['question']
        relevant = item.get('relevant_chunk_indices', [])
        
        print(f"[{i}/{len(dataset)}] {question[:50]}...")
        
        # Pobierz retrieved chunks
        sources = pipeline.get_sources(question, k=max(k_values))
        retrieved = [s['chunk_id'] for s in sources]
        
        # Oblicz metryki
        metrics = {}
        for k in k_values:
            metrics[f'precision@{k}'] = precision_at_k(retrieved, relevant, k)
            metrics[f'recall@{k}'] = recall_at_k(retrieved, relevant, k)
            metrics[f'f1@{k}'] = f1_at_k(retrieved, relevant, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved, relevant, k)
        
        metrics['mrr'] = mrr(retrieved, relevant)
        metrics['ap'] = average_precision(retrieved, relevant)
        
        all_metrics.append({
            'question': question,
            'relevant_count': len(relevant),
            'retrieved': retrieved[:5],  # Top 5 dla referencji
            'metrics': metrics
        })
        
        # Print progress
        print(f"   P@5: {metrics['precision@5']:.3f} | R@5: {metrics['recall@5']:.3f} | MRR: {metrics['mrr']:.3f}")
    
    # Agregacja
    print(f"\n{'='*70}")
    print("📊 PODSUMOWANIE RETRIEVAL METRICS")
    print(f"{'='*70}\n")
    
    summary = {}
    metric_keys = all_metrics[0]['metrics'].keys()
    
    for key in metric_keys:
        values = [m['metrics'][key] for m in all_metrics]
        summary[f'avg_{key}'] = np.mean(values)
        summary[f'std_{key}'] = np.std(values)
    
    # Wyświetl
    print(f"{'Metryka':<20} {'Średnia':<12} {'Std Dev':<12}")
    print("-" * 44)
    
    for k in k_values:
        print(f"Precision@{k:<10} {summary[f'avg_precision@{k}']:<12.3f} {summary[f'std_precision@{k}']:<12.3f}")
    print()
    
    for k in k_values:
        print(f"Recall@{k:<13} {summary[f'avg_recall@{k}']:<12.3f} {summary[f'std_recall@{k}']:<12.3f}")
    print()
    
    for k in k_values:
        print(f"F1@{k:<16} {summary[f'avg_f1@{k}']:<12.3f} {summary[f'std_f1@{k}']:<12.3f}")
    print()
    
    for k in k_values:
        print(f"NDCG@{k:<14} {summary[f'avg_ndcg@{k}']:<12.3f} {summary[f'std_ndcg@{k}']:<12.3f}")
    print()
    
    print(f"{'MRR':<20} {summary['avg_mrr']:<12.3f} {summary['std_mrr']:<12.3f}")
    print(f"{'AP':<20} {summary['avg_ap']:<12.3f} {summary['std_ap']:<12.3f}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'summary': summary,
        'detailed': all_metrics,
        'config': {
            'k_values': k_values,
            'pipeline_k': pipeline.k,
            'chunk_size': pipeline.chunk_size,
            'chunk_overlap': pipeline.chunk_overlap
        }
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              EVALUATE RETRIEVAL                              ║
╚══════════════════════════════════════════════════════════════╝

Użycie:
    python evaluate_retrieval.py <pdf_path> <dataset_path>

Przykład:
    python evaluate_retrieval.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    dataset_path = sys.argv[2]
    
    # Wczytaj dataset
    print(f"\n📂 Ładowanie datasetu: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Obsłuż oba formaty (stary i nowy)
    if 'data' in data:
        dataset = data['data']
        config = data.get('metadata', {}).get('baseline_config', {})
    else:
        dataset = data
        config = {"chunk_size": 800, "chunk_overlap": 100}
    
    print(f"   ✓ Załadowano {len(dataset)} pytań")
    
    # Sprawdź czy mamy annotacje
    has_annotations = all('relevant_chunk_indices' in item for item in dataset)
    if not has_annotations:
        print("\n❌ ERROR: Dataset nie ma annotacji 'relevant_chunk_indices'!")
        print("   Najpierw uruchom: python prepare_dataset.py <pdf_path>")
        sys.exit(1)
    
    # Stwórz pipeline (BASELINE CONFIG!)
    print(f"\n🔧 Tworzenie pipeline...")
    print(f"   chunk_size={config.get('chunk_size', 800)}")
    print(f"   chunk_overlap={config.get('chunk_overlap', 100)}")
    
    pipeline = RAGPipeline(
        chunk_size=config.get('chunk_size', 800),
        chunk_overlap=config.get('chunk_overlap', 100),
        k=10  # Retrieval k
    )
    pipeline.process_document(pdf_path)
    print(f"   ✓ Utworzono {pipeline.num_chunks} chunków")
    
    # Ewaluacja
    results = evaluate_retrieval(pipeline, dataset, k_values=[1, 3, 5, 10])
    
    # Zapisz wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"retrieval_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Wyniki zapisane: {output_file}")