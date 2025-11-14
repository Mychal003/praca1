"""
Pełna ewaluacja systemu RAG z retrieval metrics.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.rag_pipeline import RAGPipeline
from evaluation.evaluate_simple import evaluate_system

print("\n" + "="*70)
print("🚀 PEŁNA EWALUACJA RAG Z RETRIEVAL METRICS")
print("="*70)

# Wczytaj annotated dataset - FIX: Dodaj encoding='utf-8'
annotated_file = 'test_dataset_ground_truth_with_relevance.json'

print(f"\n📂 Loading annotated dataset: {annotated_file}")

try:
    with open(annotated_file, 'r', encoding='utf-8') as f:  # ← DODAJ encoding='utf-8'
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} questions with relevance annotations")
except FileNotFoundError:
    print(f"❌ ERROR: File not found: {annotated_file}")
    print(f"\n💡 First run annotation:")
    print(f"   python evaluation/annotate_relevant_chunks.py uploads/Archer_D7UN_V1_UG.pdf test_dataset_ground_truth.json")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading dataset: {e}")
    sys.exit(1)

# Sprawdź czy mamy relevant_chunk_indices
has_retrieval_data = any('relevant_chunk_indices' in item for item in dataset)
if not has_retrieval_data:
    print("⚠️  WARNING: Dataset doesn't have relevance annotations!")
    print("   Retrieval metrics will be skipped.")

# Stwórz pipeline
print("\n🔧 Initializing RAG pipeline...")
pdf_path = 'uploads/Archer_D7UN_V1_UG.pdf'

if not os.path.exists(pdf_path):
    print(f"❌ ERROR: PDF not found: {pdf_path}")
    sys.exit(1)

pipeline = RAGPipeline(chunk_size=800, chunk_overlap=100, k=7)
pipeline.process_document(pdf_path)

print("✅ Pipeline ready!")

# PEŁNA EWALUACJA
print("\n" + "="*70)
print("📊 ROZPOCZYNAM EWALUACJĘ...")
print("="*70)

results = evaluate_system(
    pipeline, 
    dataset, 
    evaluate_retrieval_metrics=has_retrieval_data
)

# Zapisz wyniki - FIX: Dodaj encoding='utf-8'
output_file = 'full_evaluation_results.json'
with open(output_file, 'w', encoding='utf-8') as f:  # ← DODAJ encoding='utf-8'
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*70)
print("✅ EWALUACJA ZAKOŃCZONA!")
print("="*70)
print(f"\n💾 Wyniki zapisane: {output_file}")

# Podsumowanie kluczowych metryk
print("\n" + "="*70)
print("🎯 KLUCZOWE METRYKI")
print("="*70)

summary = results['summary']

print("\n📈 GENERATION METRICS:")
print(f"  • ROUGE-1 F1:          {summary['avg_rouge1_f1']:.3f}")
print(f"  • Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
print(f"  • Latencja:            {summary['avg_latency']:.2f}s")

if has_retrieval_data:
    print("\n🔍 RETRIEVAL METRICS:")
    print(f"  • Precision@5:  {summary.get('avg_precision@5', 0):.3f}")
    print(f"  • Recall@5:     {summary.get('avg_recall@5', 0):.3f}")
    print(f"  • F1@5:         {summary.get('avg_f1@5', 0):.3f}")
    print(f"  • MRR:          {summary.get('avg_mrr', 0):.3f}")
    print(f"  • NDCG@5:       {summary.get('avg_ndcg@5', 0):.3f}")

print("\n" + "="*70 + "\n")