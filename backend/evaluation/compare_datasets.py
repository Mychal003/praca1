"""
Porównanie wyników z manualnym dataset vs ground truth dataset.
Pokazuje czy problem był w oczekiwanych odpowiedziach.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.rag_pipeline import RAGPipeline
from evaluation.evaluate_simple import evaluate_system

# Import obu datasetów

# Wczytaj ground truth
with open('test_dataset_ground_truth.json', 'r', encoding='utf-8') as f:
    GROUND_TRUTH_DATASET = json.load(f)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║   PORÓWNANIE: Manual Dataset vs Ground Truth Dataset        ║
╚══════════════════════════════════════════════════════════════╝

📊 Porównujemy dwa datasety na TYM SAMYM systemie RAG:

1. MANUAL DATASET:      Twoje ręcznie napisane expected answers
2. GROUND TRUTH DATASET: Odpowiedzi wyekstrahowane z dokumentu przez GPT-4

Jeśli ground truth da WYŻSZE wyniki → problem był w twoich
oczekiwanych odpowiedziach, nie w systemie RAG!
""")

def compare_datasets(pdf_path: str):
    """
    Porównuje wyniki z dwoma datasetami.
    """
    
    print("\n" + "="*70)
    print("🔧 Inicjalizacja systemu RAG...")
    print("="*70)
    
    # Stwórz pipeline (najlepsza config z twoich eksperymentów)
    pipeline = RAGPipeline(
        chunk_size=800,
        chunk_overlap=100, 
        k=5
    )
    
    pipeline.process_document(pdf_path)
    
    # =========================================================================
    # TEST 1: Manual Dataset
    # =========================================================================
    print("\n" + "="*70)
    print("📝 TEST 1: MANUAL DATASET (Twoje ręczne odpowiedzi)")
    print("="*70)
    
    manual_results = evaluate_system(pipeline, MANUAL_DATASET)
    
    # =========================================================================
    # TEST 2: Ground Truth Dataset
    # =========================================================================
    print("\n" + "="*70)
    print("🤖 TEST 2: GROUND TRUTH DATASET (Wyekstrahowane z dokumentu)")
    print("="*70)
    
    ground_truth_results = evaluate_system(pipeline, GROUND_TRUTH_DATASET)
    
    # =========================================================================
    # PORÓWNANIE
    # =========================================================================
    print("\n" + "="*70)
    print("📊 PORÓWNANIE WYNIKÓW")
    print("="*70)
    
    manual_rouge = manual_results['summary']['avg_rouge1_f1']
    gt_rouge = ground_truth_results['summary']['avg_rouge1_f1']
    
    manual_overlap = manual_results['summary']['avg_token_overlap']
    gt_overlap = ground_truth_results['summary']['avg_token_overlap']
    
    manual_latency = manual_results['summary']['avg_latency']
    gt_latency = ground_truth_results['summary']['avg_latency']
    
    improvement_rouge = ((gt_rouge - manual_rouge) / manual_rouge) * 100
    improvement_overlap = ((gt_overlap - manual_overlap) / manual_overlap) * 100
    
    print(f"\n{'Metryka':<25} {'Manual':<15} {'Ground Truth':<15} {'Zmiana'}")
    print("-" * 70)
    print(f"{'ROUGE-1 F1':<25} {manual_rouge:<15.3f} {gt_rouge:<15.3f} {improvement_rouge:+.1f}%")
    print(f"{'Token Overlap':<25} {manual_overlap:<15.3f} {gt_overlap:<15.3f} {improvement_overlap:+.1f}%")
    print(f"{'Latencja (s)':<25} {manual_latency:<15.2f} {gt_latency:<15.2f} {gt_latency - manual_latency:+.2f}s")
    
    print("\n" + "="*70)
    print("🎯 WNIOSKI")
    print("="*70)
    
    if improvement_rouge > 20:
        print(f"""
✅ ZNACZĄCA POPRAWA (+{improvement_rouge:.1f}%)!

Problem był głównie w OCZEKIWANYCH ODPOWIEDZIACH, nie w systemie RAG!

Interpretacja:
- Twoje ręczne odpowiedzi używały innych słów niż dokument
- System RAG faktycznie generował poprawne odpowiedzi
- ROUGE karał za różnice leksykalne mimo poprawności merytorycznej

Rekomendacja dla pracy inżynierskiej:
→ Użyj GROUND TRUTH DATASET jako baseline
→ Pokaż tę różnicę w sekcji "Metodologia Ewaluacji"
→ Wyjaśnij, że metryki leksykalne (ROUGE) mają ograniczenia
        """)
    
    elif improvement_rouge > 10:
        print(f"""
⚠️  UMIARKOWANA POPRAWA (+{improvement_rouge:.1f}%)

Część problemu była w expected answers, ale system też ma rezerwy.

Rekomendacja:
→ Użyj ground truth dataset
→ Kontynuuj z implementacją hybrid retrieval
        """)
    
    elif improvement_rouge > 0:
        print(f"""
➡️  NIEWIELKA POPRAWA (+{improvement_rouge:.1f}%)

Ground truth pomógł trochę, ale główny problem jest w systemie RAG.

Rekomendacja:
→ PRIORYTET: Implementuj hybrid retrieval + reranking
→ Ground truth będzie pomocny do dokładnej ewaluacji poprawek
        """)
    
    else:
        print(f"""
🤔 BRAK POPRAWY lub POGORSZENIE ({improvement_rouge:+.1f}%)

To dziwne - sprawdź:
1. Czy ground truth dataset został poprawnie wygenerowany?
2. Czy zawiera sensowne odpowiedzi? (otwórz plik JSON i sprawdź)
        """)
    
    # =========================================================================
    # ANALIZA SZCZEGÓŁOWA - Najwięksi zwycięzcy
    # =========================================================================
    print("\n" + "="*70)
    print("🏆 TOP 5 PYTAŃ Z NAJWIĘKSZĄ POPRAWĄ")
    print("="*70)
    
    improvements = []
    for manual, gt in zip(manual_results['detailed_results'], 
                          ground_truth_results['detailed_results']):
        improvement = gt['rouge1_f1'] - manual['rouge1_f1']
        improvements.append({
            'question': manual['question'],
            'manual_rouge': manual['rouge1_f1'],
            'gt_rouge': gt['rouge1_f1'],
            'improvement': improvement
        })
    
    # Sortuj po improvement
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    for i, item in enumerate(improvements[:5], 1):
        print(f"\n{i}. {item['question'][:60]}...")
        print(f"   Manual ROUGE:  {item['manual_rouge']:.3f}")
        print(f"   GT ROUGE:      {item['gt_rouge']:.3f}")
        print(f"   Improvement:   +{item['improvement']:.3f} ({item['improvement']/item['manual_rouge']*100:+.1f}%)")
    
    # =========================================================================
    # ZAPISZ WYNIKI
    # =========================================================================
    comparison_results = {
        "manual_dataset": manual_results,
        "ground_truth_dataset": ground_truth_results,
        "comparison": {
            "manual_rouge": manual_rouge,
            "gt_rouge": gt_rouge,
            "improvement_percent": improvement_rouge,
            "manual_overlap": manual_overlap,
            "gt_overlap": gt_overlap
        },
        "top_improvements": improvements[:10]
    }
    
    output_file = "comparison_manual_vs_groundtruth.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Wyniki zapisane: {output_file}")
    print("="*70 + "\n")
    
    return comparison_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Użycie:
    python evaluation/compare_datasets.py <pdf_path>

Przykład:
    python evaluation/compare_datasets.py uploads/Archer_D7UN_V1_UG.pdf
        """)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    compare_datasets(pdf_path)