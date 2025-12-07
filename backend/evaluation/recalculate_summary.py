"""
Skrypt do przeliczenia średnich w pliku JSON z wynikami ewaluacji.

Użycie:
    python recalculate_summary.py <plik_json>
    python recalculate_summary.py generation_results_1200_10_0_with_llm_judge_20251205_222026.json
"""

import json
import sys

def recalculate_summary(filename):
    """Przelicza wszystkie średnie w pliku JSON."""
    
    print(f"\n📂 Wczytuję: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['detailed']
    n = len(results)
    
    print(f"   ✓ Wczytano {n} wyników\n")
    
    # Przelicz średnie podstawowe
    data['summary']['avg_rouge1_f1'] = sum(r['metrics']['rouge1_f1'] for r in results) / n
    data['summary']['avg_semantic_similarity'] = sum(r['metrics']['semantic_similarity'] for r in results) / n
    data['summary']['avg_token_overlap'] = sum(r['metrics']['token_overlap'] for r in results) / n
    data['summary']['avg_latency'] = sum(r['metrics']['latency'] for r in results) / n
    
    # LLM Judge
    llm_results = [r for r in results if 'llm_judge' in r['metrics']]
    
    if llm_results:
        for key in ['correctness', 'completeness', 'relevance', 'groundedness', 'overall']:
            scores = [r['metrics']['llm_judge'][key] for r in llm_results 
                     if r['metrics']['llm_judge'].get(key) is not None]
            if scores:
                data['summary'][f'avg_llm_{key}'] = sum(scores) / len(scores)
    
    # Kategorie
    categories = set(r['category'] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        data['summary'][f'{cat}_rouge1_f1'] = sum(r['metrics']['rouge1_f1'] for r in cat_results) / len(cat_results)
        data['summary'][f'{cat}_semantic'] = sum(r['metrics']['semantic_similarity'] for r in cat_results) / len(cat_results)
    
    # Zapisz
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Wyświetl wyniki
    print("=" * 60)
    print("✅ SUMMARY PRZELICZONE!")
    print("=" * 60)
    print(f"\n{'Metryka':<25} {'Wartość':<12}")
    print("-" * 37)
    print(f"{'ROUGE-1 F1':<25} {data['summary']['avg_rouge1_f1']:.3f}")
    print(f"{'Semantic Similarity':<25} {data['summary']['avg_semantic_similarity']:.3f}")
    print(f"{'Token Overlap':<25} {data['summary']['avg_token_overlap']:.3f}")
    print(f"{'Latencja (s)':<25} {data['summary']['avg_latency']:.2f}")
    
    if llm_results:
        print()
        print(f"{'LLM Correctness':<25} {data['summary'].get('avg_llm_correctness', 'N/A'):.3f}")
        print(f"{'LLM Completeness':<25} {data['summary'].get('avg_llm_completeness', 'N/A'):.3f}")
        print(f"{'LLM Relevance':<25} {data['summary'].get('avg_llm_relevance', 'N/A'):.3f}")
        print(f"{'LLM Groundedness':<25} {data['summary'].get('avg_llm_groundedness', 'N/A'):.3f}")
        print(f"{'LLM Overall':<25} {data['summary'].get('avg_llm_overall', 'N/A'):.3f}")
    
    print()
    print("📊 Per kategoria:")
    print("-" * 37)
    for cat in sorted(categories):
        rouge = data['summary'].get(f'{cat}_rouge1_f1', 0)
        semantic = data['summary'].get(f'{cat}_semantic', 0)
        print(f"{cat.upper():<20} ROUGE: {rouge:.3f} | Semantic: {semantic:.3f}")
    
    print()
    print("=" * 60)
    print(f"💾 Zapisano: {filename}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Użycie:
    python recalculate_summary.py <plik_json>

Przykład:
    python recalculate_summary.py generation_results_1200_10_0_with_llm_judge_20251205_222026.json
        """)
        sys.exit(1)
    
    filename = sys.argv[1]
    recalculate_summary(filename)