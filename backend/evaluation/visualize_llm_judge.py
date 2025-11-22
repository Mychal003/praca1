# backend/evaluation/visualize_llm_judge.py
"""
Wizualizacja wyników LLM Judge vs tradycyjne metryki.
Obsługuje dane z run_experiments() (lista konfiguracji).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
from scipy.stats import pearsonr

def create_comparison_chart(results_file: str):
    """
    Tworzy wykres porównujący wszystkie metryki.
    Obsługuje zarówno dane z run_experiments (lista) jak i compare_metrics (dict).
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # =========================================================================
    # 🆕 DETECT FORMAT: Lista (run_experiments) vs Dict (compare_metrics)
    # =========================================================================
    
    if isinstance(data, list):
        # Format z run_experiments - wybierz najlepszą konfigurację
        print("📊 Wykryto format run_experiments (lista konfiguracji)")
        print(f"   Liczba konfiguracji: {len(data)}")
        
        # Sprawdź która konfiguracja ma LLM Judge scores
        configs_with_llm = [cfg for cfg in data 
                           if any('llm_judge_scores' in r 
                                 for r in cfg.get('detailed_results', []))]
        
        if not configs_with_llm:
            print("❌ ERROR: Brak wyników LLM Judge w danych!")
            print("   Uruchom ponownie z flagą --llm-judge")
            sys.exit(1)
        
        # Użyj najlepszej konfiguracji (według LLM Overall)
        best_config = max(configs_with_llm, 
                         key=lambda x: x['summary'].get('avg_llm_overall', 0))
        
        print(f"   ✅ Wybrano najlepszą konfigurację: {best_config['config']['name']}")
        print(f"      LLM Overall: {best_config['summary'].get('avg_llm_overall', 0):.3f}")
        
        results = best_config['detailed_results']
        summary = best_config['summary']
        
        # Oblicz korelacje
        rouge = [r['rouge1_f1'] for r in results if 'llm_judge_scores' in r]
        semantic = [r.get('semantic_similarity', 0) for r in results if 'llm_judge_scores' in r]
        llm_overall = [r['llm_judge_scores']['overall'] for r in results if 'llm_judge_scores' in r]
        
        if len(rouge) < 2:
            print("⚠️  Za mało danych do obliczenia korelacji")
            corr_rouge, corr_semantic = 0, 0
        else:
            corr_rouge, _ = pearsonr(rouge, llm_overall)
            corr_semantic, _ = pearsonr(semantic, llm_overall)
        
        correlations = {
            'rouge_vs_llm': corr_rouge,
            'semantic_vs_llm': corr_semantic
        }
        
    elif isinstance(data, dict) and 'results' in data:
        # Format z compare_metrics
        print("📊 Wykryto format compare_metrics (dict)")
        results = data['results']['detailed_results']
        correlations = data.get('correlations', {})
        
    else:
        print("❌ ERROR: Nieznany format danych!")
        print("   Oczekiwano: lista (run_experiments) lub dict z 'results' (compare_metrics)")
        sys.exit(1)
    
    # =========================================================================
    # Filtruj wyniki - tylko te z LLM Judge scores
    # =========================================================================
    
    results = [r for r in results if 'llm_judge_scores' in r]
    
    if not results:
        print("❌ ERROR: Brak wyników z LLM Judge scores!")
        sys.exit(1)
    
    print(f"\n📈 Tworzę wykresy dla {len(results)} pytań...\n")
    
    # Wyciągnij metryki
    rouge = [r['rouge1_f1'] for r in results]
    semantic = [r.get('semantic_similarity', 0) for r in results]
    llm_correctness = [r['llm_judge_scores']['correctness'] for r in results]
    llm_completeness = [r['llm_judge_scores']['completeness'] for r in results]
    llm_relevance = [r['llm_judge_scores']['relevance'] for r in results]
    llm_overall = [r['llm_judge_scores']['overall'] for r in results]
    
    # Wykres
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # =========================================================================
    # 1. ROUGE vs LLM Overall
    # =========================================================================
    ax1.scatter(rouge, llm_overall, alpha=0.6, s=100, color='#667eea')
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (perfect correlation)')
    ax1.set_xlabel('ROUGE-1 F1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LLM Judge (Overall)', fontsize=12, fontweight='bold')
    ax1.set_title(f'ROUGE vs LLM Judge\nCorrelation: {correlations["rouge_vs_llm"]:.3f}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # =========================================================================
    # 2. Semantic vs LLM Overall
    # =========================================================================
    ax2.scatter(semantic, llm_overall, alpha=0.6, s=100, color='#f093fb')
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (perfect correlation)')
    ax2.set_xlabel('Semantic Similarity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('LLM Judge (Overall)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Semantic vs LLM Judge\nCorrelation: {correlations["semantic_vs_llm"]:.3f}', 
                 fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Dodaj tekst o różnicy korelacji
    diff = correlations["semantic_vs_llm"] - correlations["rouge_vs_llm"]
    ax2.text(0.05, 0.95, f'Semantic advantage: +{diff:.3f}',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            verticalalignment='top')
    
    # =========================================================================
    # 3. LLM Judge dimensions (z Groundedness!)
    # =========================================================================
    dimensions = ['Correctness', 'Completeness', 'Relevance', 'Groundedness', 'Overall']
    
    # Sprawdź czy mamy groundedness
    has_groundedness = all(r['llm_judge_scores']['groundedness'] is not None for r in results)
    
    if has_groundedness:
        llm_groundedness = [r['llm_judge_scores']['groundedness'] for r in results]
        avg_scores = [
            np.mean(llm_correctness),
            np.mean(llm_completeness),
            np.mean(llm_relevance),
            np.mean(llm_groundedness),
            np.mean(llm_overall)
        ]
    else:
        dimensions = dimensions[:3] + [dimensions[4]]  # Skip groundedness
        avg_scores = [
            np.mean(llm_correctness),
            np.mean(llm_completeness),
            np.mean(llm_relevance),
            np.mean(llm_overall)
        ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#98D8C8', '#FFA07A'][:len(dimensions)]
    
    # Highlight groundedness jeśli = 1.0
    if has_groundedness and avg_scores[3] == 1.0:
        colors[3] = '#00FF00'  # Bright green
    
    bars = ax3.bar(dimensions, avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax3.set_title('LLM Judge: Average Scores by Dimension', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Score')
    
    for bar, score in zip(bars, avg_scores):
        y_pos = score + 0.02 if score < 0.95 else score - 0.05
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{score:.3f}', ha='center', fontweight='bold', fontsize=10)
    
    # Groundedness annotation jeśli perfect
    if has_groundedness and avg_scores[3] == 1.0:
        ax3.annotate('🏆 ZERO\nHALUCYNACJI!', 
                    xy=(3, 1.0), xytext=(3, 0.85),
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax3.legend(loc='lower left', fontsize=9)
    
    # =========================================================================
    # 4. Wszystkie metryki razem (box plot)
    # =========================================================================
    data_to_plot = [rouge, semantic, llm_correctness, llm_overall]
    labels = ['ROUGE-1\nF1', 'Semantic\nSimilarity', 'LLM\nCorrectness', 'LLM\nOverall']
    
    bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors_box = ['#667eea', '#f093fb', '#FF6B6B', '#98D8C8']
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
    
    # Kolor whiskers i medians
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(2)
        median.set_color('black')
    
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of All Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.05)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # =========================================================================
    # Zapisz
    # =========================================================================
    output_file = results_file.replace('.json', '_llm_judge_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Wykres zapisany: {output_file}")
    
    # =========================================================================
    # Statystyki do konsoli
    # =========================================================================
    print(f"\n{'='*70}")
    print("📊 STATYSTYKI")
    print(f"{'='*70}")
    print(f"Liczba pytań:          {len(results)}")
    print(f"\nŚrednie wyniki:")
    print(f"  ROUGE-1 F1:          {np.mean(rouge):.3f}")
    print(f"  Semantic Similarity: {np.mean(semantic):.3f}")
    print(f"  LLM Correctness:     {np.mean(llm_correctness):.3f}")
    print(f"  LLM Completeness:    {np.mean(llm_completeness):.3f}")
    print(f"  LLM Relevance:       {np.mean(llm_relevance):.3f}")
    if has_groundedness:
        print(f"  LLM Groundedness:    {np.mean(llm_groundedness):.3f} {'🏆' if np.mean(llm_groundedness) == 1.0 else ''}")
    print(f"  LLM Overall:         {np.mean(llm_overall):.3f}")
    print(f"\nKorelacje z LLM Overall:")
    print(f"  ROUGE-1:             {correlations['rouge_vs_llm']:+.3f}")
    print(f"  Semantic Similarity: {correlations['semantic_vs_llm']:+.3f}")
    print(f"  Różnica:             {diff:+.3f} (na korzyść Semantic)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Użycie:
    python evaluation/visualize_llm_judge.py <results.json>

Obsługuje:
    - evaluation_results_*.json (z run_experiments --llm-judge)
    - metrics_comparison.json (z compare_metrics.py)
        """)
        sys.exit(1)
    
    create_comparison_chart(sys.argv[1])