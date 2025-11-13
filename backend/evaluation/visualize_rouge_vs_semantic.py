"""
Wizualizacja porównująca ROUGE vs Semantic Similarity.
Pokazuje że ROUGE często karze dobre odpowiedzi.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import numpy as np

def create_comparison_scatter(results_file: str):
    """
    Tworzy scatter plot ROUGE vs Semantic Similarity.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Weź detailed results (z ground truth dataset)
    if 'ground_truth_dataset' in data:
        results = data['ground_truth_dataset']['detailed_results']
    else:
        # Jeśli to plik z advanced_experiments
        results = data['chunk_size_experiment'][0]['detailed_results']
    
    # Wyciągnij metryki
    rouge_scores = [r['rouge1_f1'] for r in results]
    semantic_scores = [r.get('semantic_similarity', 0) for r in results]
    
    # Wykres
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # WYKRES 1: Scatter ROUGE vs Semantic
    ax1.scatter(rouge_scores, semantic_scores, alpha=0.6, s=100, color='#667eea')
    
    # Linia y=x (idealne dopasowanie)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Idealne dopasowanie (ROUGE = Semantic)')
    
    ax1.set_xlabel('ROUGE-1 F1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Semantic Similarity', fontsize=12, fontweight='bold')
    ax1.set_title('ROUGE-1 vs Semantic Similarity', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Dodaj regiony
    ax1.axhspan(0.7, 1.0, alpha=0.1, color='green', label='High Semantic')
    ax1.axvspan(0, 0.5, alpha=0.1, color='red')
    
    # Annotate ciekawe punkty (wysokie semantic, niskie ROUGE)
    for i, (rouge, semantic) in enumerate(zip(rouge_scores, semantic_scores)):
        if semantic > 0.75 and rouge < 0.4:
            ax1.annotate(f'Q{i+1}', (rouge, semantic), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # WYKRES 2: Histogram różnicy
    differences = [semantic - rouge for semantic, rouge in zip(semantic_scores, rouge_scores)]
    
    ax2.hist(differences, bins=20, color='#667eea', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Brak różnicy')
    ax2.axvline(np.mean(differences), color='green', linestyle='--', linewidth=2, 
               label=f'Średnia różnica: {np.mean(differences):.3f}')
    
    ax2.set_xlabel('Semantic Similarity - ROUGE-1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Liczba pytań', fontsize=12, fontweight='bold')
    ax2.set_title('Rozkład różnic: Semantic > ROUGE oznacza że\nROUGE niedoszacowuje jakości', 
                 fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = results_file.replace('.json', '_rouge_vs_semantic.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Wykres zapisany: {output_file}")
    
    # Statystyki
    print(f"\n{'='*70}")
    print("📊 STATYSTYKI PORÓWNANIA")
    print(f"{'='*70}")
    print(f"Średnia ROUGE-1:           {np.mean(rouge_scores):.3f}")
    print(f"Średnia Semantic Sim:      {np.mean(semantic_scores):.3f}")
    print(f"Różnica (Semantic - ROUGE): {np.mean(differences):.3f}")
    print(f"\nLiczba pytań gdzie Semantic > ROUGE o >0.3: {sum(1 for d in differences if d > 0.3)}/{len(differences)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python evaluation/visualize_rouge_vs_semantic.py <results.json>")
        sys.exit(1)
    
    create_comparison_scatter(sys.argv[1])