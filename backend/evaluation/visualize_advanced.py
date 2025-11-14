"""
Zaawansowane wizualizacje dla pracy inżynierskiej.
Tworzy kompleksowe wykresy z wszystkich eksperymentów.
"""

import matplotlib
matplotlib.use('Agg')  # FIX: Użyj nieinteraktywnego backendu
import matplotlib.pyplot as plt

import json
import numpy as np
import sys


def create_comprehensive_charts(results_file: str):
    """
    Tworzy kompleksowy zestaw wykresów z rozszerzonych eksperymentów.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Przygotuj figurę z 9 subplotami (3x3) - ZMIENIONE z 2x3 na 3x3
    fig = plt.figure(figsize=(20, 15))  # ZMIENIONE: większy rozmiar
    
    # =========================================================================
    # WYKRES 1: Chunk Size vs ROUGE-1
    # =========================================================================
    ax1 = plt.subplot(3, 3, 1)  # ZMIENIONE: 3x3
    plot_experiment(ax1, data['chunk_size_experiment'], 
                   'chunk_size', 'avg_rouge1_f1',
                   'Chunk Size vs ROUGE-1',
                   'Chunk Size', 'ROUGE-1 F1')
    
    # =========================================================================
    # WYKRES 2: Chunk Size vs Semantic Similarity (NOWY!)
    # =========================================================================
    ax2 = plt.subplot(3, 3, 2)  # NOWY
    plot_experiment(ax2, data['chunk_size_experiment'], 
                   'chunk_size', 'avg_semantic_similarity',
                   'Chunk Size vs Semantic Similarity',
                   'Chunk Size', 'Semantic Similarity')
    
    # =========================================================================
    # WYKRES 3: K vs ROUGE-1
    # =========================================================================
    ax3 = plt.subplot(3, 3, 3)  # ZMIENIONE: pozycja
    plot_experiment(ax3, data['k_experiment'], 
                   'k', 'avg_rouge1_f1',
                   'K vs ROUGE-1',
                   'Liczba dokumentów (k)', 'ROUGE-1 F1')
    
    # =========================================================================
    # WYKRES 4: K vs Semantic Similarity (NOWY!)
    # =========================================================================
    ax4 = plt.subplot(3, 3, 4)  # NOWY
    plot_experiment(ax4, data['k_experiment'], 
                   'k', 'avg_semantic_similarity',
                   'K vs Semantic Similarity',
                   'Liczba dokumentów (k)', 'Semantic Similarity')
    
    # =========================================================================
    # WYKRES 5: Overlap vs ROUGE-1
    # =========================================================================
    ax5 = plt.subplot(3, 3, 5)  # ZMIENIONE: pozycja
    plot_experiment(ax5, data['overlap_experiment'], 
                   'chunk_overlap', 'avg_rouge1_f1',
                   'Overlap vs ROUGE-1',
                   'Chunk Overlap', 'ROUGE-1 F1')
    
    # =========================================================================
    # WYKRES 6: Overlap vs Semantic Similarity (NOWY!)
    # =========================================================================
    ax6 = plt.subplot(3, 3, 6)  # NOWY
    plot_experiment(ax6, data['overlap_experiment'], 
                   'chunk_overlap', 'avg_semantic_similarity',
                   'Overlap vs Semantic Similarity',
                   'Chunk Overlap', 'Semantic Similarity')
    
    # =========================================================================
    # WYKRES 7: ROUGE vs Semantic (scatter) (NOWY!)
    # =========================================================================
    ax7 = plt.subplot(3, 3, 7)  # NOWY
    plot_rouge_vs_semantic(ax7, data['chunk_size_experiment'][0])  # Użyj najlepszej config
    
    # =========================================================================
    # WYKRES 8: Trade-off Chunk Size
    # =========================================================================
    ax8 = plt.subplot(3, 3, 8)  # ZMIENIONE: pozycja
    plot_tradeoff(ax8, data['chunk_size_experiment'],
                 'Chunk Size: Jakość vs Wydajność')
    
    # =========================================================================
    # WYKRES 9: Trade-off K
    # =========================================================================
    ax9 = plt.subplot(3, 3, 9)  # ZMIENIONE: pozycja
    plot_tradeoff(ax9, data['k_experiment'],
                 'K: Jakość vs Wydajność')
    
    plt.tight_layout()
    
    # Zapisz
    output_file = results_file.replace('.json', '_comprehensive_charts.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Wykresy zapisane: {output_file}")
    
    # plt.show()  # USUNIĘTE - nie potrzebne w trybie Agg


def plot_experiment(ax, results, param_key, metric_key, title, xlabel, ylabel):
    """
    Rysuje wykres liniowy dla jednego eksperymentu.
    """
    params = [r['config'][param_key] for r in results]
    values = [r['summary'][metric_key] for r in results]
    
    ax.plot(params, values, marker='o', linewidth=2, markersize=8, color='#667eea')
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Dodaj wartości na punktach
    for p, v in zip(params, values):
        ax.annotate(f'{v:.3f}', (p, v), textcoords="offset points", 
                   xytext=(0,8), ha='center', fontsize=9)

def plot_retrieval_metrics(ax, results):
    """
    Wykres słupkowy dla retrieval metrics.
    """
    metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5']
    values = [results['summary'][f'avg_{m}'] for m in metrics]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(range(len(metrics)), values, color=colors, alpha=0.7)
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], rotation=45)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Retrieval Performance Metrics', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Dodaj wartości
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
               f'{value:.3f}', ha='center', fontsize=9)
        
def plot_tradeoff(ax, results, title):
    """
    Rysuje wykres scatter dla trade-off jakość vs latencja.
    """
    latencies = [r['summary']['avg_latency'] for r in results]
    rouge_scores = [r['summary']['avg_rouge1_f1'] for r in results]
    names = [r['config']['name'] for r in results]
    
    scatter = ax.scatter(latencies, rouge_scores, s=200, alpha=0.6, c=range(len(results)), 
                        cmap='viridis')
    
    # Dodaj etykiety
    for lat, rouge, name in zip(latencies, rouge_scores, names):
        ax.annotate(name, (lat, rouge), textcoords="offset points", 
                   xytext=(5,5), ha='left', fontsize=8)
    
    ax.set_xlabel('Latencja (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ROUGE-1 F1', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

def plot_rouge_vs_semantic(ax, result):
    """
    Scatter plot ROUGE vs Semantic Similarity dla jednej konfiguracji.
    """
    rouge_scores = [r['rouge1_f1'] for r in result['detailed_results']]
    semantic_scores = [r.get('semantic_similarity', 0) for r in result['detailed_results']]
    
    # Scatter plot
    ax.scatter(rouge_scores, semantic_scores, alpha=0.6, s=80, color='#667eea')
    
    # Linia y=x
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
    
    # Średnie
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_semantic = sum(semantic_scores) / len(semantic_scores)
    
    # Punkt średniej
    ax.scatter([avg_rouge], [avg_semantic], s=200, color='red', 
              marker='*', edgecolors='black', linewidth=2,
              label=f'Średnia (R:{avg_rouge:.2f}, S:{avg_semantic:.2f})')
    
    ax.set_xlabel('ROUGE-1 F1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Semantic Similarity', fontsize=11, fontweight='bold')
    ax.set_title('ROUGE-1 vs Semantic Similarity', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

def plot_category_comparison(ax, results):
    """
    Porównuje wyniki dla różnych kategorii pytań (factual, procedural, troubleshooting).
    """
    # Użyj najlepszej konfiguracji
    best_result = max(results, key=lambda x: x['summary']['avg_rouge1_f1'])
    
    # Grupuj wyniki po kategorii
    categories = {'factual': [], 'procedural': [], 'troubleshooting': []}
    
    for detail in best_result['detailed_results']:
        # Najpierw sprawdź w TEST_DATASET jaką kategorię ma to pytanie
        # (zakładamy że pytania są w tej samej kolejności)
        pass  # Tutaj można dodać kategoryzację jeśli dostępna
    
    # Tymczasowo: pokaż ogólne stats
    ax.text(0.5, 0.5, 'Category Breakdown\n(Requires category tagging)', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Analiza według kategorii pytań', fontsize=12, fontweight='bold')
    ax.axis('off')


def create_latex_table(results_file: str):
    """
    Generuje tabelę w formacie LaTeX gotową do wklejenia w pracę.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print("📋 TABELA LATEX - Eksperyment: Chunk Size (Z SEMANTIC SIMILARITY)")
    print(f"{'='*80}\n")
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Wpływ rozmiaru chunków na jakość odpowiedzi systemu RAG}")
    print(r"\label{tab:chunk_size}")
    print(r"\begin{tabular}{|l|c|c|c|c|c|}")  # ZMIENIONE: dodano kolumnę
    print(r"\hline")
    print(r"\textbf{Konfiguracja} & \textbf{Size} & \textbf{ROUGE-1} & \textbf{Semantic} & \textbf{Overlap} & \textbf{Latencja} \\")
    print(r"\hline")
    
    for result in data['chunk_size_experiment']:
        name = result['config']['name']
        chunk = result['config']['chunk_size']
        rouge = result['summary']['avg_rouge1_f1']
        semantic = result['summary'].get('avg_semantic_similarity', 0)  # NOWE!
        overlap = result['summary']['avg_token_overlap']
        latency = result['summary']['avg_latency']
        
        print(f"{name} & {chunk} & {rouge:.3f} & {semantic:.3f} & {overlap:.3f} & {latency:.2f} \\\\")  # ZMIENIONE
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    
    print(f"\n{'='*80}")
    print("📋 TABELA LATEX - Eksperyment: Liczba dokumentów (k)")
    print(f"{'='*80}\n")
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Wpływ liczby retrievanych dokumentów na jakość odpowiedzi}")
    print(r"\label{tab:k_value}")
    print(r"\begin{tabular}{|l|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Konfiguracja} & \textbf{k} & \textbf{ROUGE-1 F1} & \textbf{Token Overlap} & \textbf{Latencja (s)} \\")
    print(r"\hline")
    
    for result in data['k_experiment']:
        name = result['config']['name']
        k = result['config']['k']
        rouge = result['summary']['avg_rouge1_f1']
        overlap = result['summary']['avg_token_overlap']
        latency = result['summary']['avg_latency']
        
        print(f"{name} & {k} & {rouge:.3f} & {overlap:.3f} & {latency:.2f} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def create_summary_stats(results_file: str):
    """
    Tworzy podsumowanie statystyk do pracy.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print("📊 STATYSTYKI PODSUMOWUJĄCE - DO PRACY")
    print(f"{'='*80}\n")
    
    all_rouge = []
    all_latency = []
    
    for exp_name, exp_results in data.items():
        if exp_name == 'metadata':
            continue
        
        for result in exp_results:
            all_rouge.append(result['summary']['avg_rouge1_f1'])
            all_latency.append(result['summary']['avg_latency'])
    
    print(f"Całkowita liczba konfiguracji przetestowanych: {len(all_rouge)}")
    print(f"\nROUGE-1 F1:")
    print(f"  • Minimum: {min(all_rouge):.3f}")
    print(f"  • Maksimum: {max(all_rouge):.3f}")
    print(f"  • Średnia: {np.mean(all_rouge):.3f}")
    print(f"  • Mediana: {np.median(all_rouge):.3f}")
    print(f"  • Odchylenie std: {np.std(all_rouge):.3f}")
    
    print(f"\nLatencja (sekundy):")
    print(f"  • Minimum: {min(all_latency):.2f}s")
    print(f"  • Maksimum: {max(all_latency):.2f}s")
    print(f"  • Średnia: {np.mean(all_latency):.2f}s")
    print(f"  • Mediana: {np.median(all_latency):.2f}s")
    
    # Najlepsza konfiguracja ogólnie
    best_idx = all_rouge.index(max(all_rouge))
    print(f"\n🏆 Najlepsza konfiguracja ogólnie:")
    print(f"  • ROUGE-1 F1: {max(all_rouge):.3f}")
    print(f"  • Latencja: {all_latency[best_idx]:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""


Użycie:
   python evaluation/visualize_advanced.py <advanced_results.json>

Generuje:
   • Kompleksowe wykresy (6 subplotów)
   • Tabele LaTeX gotowe do wklejenia
   • Statystyki podsumowujące

Przykład:
   python evaluation/visualize_advanced.py advanced_results_20250129_150000.json

═══════════════════════════════════════════════════════════════
        """)
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print("📊 Tworzę wykresy...")
    create_comprehensive_charts(results_file)
    
    print("\n📋 Generuję tabele LaTeX...")
    create_latex_table(results_file)
    
    print("\n📈 Obliczam statystyki...")
    create_summary_stats(results_file)
    
    print("\n✅ Wszystko gotowe!")