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
    
    # Przygotuj figurę z 6 subplotami (2x3)
    fig = plt.figure(figsize=(18, 12))
    
    # =========================================================================
    # WYKRES 1: Chunk Size vs ROUGE-1
    # =========================================================================
    ax1 = plt.subplot(2, 3, 1)
    plot_experiment(ax1, data['chunk_size_experiment'], 
                   'chunk_size', 'avg_rouge1_f1',
                   'Wpływ rozmiaru chunków na jakość',
                   'Chunk Size', 'ROUGE-1 F1')
    
    # =========================================================================
    # WYKRES 2: K vs ROUGE-1
    # =========================================================================
    ax2 = plt.subplot(2, 3, 2)
    plot_experiment(ax2, data['k_experiment'], 
                   'k', 'avg_rouge1_f1',
                   'Wpływ liczby retrievanych dokumentów',
                   'Liczba dokumentów (k)', 'ROUGE-1 F1')
    
    # =========================================================================
    # WYKRES 3: Overlap vs ROUGE-1
    # =========================================================================
    ax3 = plt.subplot(2, 3, 3)
    plot_experiment(ax3, data['overlap_experiment'], 
                   'chunk_overlap', 'avg_rouge1_f1',
                   'Wpływ overlap na jakość',
                   'Chunk Overlap', 'ROUGE-1 F1')
    
    # =========================================================================
    # WYKRES 4: Jakość vs Latencja (Chunk Size)
    # =========================================================================
    ax4 = plt.subplot(2, 3, 4)
    plot_tradeoff(ax4, data['chunk_size_experiment'],
                 'Chunk Size: Jakość vs Wydajność')
    
    # =========================================================================
    # WYKRES 5: Jakość vs Latencja (K)
    # =========================================================================
    ax5 = plt.subplot(2, 3, 5)
    plot_tradeoff(ax5, data['k_experiment'],
                 'K: Jakość vs Wydajność')
    
    # =========================================================================
    # WYKRES 6: Porównanie kategorii pytań
    # =========================================================================
    ax6 = plt.subplot(2, 3, 6)
    plot_category_comparison(ax6, data['chunk_size_experiment'])
    
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
    print("📋 TABELA LATEX - Eksperyment: Chunk Size")
    print(f"{'='*80}\n")
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Wpływ rozmiaru chunków na jakość odpowiedzi systemu RAG}")
    print(r"\label{tab:chunk_size}")
    print(r"\begin{tabular}{|l|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Konfiguracja} & \textbf{Chunk Size} & \textbf{ROUGE-1 F1} & \textbf{Token Overlap} & \textbf{Latencja (s)} \\")
    print(r"\hline")
    
    for result in data['chunk_size_experiment']:
        name = result['config']['name']
        chunk = result['config']['chunk_size']
        rouge = result['summary']['avg_rouge1_f1']
        overlap = result['summary']['avg_token_overlap']
        latency = result['summary']['avg_latency']
        
        print(f"{name} & {chunk} & {rouge:.3f} & {overlap:.3f} & {latency:.2f} \\\\")
    
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