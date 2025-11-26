"""
Wizualizacja wyników eksperymentów RAG dla pracy inżynierskiej.
Generuje profesjonalne wykresy pokazujące wpływ parametrów na jakość.

Użycie:
    python visualize_experiments.py experiments_results_20251126_005139.json

Generuje:
    1. param_impact_grid.png      - Grid 2x3: wpływ każdego parametru
    2. chunk_size_analysis.png    - Szczegółowa analiza chunk_size
    3. k_analysis.png             - Szczegółowa analiza k
    4. overlap_analysis.png       - Szczegółowa analiza overlap
    5. quality_vs_latency.png     - Trade-off jakość vs czas
    6. best_configs_comparison.png - Porównanie najlepszych konfiguracji
    7. summary_table.txt          - Tabele do pracy (tekst + LaTeX)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from pathlib import Path

# ============================================================================
# KONFIGURACJA STYLU (profesjonalny wygląd)
# ============================================================================

# Kolory
COLOR_ROUGE = '#2E86AB'      # Niebieski
COLOR_SEMANTIC = '#A23B72'   # Różowy/fioletowy
COLOR_LATENCY = '#F18F01'    # Pomarańczowy
COLOR_CHUNKS = '#C73E1D'     # Czerwony

# Styl wykresów
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def load_results(filepath: str) -> dict:
    """Wczytuje wyniki eksperymentów."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_param_values(results: list, param_key: str) -> list:
    """Ekstrahuje wartości parametru z wyników."""
    return [r['config'][param_key] for r in results]


def extract_metrics(results: list) -> dict:
    """Ekstrahuje metryki z wyników."""
    return {
        'rouge': [r['avg_rouge1_f1'] for r in results],
        'semantic': [r['avg_semantic_similarity'] for r in results],
        'latency': [r['avg_latency'] for r in results],
        'chunks': [r['num_chunks'] for r in results],
        'names': [r['config']['name'] for r in results]
    }


# ============================================================================
# WYKRES 1: GRID 2x3 - WPŁYW PARAMETRÓW
# ============================================================================

def plot_param_impact_grid(data: dict, output_path: str):
    """
    Główny wykres: grid 2x3 pokazujący wpływ każdego parametru.
    Górny rząd: wpływ na ROUGE i Semantic
    Dolny rząd: wpływ na latencję i liczbę chunków
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Wpływ parametrów na jakość systemu RAG', fontsize=16, fontweight='bold', y=1.02)
    
    experiments = [
        ('chunk_size', 'chunk_size', 'Rozmiar chunka (znaki)', [300, 500, 800, 1200, 1500]),
        ('k_values', 'k', 'Liczba dokumentów (k)', [1, 3, 5, 7, 10]),
        ('overlap', 'chunk_overlap', 'Overlap (znaki)', [0, 50, 100, 200, 300])
    ]
    
    for col, (exp_name, param_key, xlabel, expected_values) in enumerate(experiments):
        results = data[exp_name]['results']
        params = extract_param_values(results, param_key)
        metrics = extract_metrics(results)
        
        # Górny rząd: ROUGE i Semantic
        ax_top = axes[0, col]
        
        line1, = ax_top.plot(params, metrics['rouge'], 
                            marker='o', linewidth=2.5, markersize=8,
                            color=COLOR_ROUGE, label='ROUGE-1 F1')
        line2, = ax_top.plot(params, metrics['semantic'], 
                            marker='s', linewidth=2.5, markersize=8,
                            color=COLOR_SEMANTIC, label='Semantic Similarity')
        
        # Zaznacz najlepszy punkt
        best_rouge_idx = np.argmax(metrics['rouge'])
        ax_top.scatter([params[best_rouge_idx]], [metrics['rouge'][best_rouge_idx]], 
                      s=200, facecolors='none', edgecolors=COLOR_ROUGE, linewidths=3, zorder=5)
        
        best_semantic_idx = np.argmax(metrics['semantic'])
        ax_top.scatter([params[best_semantic_idx]], [metrics['semantic'][best_semantic_idx]], 
                      s=200, facecolors='none', edgecolors=COLOR_SEMANTIC, linewidths=3, zorder=5)
        
        ax_top.set_xlabel(xlabel, fontweight='bold')
        ax_top.set_ylabel('Wynik (0-1)', fontweight='bold')
        ax_top.set_title(f'Wpływ: {exp_name.replace("_", " ").title()}', fontweight='bold')
        ax_top.legend(loc='best')
        ax_top.set_ylim(0, 1.0)
        ax_top.set_xticks(params)
        
        # Wartości na punktach (tylko dla ROUGE)
        for i, (p, r) in enumerate(zip(params, metrics['rouge'])):
            ax_top.annotate(f'{r:.3f}', (p, r), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=8, color=COLOR_ROUGE)
        
        # Dolny rząd: Latencja i Chunks (dla chunk_size) lub tylko Latencja
        ax_bot = axes[1, col]
        
        line3, = ax_bot.plot(params, metrics['latency'], 
                            marker='^', linewidth=2.5, markersize=8,
                            color=COLOR_LATENCY, label='Latencja (s)')
        ax_bot.set_ylabel('Latencja (s)', color=COLOR_LATENCY, fontweight='bold')
        ax_bot.tick_params(axis='y', labelcolor=COLOR_LATENCY)
        
        # Dla chunk_size dodaj drugą oś z liczbą chunków
        if exp_name == 'chunk_size':
            ax_bot2 = ax_bot.twinx()
            line4, = ax_bot2.plot(params, metrics['chunks'], 
                                 marker='D', linewidth=2.5, markersize=8,
                                 color=COLOR_CHUNKS, linestyle='--', label='Liczba chunków')
            ax_bot2.set_ylabel('Liczba chunków', color=COLOR_CHUNKS, fontweight='bold')
            ax_bot2.tick_params(axis='y', labelcolor=COLOR_CHUNKS)
            
            # Połączona legenda
            lines = [line3, line4]
            labels = [l.get_label() for l in lines]
            ax_bot.legend(lines, labels, loc='best')
        else:
            ax_bot.legend(loc='best')
        
        ax_bot.set_xlabel(xlabel, fontweight='bold')
        ax_bot.set_title(f'Koszty: {exp_name.replace("_", " ").title()}', fontweight='bold')
        ax_bot.set_xticks(params)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Zapisano: {output_path}")
    plt.close()


# ============================================================================
# WYKRESY 2-4: SZCZEGÓŁOWA ANALIZA KAŻDEGO PARAMETRU
# ============================================================================

def plot_single_param_analysis(data: dict, exp_name: str, param_key: str, 
                               xlabel: str, title: str, output_path: str):
    """
    Szczegółowy wykres dla pojedynczego parametru.
    Zawiera: metryki jakości, latencję, adnotacje.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    results = data[exp_name]['results']
    params = extract_param_values(results, param_key)
    metrics = extract_metrics(results)
    
    # Lewy wykres: ROUGE i Semantic
    ax1 = axes[0]
    
    ax1.plot(params, metrics['rouge'], 
             marker='o', linewidth=2.5, markersize=10,
             color=COLOR_ROUGE, label='ROUGE-1 F1')
    ax1.plot(params, metrics['semantic'], 
             marker='s', linewidth=2.5, markersize=10,
             color=COLOR_SEMANTIC, label='Semantic Similarity')
    
    # Wypełnienie obszaru między metrykami
    ax1.fill_between(params, metrics['rouge'], metrics['semantic'], 
                     alpha=0.1, color='gray')
    
    # Adnotacje z wartościami
    for i, p in enumerate(params):
        ax1.annotate(f'{metrics["rouge"][i]:.3f}', (p, metrics['rouge'][i]), 
                    textcoords="offset points", xytext=(0, 12), ha='center', 
                    fontsize=9, fontweight='bold', color=COLOR_ROUGE)
        ax1.annotate(f'{metrics["semantic"][i]:.3f}', (p, metrics['semantic'][i]), 
                    textcoords="offset points", xytext=(0, -18), ha='center', 
                    fontsize=9, fontweight='bold', color=COLOR_SEMANTIC)
    
    # Zaznacz najlepsze wartości
    best_rouge_idx = np.argmax(metrics['rouge'])
    best_semantic_idx = np.argmax(metrics['semantic'])
    
    ax1.axvline(x=params[best_rouge_idx], color=COLOR_ROUGE, linestyle=':', alpha=0.5)
    ax1.axvline(x=params[best_semantic_idx], color=COLOR_SEMANTIC, linestyle=':', alpha=0.5)
    
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wynik (0-1)', fontsize=12, fontweight='bold')
    ax1.set_title('Metryki jakości', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(params)
    ax1.grid(True, alpha=0.3)
    
    # Prawy wykres: Trade-off jakość vs latencja
    ax2 = axes[1]
    
    # Scatter z kolorami według parametru
    scatter = ax2.scatter(metrics['latency'], metrics['rouge'], 
                         c=params, cmap='viridis', s=200, alpha=0.8, 
                         edgecolors='black', linewidths=1)
    
    # Etykiety punktów
    for i, name in enumerate(metrics['names']):
        ax2.annotate(name, (metrics['latency'][i], metrics['rouge'][i]),
                    textcoords="offset points", xytext=(8, 0), ha='left',
                    fontsize=9, fontweight='bold')
    
    # Linia trendu
    z = np.polyfit(metrics['latency'], metrics['rouge'], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(min(metrics['latency']), max(metrics['latency']), 100)
    ax2.plot(x_line, p_fit(x_line), '--', color='gray', alpha=0.5, label='Trend')
    
    ax2.set_xlabel('Latencja (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROUGE-1 F1', fontsize=12, fontweight='bold')
    ax2.set_title('Trade-off: Jakość vs Czas', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label(xlabel, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Zapisano: {output_path}")
    plt.close()


# ============================================================================
# WYKRES 5: QUALITY VS LATENCY (WSZYSTKIE KONFIGURACJE)
# ============================================================================

def plot_quality_vs_latency(data: dict, output_path: str):
    """
    Scatter plot: jakość vs latencja dla wszystkich konfiguracji.
    Kolory według typu eksperymentu.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'chunk_size': '#2E86AB', 'k_values': '#A23B72', 'overlap': '#F18F01'}
    markers = {'chunk_size': 'o', 'k_values': 's', 'overlap': '^'}
    labels = {'chunk_size': 'Rozmiar chunka', 'k_values': 'Liczba dokumentów (k)', 'overlap': 'Overlap'}
    
    all_latency = []
    all_rouge = []
    all_semantic = []
    
    for exp_name in ['chunk_size', 'k_values', 'overlap']:
        results = data[exp_name]['results']
        metrics = extract_metrics(results)
        
        all_latency.extend(metrics['latency'])
        all_rouge.extend(metrics['rouge'])
        all_semantic.extend(metrics['semantic'])
        
        ax.scatter(metrics['latency'], metrics['rouge'], 
                  c=colors[exp_name], marker=markers[exp_name], 
                  s=150, alpha=0.8, label=f'{labels[exp_name]} (ROUGE)',
                  edgecolors='black', linewidths=0.5)
        
        # Etykiety
        for i, name in enumerate(metrics['names']):
            ax.annotate(name, (metrics['latency'][i], metrics['rouge'][i]),
                       textcoords="offset points", xytext=(5, 5), ha='left',
                       fontsize=8, alpha=0.8)
    
    # Zaznacz optymalny punkt (najwyższy ROUGE)
    best_idx = np.argmax(all_rouge)
    ax.scatter([all_latency[best_idx]], [all_rouge[best_idx]], 
              s=400, facecolors='none', edgecolors='green', linewidths=3, 
              zorder=10, label=f'Najlepszy (ROUGE={all_rouge[best_idx]:.3f})')
    
    # Średnia jako punkt referencyjny
    ax.axhline(y=np.mean(all_rouge), color='gray', linestyle='--', alpha=0.5, 
               label=f'Średnia ROUGE ({np.mean(all_rouge):.3f})')
    ax.axvline(x=np.mean(all_latency), color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Latencja (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE-1 F1', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off: Jakość vs Czas odpowiedzi\n(wszystkie konfiguracje)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Adnotacja z wnioskiem
    ax.text(0.02, 0.98, 
            'Optymalny obszar:\nwysoki ROUGE + niska latencja\n(lewy górny róg)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Zapisano: {output_path}")
    plt.close()


# ============================================================================
# WYKRES 6: PORÓWNANIE NAJLEPSZYCH KONFIGURACJI
# ============================================================================

def plot_best_configs_comparison(data: dict, output_path: str):
    """
    Porównanie najlepszych konfiguracji z każdego eksperymentu.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Zbierz najlepsze konfiguracje
    best_configs = []
    for exp_name in ['chunk_size', 'k_values', 'overlap']:
        best_name = data[exp_name]['best']
        for r in data[exp_name]['results']:
            if r['config']['name'] == best_name:
                best_configs.append({
                    'experiment': exp_name,
                    'name': best_name,
                    'rouge': r['avg_rouge1_f1'],
                    'semantic': r['avg_semantic_similarity'],
                    'latency': r['avg_latency'],
                    'chunks': r['num_chunks']
                })
                break
    
    # Dodaj baseline (Medium 800, k=5, overlap=100)
    for r in data['chunk_size']['results']:
        if r['config']['chunk_size'] == 800:
            best_configs.append({
                'experiment': 'baseline',
                'name': 'Baseline (800/5/100)',
                'rouge': r['avg_rouge1_f1'],
                'semantic': r['avg_semantic_similarity'],
                'latency': r['avg_latency'],
                'chunks': r['num_chunks']
            })
            break
    
    names = [c['name'] for c in best_configs]
    x = np.arange(len(names))
    width = 0.35
    
    # Lewy wykres: ROUGE i Semantic
    ax1 = axes[0]
    
    rouge_vals = [c['rouge'] for c in best_configs]
    semantic_vals = [c['semantic'] for c in best_configs]
    
    bars1 = ax1.bar(x - width/2, rouge_vals, width, label='ROUGE-1 F1', 
                   color=COLOR_ROUGE, alpha=0.8)
    bars2 = ax1.bar(x + width/2, semantic_vals, width, label='Semantic Similarity', 
                   color=COLOR_SEMANTIC, alpha=0.8)
    
    # Wartości na słupkach
    for bar, val in zip(bars1, rouge_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, semantic_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Wynik (0-1)', fontsize=12, fontweight='bold')
    ax1.set_title('Porównanie jakości najlepszych konfiguracji', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Prawy wykres: Latencja
    ax2 = axes[1]
    
    latency_vals = [c['latency'] for c in best_configs]
    colors_bars = [COLOR_LATENCY if c['experiment'] != 'baseline' else 'gray' 
                   for c in best_configs]
    
    bars3 = ax2.bar(x, latency_vals, width*1.5, color=colors_bars, alpha=0.8)
    
    for bar, val in zip(bars3, latency_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}s',
                ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Latencja (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Porównanie czasu odpowiedzi', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Linia średniej
    avg_latency = np.mean(latency_vals)
    ax2.axhline(y=avg_latency, color='red', linestyle='--', alpha=0.5,
               label=f'Średnia ({avg_latency:.2f}s)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Zapisano: {output_path}")
    plt.close()


# ============================================================================
# TABELE TEKSTOWE + LATEX
# ============================================================================

def generate_summary_tables(data: dict, output_path: str):
    """
    Generuje tabele podsumowujące w formacie tekstowym i LaTeX.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PODSUMOWANIE WYNIKÓW EKSPERYMENTÓW")
    lines.append("=" * 80)
    
    for exp_name, param_key, title in [
        ('chunk_size', 'chunk_size', 'ROZMIAR CHUNKA'),
        ('k_values', 'k', 'LICZBA DOKUMENTÓW (K)'),
        ('overlap', 'chunk_overlap', 'OVERLAP')
    ]:
        results = data[exp_name]['results']
        best_name = data[exp_name]['best']
        
        lines.append(f"\n{'─' * 80}")
        lines.append(f"📊 {title}")
        lines.append(f"{'─' * 80}")
        lines.append(f"{'Konfiguracja':<20} {'ROUGE-1':<12} {'Semantic':<12} {'Latencja':<12} {'Chunks'}")
        lines.append("-" * 70)
        
        for r in results:
            name = r['config']['name']
            marker = " 🏆" if name == best_name else ""
            lines.append(f"{name:<20} {r['avg_rouge1_f1']:<12.3f} {r['avg_semantic_similarity']:<12.3f} {r['avg_latency']:<12.2f}s {r['num_chunks']}{marker}")
        
        lines.append(f"\n   Najlepsza konfiguracja: {best_name}")
    
    # Statystyki ogólne
    all_rouge = []
    all_semantic = []
    all_latency = []
    
    for exp_name in ['chunk_size', 'k_values', 'overlap']:
        for r in data[exp_name]['results']:
            all_rouge.append(r['avg_rouge1_f1'])
            all_semantic.append(r['avg_semantic_similarity'])
            all_latency.append(r['avg_latency'])
    
    lines.append(f"\n{'=' * 80}")
    lines.append("📈 STATYSTYKI OGÓLNE (wszystkie 15 konfiguracji)")
    lines.append(f"{'=' * 80}")
    lines.append(f"\nROUGE-1 F1:")
    lines.append(f"   Min:     {min(all_rouge):.3f}")
    lines.append(f"   Max:     {max(all_rouge):.3f}")
    lines.append(f"   Średnia: {np.mean(all_rouge):.3f}")
    lines.append(f"   Std:     {np.std(all_rouge):.3f}")
    
    lines.append(f"\nSemantic Similarity:")
    lines.append(f"   Min:     {min(all_semantic):.3f}")
    lines.append(f"   Max:     {max(all_semantic):.3f}")
    lines.append(f"   Średnia: {np.mean(all_semantic):.3f}")
    lines.append(f"   Std:     {np.std(all_semantic):.3f}")
    
    lines.append(f"\nLatencja:")
    lines.append(f"   Min:     {min(all_latency):.2f}s")
    lines.append(f"   Max:     {max(all_latency):.2f}s")
    lines.append(f"   Średnia: {np.mean(all_latency):.2f}s")
    
    # LaTeX tables
    lines.append(f"\n{'=' * 80}")
    lines.append("📋 TABELE LATEX")
    lines.append(f"{'=' * 80}")
    
    for exp_name, title in [
        ('chunk_size', 'Wpływ rozmiaru chunka'),
        ('k_values', 'Wpływ liczby dokumentów (k)'),
        ('overlap', 'Wpływ overlap')
    ]:
        results = data[exp_name]['results']
        
        lines.append(f"\n% {title}")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + title + "}")
        lines.append(r"\begin{tabular}{|l|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"\textbf{Konfiguracja} & \textbf{ROUGE-1} & \textbf{Semantic} & \textbf{Latencja (s)} & \textbf{Chunki} \\")
        lines.append(r"\hline")
        
        for r in results:
            name = r['config']['name'].replace('_', r'\_')
            lines.append(f"{name} & {r['avg_rouge1_f1']:.3f} & {r['avg_semantic_similarity']:.3f} & {r['avg_latency']:.2f} & {r['num_chunks']} \\\\")
        
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    
    # Zapisz
    content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"   ✅ Zapisano: {output_path}")
    
    # Wyświetl też w konsoli
    print("\n" + content)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║         WIZUALIZACJA EKSPERYMENTÓW RAG                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Użycie:                                                             ║
║      python visualize_experiments.py <experiments_results.json>      ║
║                                                                      ║
║  Przykład:                                                           ║
║      python visualize_experiments.py experiments_results_*.json      ║
║                                                                      ║
║  Generuje:                                                           ║
║      1. param_impact_grid.png      - Wpływ parametrów (grid 2x3)     ║
║      2. chunk_size_analysis.png    - Analiza chunk_size              ║
║      3. k_analysis.png             - Analiza k                       ║
║      4. overlap_analysis.png       - Analiza overlap                 ║
║      5. quality_vs_latency.png     - Trade-off jakość vs czas        ║
║      6. best_configs_comparison.png - Porównanie najlepszych         ║
║      7. summary_tables.txt         - Tabele (tekst + LaTeX)          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(input_file).parent
    
    print(f"\n📂 Wczytuję: {input_file}")
    data = load_results(input_file)
    print(f"   ✅ Wczytano dane z {len(data.get('chunk_size', {}).get('results', []))} + "
          f"{len(data.get('k_values', {}).get('results', []))} + "
          f"{len(data.get('overlap', {}).get('results', []))} konfiguracji")
    
    print(f"\n📊 Generuję wykresy...\n")
    
    # 1. Grid 2x3
    plot_param_impact_grid(data, output_dir / "param_impact_grid.png")
    
    # 2-4. Szczegółowe analizy
    plot_single_param_analysis(
        data, 'chunk_size', 'chunk_size',
        'Rozmiar chunka (znaki)', 
        'Wpływ rozmiaru chunka na jakość RAG',
        output_dir / "chunk_size_analysis.png"
    )
    
    plot_single_param_analysis(
        data, 'k_values', 'k',
        'Liczba dokumentów (k)', 
        'Wpływ liczby pobieranych dokumentów na jakość RAG',
        output_dir / "k_analysis.png"
    )
    
    plot_single_param_analysis(
        data, 'overlap', 'chunk_overlap',
        'Overlap (znaki)', 
        'Wpływ overlap między chunkami na jakość RAG',
        output_dir / "overlap_analysis.png"
    )
    
    # 5. Quality vs Latency
    plot_quality_vs_latency(data, output_dir / "quality_vs_latency.png")
    
    # 6. Best configs comparison
    plot_best_configs_comparison(data, output_dir / "best_configs_comparison.png")
    
    # 7. Summary tables
    generate_summary_tables(data, output_dir / "summary_tables.txt")
    
    print(f"\n{'=' * 60}")
    print("✅ WSZYSTKIE WYKRESY WYGENEROWANE!")
    print(f"{'=' * 60}")
    print(f"\nPliki w katalogu: {output_dir}")
    print("""
Sugerowana kolejność w pracy:
   1. param_impact_grid.png      → Rozdział "Eksperymenty" (przegląd)
   2. chunk_size_analysis.png    → Sekcja "Wpływ rozmiaru chunka"
   3. k_analysis.png             → Sekcja "Wpływ liczby dokumentów"
   4. overlap_analysis.png       → Sekcja "Wpływ overlap"
   5. quality_vs_latency.png     → Sekcja "Trade-off jakość vs wydajność"
   6. best_configs_comparison.png → Sekcja "Optymalna konfiguracja"
   7. summary_tables.txt         → Tabele do wklejenia (tekst + LaTeX)
    """)


if __name__ == "__main__":
    main()