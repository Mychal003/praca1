"""
Skrypt do wizualizacji porównania konfiguracji LLM Judge.
Generuje wykresy słupkowe do pracy inżynierskiej.

Użycie:
    python visualize_llm_comparison.py <plik1.json> <plik2.json> [plik3.json]
    
Przykład:
    python visualize_llm_comparison.py generation_results_with_llm_judge_20251204_023303.json generation_results_800_10_100_with_llm_judge_20251207_225926.json generation_results_1200_10_0_with_llm_judge_20251205_222026.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Backend bez GUI
import matplotlib.pyplot as plt
import numpy as np

# Konfiguracja stylu
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Kolory - dodany trzeci kolor
COLORS = ['#2E86AB', '#F18F01', '#A23B72']  # Niebieski, Pomarańczowy, Fioletowy


def load_results(filepath):
    """Wczytuje wyniki z pliku JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_config_name(data):
    """Wyciąga nazwę konfiguracji z danych."""
    config = data.get('config', {})
    chunk_size = config.get('chunk_size', '?')
    k = config.get('pipeline_k', '?')
    overlap = config.get('chunk_overlap', '0')
    return f"{chunk_size}/{k}/{overlap}"


def plot_llm_judge_comparison(configs, output_dir):
    """Wykres 1: Porównanie wymiarów LLM Judge."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dimensions = ['Correctness', 'Completeness', 'Relevance', 'Groundedness', 'Overall']
    keys = ['avg_llm_correctness', 'avg_llm_completeness', 'avg_llm_relevance', 
            'avg_llm_groundedness', 'avg_llm_overall']
    
    x = np.arange(len(dimensions))
    n_configs = len(configs)
    width = 0.8 / n_configs
    
    for i, c in enumerate(configs):
        values = [c['summary'].get(k, 0) for k in keys]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=c['name'], color=COLORS[i], edgecolor='white')
        
        # Dodaj wartości na słupkach
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Wymiar oceny')
    ax.set_ylabel('Wartość (0.0 - 1.0)')
    ax.set_title('Porównanie wymiarów LLM Judge dla różnych konfiguracji')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    
    # Linia odniesienia
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(4.5, 0.905, 'Próg 0.9', fontsize=9, color='green', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_judge_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_judge_comparison.png")


def plot_all_metrics_comparison(configs, output_dir):
    """Wykres 2: Porównanie wszystkich metryk."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metrics = [
        ('ROUGE-1', 'avg_rouge1_f1'),
        ('Semantic', 'avg_semantic_similarity'),
        ('Correctness', 'avg_llm_correctness'),
        ('Completeness', 'avg_llm_completeness'),
        ('Relevance', 'avg_llm_relevance'),
        ('Groundedness', 'avg_llm_groundedness'),
        ('Overall', 'avg_llm_overall'),
    ]
    
    labels = [m[0] for m in metrics]
    keys = [m[1] for m in metrics]
    
    x = np.arange(len(labels))
    n_configs = len(configs)
    width = 0.8 / n_configs
    
    for i, c in enumerate(configs):
        values = [c['summary'].get(k, 0) for k in keys]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=c['name'], color=COLORS[i], edgecolor='white')
        
        # Dodaj wartości na słupkach
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Metryka')
    ax.set_ylabel('Wartość')
    ax.set_title('Porównanie wszystkich metryk dla różnych konfiguracji')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    
    # Separator wizualny między metrykami auto a LLM
    ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.5, 1.08, 'Metryki auto', ha='center', fontsize=9, color='gray')
    ax.text(4.5, 1.08, 'LLM Judge', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_all_metrics_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_all_metrics_comparison.png")


def plot_category_comparison(configs, output_dir):
    """Wykres 3: Porównanie per kategoria."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = ['factual', 'procedural', 'troubleshooting']
    cat_labels = ['Factual', 'Procedural', 'Troubleshooting']
    
    x = np.arange(len(categories))
    n_configs = len(configs)
    width = 0.8 / n_configs
    
    # ROUGE-1
    ax1 = axes[0]
    for i, c in enumerate(configs):
        values = [c['summary'].get(f'{cat}_rouge1_f1', 0) for cat in categories]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax1.bar(x + offset, values, width, label=c['name'], color=COLORS[i], edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Kategoria pytań')
    ax1.set_ylabel('ROUGE-1 F1')
    ax1.set_title('ROUGE-1 F1 według kategorii')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.8)
    
    # Semantic Similarity
    ax2 = axes[1]
    for i, c in enumerate(configs):
        values = [c['summary'].get(f'{cat}_semantic', 0) for cat in categories]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax2.bar(x + offset, values, width, label=c['name'], color=COLORS[i], edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Kategoria pytań')
    ax2.set_ylabel('Semantic Similarity')
    ax2.set_title('Semantic Similarity według kategorii')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_labels)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_category_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_category_comparison.png")


def plot_radar_comparison(configs, output_dir):
    """Wykres 4: Radar chart porównujący konfiguracje."""
    
    dimensions = ['Correctness', 'Completeness', 'Relevance', 'Groundedness', 'Overall']
    keys = ['avg_llm_correctness', 'avg_llm_completeness', 'avg_llm_relevance', 
            'avg_llm_groundedness', 'avg_llm_overall']
    
    # Przygotuj dane
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # Zamknij wielokąt
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, c in enumerate(configs):
        values = [c['summary'].get(k, 0) for k in keys]
        values += values[:1]  # Zamknij wielokąt
        
        ax.plot(angles, values, 'o-', linewidth=2, label=c['name'], color=COLORS[i])
        ax.fill(angles, values, alpha=0.25, color=COLORS[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Porównanie konfiguracji - Radar Chart', size=14, y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_radar_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_radar_comparison.png")


def plot_difference_chart(configs, output_dir):
    """Wykres 5: Różnice między konfiguracjami."""
    
    if len(configs) != 2:
        print(f"   ⚠️ Wykres różnic wymaga dokładnie 2 konfiguracji (otrzymano {len(configs)})")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = [
        ('ROUGE-1', 'avg_rouge1_f1'),
        ('Semantic', 'avg_semantic_similarity'),
        ('Correctness', 'avg_llm_correctness'),
        ('Completeness', 'avg_llm_completeness'),
        ('Relevance', 'avg_llm_relevance'),
        ('Groundedness', 'avg_llm_groundedness'),
        ('Overall', 'avg_llm_overall'),
    ]
    
    labels = [m[0] for m in metrics]
    keys = [m[1] for m in metrics]
    
    # Oblicz różnice (config1 - config2)
    differences = []
    for k in keys:
        v1 = configs[0]['summary'].get(k, 0)
        v2 = configs[1]['summary'].get(k, 0)
        differences.append(v1 - v2)
    
    x = np.arange(len(labels))
    colors = ['#2E86AB' if d >= 0 else '#C73E1D' for d in differences]
    
    bars = ax.bar(x, differences, color=colors, edgecolor='white')
    
    # Dodaj wartości
    for bar, diff in zip(bars, differences):
        va = 'bottom' if diff >= 0 else 'top'
        offset = 0.005 if diff >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{diff:+.3f}', ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Metryka')
    ax.set_ylabel(f'Różnica ({configs[0]["name"]} - {configs[1]["name"]})')
    ax.set_title(f'Różnice między konfiguracjami\n(wartości dodatnie = {configs[0]["name"]} lepszy)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.1, 0.1)
    
    # Legenda kolorów
    ax.text(0.02, 0.98, f'■ {configs[0]["name"]} lepszy', transform=ax.transAxes, 
            fontsize=10, va='top', color='#2E86AB')
    ax.text(0.02, 0.93, f'■ {configs[1]["name"]} lepszy', transform=ax.transAxes, 
            fontsize=10, va='top', color='#C73E1D')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_difference_chart.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_difference_chart.png")


def main():
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║         VISUALIZE LLM JUDGE COMPARISON                               ║
╚══════════════════════════════════════════════════════════════════════╝

Użycie:
    python visualize_llm_comparison.py <plik1.json> <plik2.json> [plik3.json]

Przykład:
    python visualize_llm_comparison.py generation_results_with_llm_judge_20251204_023303.json generation_results_800_10_100_with_llm_judge_20251207_225926.json generation_results_1200_10_0_with_llm_judge_20251205_222026.json
        """)
        sys.exit(1)
    
    files = sys.argv[1:]
    
    if len(files) > 3:
        print(f"⚠️ Przekazano {len(files)} plików, obsługiwane są maksymalnie 3 konfiguracje.")
        sys.exit(1)
    
    # Wczytaj konfiguracje
    configs = []
    for filepath in files:
        print(f"📂 Wczytuję: {filepath}")
        data = load_results(filepath)
        config_name = extract_config_name(data)
        
        configs.append({
            'name': config_name,
            'file': filepath,
            'summary': data['summary'],
            'config': data.get('config', {})
        })
        print(f"   ✓ Konfiguracja: {config_name}")
    
    # Katalog wyjściowy
    output_dir = Path('../images/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generuję wykresy...\n")
    
    # Generuj wykresy
    plot_llm_judge_comparison(configs, output_dir)
    plot_all_metrics_comparison(configs, output_dir)
    plot_category_comparison(configs, output_dir)
    plot_radar_comparison(configs, output_dir)
    plot_difference_chart(configs, output_dir)
    
    print(f"""
{'='*70}
✅ WSZYSTKIE WYKRESY WYGENEROWANE!
{'='*70}

Pliki zapisane w: {output_dir}/

Wygenerowane wykresy:
   1. llm_judge_comparison.png     - Porównanie wymiarów LLM Judge
   2. llm_all_metrics_comparison.png - Wszystkie metryki
   3. llm_category_comparison.png  - Porównanie per kategoria
   4. llm_radar_comparison.png     - Radar chart
   5. llm_difference_chart.png     - Różnice między konfiguracjami

Dodaj do LaTeX:
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{images/results/llm_judge_comparison.png}}
\\caption{{Porównanie wymiarów LLM Judge dla różnych konfiguracji}}
\\label{{fig:llm_judge_comparison}}
\\end{{figure}}
""")


if __name__ == '__main__':
    main()