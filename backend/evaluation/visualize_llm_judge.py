#!/usr/bin/env python3
"""
Skrypt do wizualizacji wyników LLM Judge
Generuje wykresy do pracy inżynierskiej
"""

import json
import matplotlib
matplotlib.use('Agg')  # Backend bez GUI - naprawia błąd Tkinter na Windows
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Konfiguracja stylu
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Kolory
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'tertiary': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D',
    'factual': '#3498db',
    'procedural': '#2ecc71',
    'troubleshooting': '#e74c3c'
}


def load_data(filepath):
    """Wczytuje dane z pliku JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_llm_judge_dimensions(summary, output_dir):
    """Wykres 1: Wyniki per wymiar LLM Judge"""
    
    dimensions = ['Correctness', 'Completeness', 'Relevance', 'Groundedness', 'Overall']
    values = [
        summary['avg_llm_correctness'],
        summary['avg_llm_completeness'],
        summary['avg_llm_relevance'],
        summary['avg_llm_groundedness'],
        summary['avg_llm_overall']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS['primary'] if v >= 0.9 else COLORS['tertiary'] if v >= 0.85 else COLORS['success'] for v in values]
    
    bars = ax.bar(dimensions, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Dodaj wartości na słupkach
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Wynik (0-1)')
    ax.set_title('Ocena LLM Judge - wymiary jakości odpowiedzi')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Próg wysokiej jakości (0.9)')
    ax.legend(loc='lower right')
    
    # Linia dla średniej
    avg = np.mean(values)
    ax.axhline(y=avg, color='gray', linestyle=':', alpha=0.7)
    ax.text(4.5, avg + 0.02, f'Średnia: {avg:.3f}', ha='right', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_judge_dimensions.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_judge_dimensions.png")


def plot_category_comparison(summary, output_dir):
    """Wykres 2: Porównanie kategorii pytań"""
    
    categories = ['Factual', 'Procedural', 'Troubleshooting']
    rouge_scores = [
        summary['factual_rouge1_f1'],
        summary['procedural_rouge1_f1'],
        summary['troubleshooting_rouge1_f1']
    ]
    semantic_scores = [
        summary['factual_semantic'],
        summary['procedural_semantic'],
        summary['troubleshooting_semantic']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, rouge_scores, width, label='ROUGE-1 F1', 
                   color=COLORS['primary'], edgecolor='white')
    bars2 = ax.bar(x + width/2, semantic_scores, width, label='Semantic Similarity',
                   color=COLORS['tertiary'], edgecolor='white')
    
    # Dodaj wartości
    for bar, val in zip(bars1, rouge_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, semantic_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Wynik')
    ax.set_title('Jakość odpowiedzi według kategorii pytań')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: category_comparison.png")


def plot_rouge_vs_llm(detailed, output_dir):
    """Wykres 3: Scatter plot ROUGE vs LLM Judge Overall"""
    
    rouge_scores = []
    llm_scores = []
    categories = []
    
    for item in detailed:
        rouge_scores.append(item['metrics']['rouge1_f1'])
        llm_scores.append(item['metrics']['llm_judge']['overall'])
        categories.append(item['category'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Koloruj według kategorii
    for cat in ['factual', 'procedural', 'troubleshooting']:
        cat_rouge = [r for r, c in zip(rouge_scores, categories) if c == cat]
        cat_llm = [l for l, c in zip(llm_scores, categories) if c == cat]
        ax.scatter(cat_rouge, cat_llm, label=cat.capitalize(), 
                   color=COLORS[cat], s=100, alpha=0.7, edgecolors='white')
    
    # Linia trendu
    z = np.polyfit(rouge_scores, llm_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(rouge_scores), max(rouge_scores), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, label='Trend')
    
    # Korelacja
    correlation = np.corrcoef(rouge_scores, llm_scores)[0, 1]
    ax.text(0.05, 0.95, f'Korelacja: {correlation:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('ROUGE-1 F1')
    ax.set_ylabel('LLM Judge Overall')
    ax.set_title('Porównanie metryki automatycznej (ROUGE) z oceną LLM')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    
    # Zaznacz obszar rozbieżności
    ax.axvspan(0, 0.3, alpha=0.1, color='red', label='Niski ROUGE')
    ax.axhspan(0.8, 1.1, alpha=0.1, color='green', label='Wysoki LLM')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rouge_vs_llm_judge.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: rouge_vs_llm_judge.png")


def plot_llm_dimensions_radar(summary, output_dir):
    """Wykres 4: Radar chart wymiarów LLM Judge"""
    
    dimensions = ['Correctness', 'Completeness', 'Relevance', 'Groundedness', 'Overall']
    values = [
        summary['avg_llm_correctness'],
        summary['avg_llm_completeness'],
        summary['avg_llm_relevance'],
        summary['avg_llm_groundedness'],
        summary['avg_llm_overall']
    ]
    
    # Zamknij wykres
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['primary'])
    ax.fill(angles, values, alpha=0.25, color=COLORS['primary'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=12)
    ax.set_ylim(0, 1)
    ax.set_title('Profil jakości odpowiedzi - LLM Judge', size=14, y=1.1)
    
    # Dodaj wartości przy punktach
    for angle, value, dim in zip(angles[:-1], values[:-1], dimensions):
        ax.annotate(f'{value:.3f}', xy=(angle, value), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_judge_radar.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: llm_judge_radar.png")


def plot_groundedness_analysis(detailed, output_dir):
    """Wykres 5: Analiza Groundedness - najsłabszy wymiar"""
    
    questions_short = []
    groundedness_scores = []
    overall_scores = []
    
    for i, item in enumerate(detailed):
        questions_short.append(f"Q{i+1}")
        groundedness_scores.append(item['metrics']['llm_judge']['groundedness'])
        overall_scores.append(item['metrics']['llm_judge']['overall'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(questions_short))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, groundedness_scores, width, label='Groundedness',
                   color=COLORS['success'], edgecolor='white')
    bars2 = ax.bar(x + width/2, overall_scores, width, label='Overall',
                   color=COLORS['primary'], edgecolor='white', alpha=0.7)
    
    ax.set_ylabel('Wynik')
    ax.set_title('Groundedness vs Overall dla każdego pytania')
    ax.set_xticks(x)
    ax.set_xticklabels(questions_short, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.axhline(y=0.84, color='red', linestyle='--', alpha=0.5, label='Średnia Groundedness')
    
    # Zaznacz pytania z niskim groundedness
    for i, (g, o) in enumerate(zip(groundedness_scores, overall_scores)):
        if g < 0.7:
            ax.annotate('⚠️', xy=(i, g), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'groundedness_analysis.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: groundedness_analysis.png")


def plot_metrics_heatmap(detailed, output_dir):
    """Wykres 6: Heatmapa wszystkich metryk per pytanie"""
    
    # Przygotuj dane
    metrics_names = ['ROUGE-1', 'Semantic', 'Correctness', 'Completeness', 
                     'Relevance', 'Groundedness', 'Overall']
    
    data = []
    questions = []
    
    for i, item in enumerate(detailed):
        questions.append(f"Q{i+1}")
        row = [
            item['metrics']['rouge1_f1'],
            item['metrics']['semantic_similarity'],
            item['metrics']['llm_judge']['correctness'],
            item['metrics']['llm_judge']['completeness'],
            item['metrics']['llm_judge']['relevance'],
            item['metrics']['llm_judge']['groundedness'],
            item['metrics']['llm_judge']['overall']
        ]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(questions)))
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.set_yticklabels(questions)
    
    # Dodaj wartości w komórkach
    for i in range(len(questions)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title('Mapa cieplna wszystkich metryk dla każdego pytania')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Wynik', rotation=-90, va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: metrics_heatmap.png")


def generate_interesting_cases(detailed, output_dir):
    """Generuje plik z interesującymi przypadkami do analizy"""
    
    cases = []
    
    for i, item in enumerate(detailed):
        rouge = item['metrics']['rouge1_f1']
        llm_overall = item['metrics']['llm_judge']['overall']
        groundedness = item['metrics']['llm_judge']['groundedness']
        
        # Przypadek 1: Niski ROUGE ale wysoki LLM
        if rouge < 0.4 and llm_overall >= 0.8:
            cases.append({
                'type': 'LOW_ROUGE_HIGH_LLM',
                'question': item['question'],
                'expected': item['expected'],
                'generated': item['generated'],
                'rouge': rouge,
                'llm_overall': llm_overall,
                'interpretation': 'ROUGE karze za dodatkowe/inne słowa, ale odpowiedź jest poprawna'
            })
        
        # Przypadek 2: Niski Groundedness
        if groundedness < 0.7:
            cases.append({
                'type': 'LOW_GROUNDEDNESS',
                'question': item['question'],
                'generated': item['generated'],
                'groundedness': groundedness,
                'interpretation': 'System dodał informacje spoza kontekstu'
            })
    
    with open(output_dir / 'interesting_cases.json', 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Zapisano: interesting_cases.json ({len(cases)} przypadków)")
    
    return cases


def main():
    import sys
    
    # Ścieżki - można podać plik jako argument
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path('generation_results_with_llm_judge_20251127_005025.json')
    
    output_dir = Path('../images_v2/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Wczytuję: {input_file}")
    data = load_data(input_file)
    
    summary = data['summary']
    detailed = data['detailed']
    
    print(f"   ✅ Wczytano {len(detailed)} wyników\n")
    
    print("📊 Generuję wykresy LLM Judge...\n")
    
    # Generuj wykresy
    plot_llm_judge_dimensions(summary, output_dir)
    plot_category_comparison(summary, output_dir)
    plot_rouge_vs_llm(detailed, output_dir)
    plot_llm_dimensions_radar(summary, output_dir)
    plot_groundedness_analysis(detailed, output_dir)
    plot_metrics_heatmap(detailed, output_dir)
    
    print("\n📋 Generuję analizę przypadków...\n")
    cases = generate_interesting_cases(detailed, output_dir)
    
    # Podsumowanie
    print("\n" + "="*70)
    print("✅ WSZYSTKIE WYKRESY LLM JUDGE WYGENEROWANE!")
    print("="*70)
    print(f"\nPliki w katalogu: {output_dir}/")
    print("""
Sugerowana kolejność w pracy:
   1. llm_judge_dimensions.png    → Główne wyniki LLM Judge
   2. llm_judge_radar.png         → Profil jakości (alternatywa)
   3. category_comparison.png     → Analiza per kategoria
   4. rouge_vs_llm_judge.png      → Porównanie metryk
   5. groundedness_analysis.png   → Analiza słabego wymiaru
   6. metrics_heatmap.png         → Szczegółowy przegląd
   7. interesting_cases.json      → Przypadki do omówienia w tekście
""")


if __name__ == '__main__':
    main()