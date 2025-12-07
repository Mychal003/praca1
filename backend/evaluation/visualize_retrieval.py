#!/usr/bin/env python3
"""
Skrypt do wizualizacji wyników ewaluacji RETRIEVAL
Generuje wykresy do pracy inżynierskiej

Użycie:
    python visualize_retrieval.py retrieval_results_20251204_191151.json
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

# Kolory
COLORS = {
    'precision': '#2E86AB',
    'recall': '#F18F01',
    'f1': '#C73E1D',
    'ndcg': '#A23B72',
    'mrr': '#28A745'
}


def load_data(filepath):
    """Wczytuje dane z pliku JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(summary, k_values=[1, 3, 5, 10]):
    """Wyciąga metryki z summary"""
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'ndcg': []
    }
    
    for k in k_values:
        metrics['precision'].append(summary.get(f'avg_precision@{k}', 0))
        metrics['recall'].append(summary.get(f'avg_recall@{k}', 0))
        metrics['f1'].append(summary.get(f'avg_f1@{k}', 0))
        metrics['ndcg'].append(summary.get(f'avg_ndcg@{k}', 0))
    
    metrics['mrr'] = summary.get('avg_mrr', 0)
    metrics['ap'] = summary.get('avg_ap', 0)
    
    return metrics


def plot_precision_recall(metrics, k_values, output_dir):
    """Wykres 1: Precision i Recall dla różnych k"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, metrics['precision'], width, 
                   label='Precision@k', color=COLORS['precision'], edgecolor='white')
    bars2 = ax.bar(x + width/2, metrics['recall'], width,
                   label='Recall@k', color=COLORS['recall'], edgecolor='white')
    
    # Dodaj wartości na słupkach
    for bar, val in zip(bars1, metrics['precision']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, metrics['recall']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Liczba pobieranych fragmentów (k)')
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Porównanie Precision i Recall dla różnych wartości k')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_precision_recall.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: retrieval_precision_recall.png")


def plot_retrieval_metrics_summary(metrics, k_values, output_dir):
    """Wykres 2: Podsumowanie wszystkich metryk dla k=5"""
    
    # Znajdź indeks k=5
    k5_idx = k_values.index(5) if 5 in k_values else 2
    
    metric_names = ['Precision@5', 'Recall@5', 'F1@5', 'NDCG@5', 'MRR']
    values = [
        metrics['precision'][k5_idx],
        metrics['recall'][k5_idx],
        metrics['f1'][k5_idx],
        metrics['ndcg'][k5_idx],
        metrics['mrr']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS['precision'], COLORS['recall'], COLORS['f1'], 
              COLORS['ndcg'], COLORS['mrr']]
    
    bars = ax.bar(metric_names, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Dodaj wartości
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Podsumowanie metryk wyszukiwania (k=5)')
    ax.set_ylim(0, 1.15)
    
    # Linia odniesienia
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Próg 0.5')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_metrics_summary.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: retrieval_metrics_summary.png")


def plot_ndcg(metrics, k_values, output_dir):
    """Wykres 3: NDCG dla różnych k"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, metrics['ndcg'], 'o-', color=COLORS['ndcg'], 
            linewidth=2, markersize=10, label='NDCG@k')
    
    # Dodaj wartości
    for k, val in zip(k_values, metrics['ndcg']):
        ax.annotate(f'{val:.3f}', xy=(k, val), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=10)
    
    ax.set_xlabel('Liczba pobieranych fragmentów (k)')
    ax.set_ylabel('NDCG@k')
    ax.set_title('Zmiana NDCG w zależności od liczby pobieranych fragmentów')
    ax.set_xticks(k_values)
    ax.set_ylim(0.6, 1.0)
    ax.legend()
    
    # Adnotacja
    ax.annotate('NDCG mierzy jakość rankingu\n(czy trafne fragmenty są wysoko)', 
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_ndcg.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: retrieval_ndcg.png")


def plot_f1_scores(metrics, k_values, output_dir):
    """Wykres 4: F1 Score dla różnych k"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, metrics['f1'], 's-', color=COLORS['f1'], 
            linewidth=2, markersize=10, label='F1@k')
    
    # Znajdź najlepsze k
    best_idx = np.argmax(metrics['f1'])
    best_k = k_values[best_idx]
    best_f1 = metrics['f1'][best_idx]
    
    # Zaznacz najlepszy punkt
    ax.scatter([best_k], [best_f1], s=200, c='gold', marker='*', 
               zorder=5, edgecolors='black', linewidths=1)
    
    # Dodaj wartości
    for k, val in zip(k_values, metrics['f1']):
        ax.annotate(f'{val:.3f}', xy=(k, val), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=10)
    
    ax.set_xlabel('Liczba pobieranych fragmentów (k)')
    ax.set_ylabel('F1@k')
    ax.set_title('F1 Score dla różnych wartości k')
    ax.set_xticks(k_values)
    ax.set_ylim(0, 0.7)
    ax.legend()
    
    ax.annotate(f'Najlepszy: k={best_k}\nF1={best_f1:.3f}', 
                xy=(best_k, best_f1), xytext=(30, -20),
                textcoords='offset points', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_f1.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: retrieval_f1.png")


def plot_all_metrics_lines(metrics, k_values, output_dir):
    """Wykres 5: Wszystkie metryki na jednym wykresie liniowym"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(k_values, metrics['precision'], 'o-', color=COLORS['precision'], 
            linewidth=2, markersize=8, label='Precision@k')
    ax.plot(k_values, metrics['recall'], 's-', color=COLORS['recall'], 
            linewidth=2, markersize=8, label='Recall@k')
    ax.plot(k_values, metrics['f1'], '^-', color=COLORS['f1'], 
            linewidth=2, markersize=8, label='F1@k')
    ax.plot(k_values, metrics['ndcg'], 'D-', color=COLORS['ndcg'], 
            linewidth=2, markersize=8, label='NDCG@k')
    
    # MRR jako linia pozioma
    ax.axhline(y=metrics['mrr'], color=COLORS['mrr'], linestyle='--', 
               linewidth=2, label=f"MRR = {metrics['mrr']:.3f}")
    
    ax.set_xlabel('Liczba pobieranych fragmentów (k)')
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Porównanie wszystkich metryk retrieval')
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_all_metrics.png', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Zapisano: retrieval_all_metrics.png")


def generate_latex_table(metrics, k_values):
    """Generuje tabelę LaTeX"""
    
    print("\n" + "="*70)
    print("📋 TABELA LATEX (skopiuj do pracy)")
    print("="*70 + "\n")
    
    print(r"""\begin{table}[H]
\centering
\caption{Wyniki ewaluacji wyszukiwania}
\label{tab:retrieval_results}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Metryka} & \textbf{k=1} & \textbf{k=3} & \textbf{k=5} & \textbf{k=10} \\
\hline""")
    
    # Precision
    print(f"Precision@k & {metrics['precision'][0]:.3f} & {metrics['precision'][1]:.3f} & {metrics['precision'][2]:.3f} & {metrics['precision'][3]:.3f} \\\\")
    
    # Recall
    print(f"Recall@k & {metrics['recall'][0]:.3f} & {metrics['recall'][1]:.3f} & {metrics['recall'][2]:.3f} & {metrics['recall'][3]:.3f} \\\\")
    
    # F1
    print(f"F1@k & {metrics['f1'][0]:.3f} & {metrics['f1'][1]:.3f} & {metrics['f1'][2]:.3f} & {metrics['f1'][3]:.3f} \\\\")
    
    # NDCG
    print(f"NDCG@k & {metrics['ndcg'][0]:.3f} & {metrics['ndcg'][1]:.3f} & {metrics['ndcg'][2]:.3f} & {metrics['ndcg'][3]:.3f} \\\\")
    
    print(r"\hline")
    print(f"MRR & \\multicolumn{{4}}{{c|}}{{{metrics['mrr']:.3f}}} \\\\")
    print(r"""\hline
\end{tabular}
\end{table}""")
    
    print("\n" + "="*70 + "\n")


def main():
    # Ścieżki
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path('retrieval_results_20251204_191151.json')
    
    output_dir = Path('../images/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Wczytuję: {input_file}")
    data = load_data(input_file)
    
    summary = data['summary']
    k_values = data.get('config', {}).get('k_values', [1, 3, 5, 10])
    
    print(f"   ✅ Wczytano dane dla k={k_values}\n")
    
    # Wyciągnij metryki
    metrics = extract_metrics(summary, k_values)
    
    print("📊 Generuję wykresy RETRIEVAL...\n")
    
    # Generuj wykresy
    plot_precision_recall(metrics, k_values, output_dir)
    plot_retrieval_metrics_summary(metrics, k_values, output_dir)
    plot_ndcg(metrics, k_values, output_dir)
    plot_f1_scores(metrics, k_values, output_dir)
    plot_all_metrics_lines(metrics, k_values, output_dir)
    
    # Generuj tabelę LaTeX
    generate_latex_table(metrics, k_values)
    
    print("="*70)
    print("✅ WSZYSTKIE WYKRESY RETRIEVAL WYGENEROWANE!")
    print("="*70)
    print(f"\nPliki w katalogu: {output_dir}/")
    print("""
Wygenerowane pliki:
   1. retrieval_precision_recall.png  → Precision vs Recall
   2. retrieval_metrics_summary.png   → Podsumowanie dla k=5
   3. retrieval_ndcg.png              → NDCG dla różnych k
   4. retrieval_f1.png                → F1 Score dla różnych k
   5. retrieval_all_metrics.png       → Wszystkie metryki razem
""")


if __name__ == '__main__':
    main()