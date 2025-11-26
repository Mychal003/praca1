"""
Uniwersalna wizualizacja wyników ewaluacji RAG.
Auto-wykrywa typ wyników i generuje odpowiednie wykresy.

Użycie:
    python visualize.py <results.json>
    python visualize.py <results.json> --latex  # Dodatkowo generuje tabele LaTeX
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
from datetime import datetime

# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_result_type(data) -> str:
    """
    Auto-wykrywa typ wyników na podstawie struktury JSON.
    
    Returns:
        'experiments' | 'retrieval' | 'generation' | 'unknown'
    """
    # Format: experiments (run_experiments.py)
    if isinstance(data, dict) and 'chunk_size' in data and 'k_values' in data:
        return 'experiments'
    
    # Format: retrieval (evaluate_retrieval.py)
    if isinstance(data, dict) and 'summary' in data:
        if 'avg_precision@5' in data.get('summary', {}):
            return 'retrieval'
        if 'avg_rouge1_f1' in data.get('summary', {}):
            return 'generation'
    
    # Format: lista konfiguracji (stary format)
    if isinstance(data, list) and len(data) > 0:
        if 'config' in data[0] and 'summary' in data[0]:
            return 'experiments_old'
    
    return 'unknown'


# ============================================================================
# EXPERIMENTS VISUALIZATION
# ============================================================================

def visualize_experiments(data: dict, output_prefix: str, generate_latex: bool = False):
    """
    Wizualizacja wyników eksperymentów (chunk_size, k, overlap).
    """
    print("📊 Generuję wykresy dla EXPERIMENTS...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Chunk Size vs ROUGE & Semantic
    ax1 = plt.subplot(2, 3, 1)
    plot_experiment_line(ax1, data['chunk_size']['results'], 
                        'chunk_size', ['avg_rouge1_f1', 'avg_semantic_similarity'],
                        'Chunk Size: Quality Metrics', 'Chunk Size')
    
    # 2. K vs ROUGE & Semantic
    ax2 = plt.subplot(2, 3, 2)
    plot_experiment_line(ax2, data['k_values']['results'], 
                        'k', ['avg_rouge1_f1', 'avg_semantic_similarity'],
                        'K (Documents): Quality Metrics', 'k')
    
    # 3. Overlap vs ROUGE & Semantic
    ax3 = plt.subplot(2, 3, 3)
    plot_experiment_line(ax3, data['overlap']['results'], 
                        'chunk_overlap', ['avg_rouge1_f1', 'avg_semantic_similarity'],
                        'Overlap: Quality Metrics', 'Chunk Overlap')
    
    # 4. Chunk Size: Quality vs Latency Trade-off
    ax4 = plt.subplot(2, 3, 4)
    plot_tradeoff(ax4, data['chunk_size']['results'], 'Chunk Size Trade-off')
    
    # 5. K: Quality vs Latency Trade-off
    ax5 = plt.subplot(2, 3, 5)
    plot_tradeoff(ax5, data['k_values']['results'], 'K Trade-off')
    
    # 6. Best Configurations Summary
    ax6 = plt.subplot(2, 3, 6)
    plot_best_summary(ax6, data)
    
    plt.tight_layout()
    
    output_file = f"{output_prefix}_experiments_charts.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file}")
    plt.close()
    
    if generate_latex:
        generate_experiments_latex(data)


def plot_experiment_line(ax, results: list, param_key: str, metric_keys: list, 
                         title: str, xlabel: str):
    """Wykres liniowy dla eksperymentu."""
    params = [r['config'][param_key] for r in results]
    
    colors = ['#667eea', '#f093fb']
    labels = ['ROUGE-1 F1', 'Semantic Sim.']
    
    for metric, color, label in zip(metric_keys, colors, labels):
        values = [r[metric] for r in results]
        ax.plot(params, values, marker='o', linewidth=2, markersize=8, 
               color=color, label=label)
        
        # Wartości na punktach
        for p, v in zip(params, values):
            ax.annotate(f'{v:.3f}', (p, v), textcoords="offset points", 
                       xytext=(0, 8), ha='center', fontsize=8)
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)


def plot_tradeoff(ax, results: list, title: str):
    """Scatter plot jakość vs latencja."""
    latencies = [r['avg_latency'] for r in results]
    rouge_scores = [r['avg_rouge1_f1'] for r in results]
    names = [r['config']['name'] for r in results]
    
    scatter = ax.scatter(latencies, rouge_scores, s=150, alpha=0.7, 
                        c=range(len(results)), cmap='viridis')
    
    for lat, rouge, name in zip(latencies, rouge_scores, names):
        ax.annotate(name, (lat, rouge), textcoords="offset points", 
                   xytext=(5, 5), ha='left', fontsize=8)
    
    ax.set_xlabel('Latency (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ROUGE-1 F1', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_best_summary(ax, data: dict):
    """Podsumowanie najlepszych konfiguracji."""
    experiments = ['chunk_size', 'k_values', 'overlap']
    best_names = [data[exp]['best'] for exp in experiments]
    best_scores = []
    
    for exp in experiments:
        best_config = data[exp]['best']
        for r in data[exp]['results']:
            if r['config']['name'] == best_config:
                best_scores.append(r['avg_rouge1_f1'])
                break
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(experiments, best_scores, color=colors, alpha=0.8)
    
    ax.set_ylabel('ROUGE-1 F1', fontsize=11, fontweight='bold')
    ax.set_title('Best Configuration per Experiment', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, score, name in zip(bars, best_scores, best_names):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.02,
               f'{name}\n{score:.3f}', ha='center', fontsize=9, fontweight='bold')


def generate_experiments_latex(data: dict):
    """Generuje tabele LaTeX."""
    print("\n📋 TABELA LATEX - Experiments")
    print("="*70)
    
    for exp_name in ['chunk_size', 'k_values', 'overlap']:
        results = data[exp_name]['results']
        
        print(f"\n% {exp_name.upper()}")
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\begin{tabular}{|l|c|c|c|}")
        print(r"\hline")
        print(r"\textbf{Config} & \textbf{ROUGE-1} & \textbf{Semantic} & \textbf{Latency} \\")
        print(r"\hline")
        
        for r in results:
            name = r['config']['name']
            rouge = r['avg_rouge1_f1']
            semantic = r['avg_semantic_similarity']
            latency = r['avg_latency']
            print(f"{name} & {rouge:.3f} & {semantic:.3f} & {latency:.2f}s \\\\")
        
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ============================================================================
# RETRIEVAL VISUALIZATION
# ============================================================================

def visualize_retrieval(data: dict, output_prefix: str, generate_latex: bool = False):
    """
    Wizualizacja wyników retrieval - osobne pliki dla każdego wykresu.
    """
    print("📊 Generuję wykresy dla RETRIEVAL...")
    
    summary = data['summary']
    k_values = [1, 3, 5, 10]
    
    # =========================================================================
    # WYKRES 1: Precision vs Recall at different k
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    precision = [summary[f'avg_precision@{k}'] for k in k_values]
    recall = [summary[f'avg_recall@{k}'] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, precision, width, label='Precision@k', color='#667eea', alpha=0.8)
    bars2 = ax1.bar(x + width/2, recall, width, label='Recall@k', color='#f093fb', alpha=0.8)
    
    ax1.set_xlabel('k', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file1 = f"{output_prefix}_precision_recall.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file1}")
    plt.close()
    
    # =========================================================================
    # WYKRES 2: All retrieval metrics
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5']
    values = [summary[f'avg_{m}'] for m in metrics]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    bars = ax2.bar(range(len(metrics)), values, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m.upper() for m in metrics], rotation=45, ha='right')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Retrieval Metrics Summary', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file2 = f"{output_prefix}_retrieval_metrics.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file2}")
    plt.close()
    
    # =========================================================================
    # WYKRES 3: NDCG progression
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ndcg = [summary[f'avg_ndcg@{k}'] for k in k_values]
    
    ax3.plot(k_values, ndcg, marker='o', linewidth=2, markersize=10, color='#667eea')
    ax3.fill_between(k_values, ndcg, alpha=0.3, color='#667eea')
    ax3.set_xlabel('k', fontsize=11, fontweight='bold')
    ax3.set_ylabel('NDCG@k', fontsize=11, fontweight='bold')
    ax3.set_title('NDCG: Ranking Quality', fontsize=12, fontweight='bold')
    ax3.set_xticks(k_values)
    ax3.set_ylim(0, 1.0)
    ax3.grid(alpha=0.3)
    
    for k, n in zip(k_values, ndcg):
        ax3.annotate(f'{n:.3f}', (k, n), textcoords="offset points", 
                    xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file3 = f"{output_prefix}_ndcg.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file3}")
    plt.close()
    
    if generate_latex:
        generate_retrieval_latex(data)

# ============================================================================
# GENERATION VISUALIZATION
# ============================================================================

def visualize_generation(data: dict, output_prefix: str, generate_latex: bool = False):
    """
    Wizualizacja wyników generation (z opcjonalnym LLM Judge).
    """
    print("📊 Generuję wykresy dla GENERATION...")
    
    summary = data['summary']
    detailed = data['detailed']
    has_llm = 'avg_llm_overall' in summary
    
    if has_llm:
        fig = plt.figure(figsize=(18, 10))
        rows, cols = 2, 3
    else:
        fig = plt.figure(figsize=(16, 5))
        rows, cols = 1, 3
    
    # 1. Basic metrics comparison
    ax1 = plt.subplot(rows, cols, 1)
    metrics = ['ROUGE-1', 'Semantic']
    values = [summary['avg_rouge1_f1'], summary['avg_semantic_similarity']]
    colors = ['#667eea', '#f093fb']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Generation Quality Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 2. ROUGE vs Semantic scatter
    ax2 = plt.subplot(rows, cols, 2)
    rouge = [r['metrics']['rouge1_f1'] for r in detailed]
    semantic = [r['metrics']['semantic_similarity'] for r in detailed]
    
    ax2.scatter(rouge, semantic, alpha=0.6, s=80, color='#667eea')
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    ax2.set_xlabel('ROUGE-1 F1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Semantic Similarity', fontsize=11, fontweight='bold')
    ax2.set_title('ROUGE vs Semantic Correlation', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 3. Per-category performance
    ax3 = plt.subplot(rows, cols, 3)
    categories = set(r['category'] for r in detailed)
    cat_rouge = []
    cat_semantic = []
    
    for cat in sorted(categories):
        cat_results = [r for r in detailed if r['category'] == cat]
        cat_rouge.append(np.mean([r['metrics']['rouge1_f1'] for r in cat_results]))
        cat_semantic.append(np.mean([r['metrics']['semantic_similarity'] for r in cat_results]))
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax3.bar(x - width/2, cat_rouge, width, label='ROUGE-1', color='#667eea', alpha=0.8)
    ax3.bar(x + width/2, cat_semantic, width, label='Semantic', color='#f093fb', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(sorted(categories), rotation=45, ha='right')
    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title('Performance by Category', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    
    # LLM Judge plots (if available)
    if has_llm:
        # 4. LLM Judge dimensions
        ax4 = plt.subplot(rows, cols, 4)
        llm_dims = ['correctness', 'completeness', 'relevance', 'overall']
        llm_values = [summary.get(f'avg_llm_{d}', 0) for d in llm_dims]
        colors_llm = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#98D8C8']
        
        bars = ax4.bar(llm_dims, llm_values, color=colors_llm, alpha=0.8)
        ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax4.set_title('LLM Judge Dimensions', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 1.0)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, llm_values):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        # 5. ROUGE vs LLM Judge scatter
        ax5 = plt.subplot(rows, cols, 5)
        llm_overall = [r['metrics']['llm_judge']['overall'] for r in detailed 
                      if 'llm_judge' in r['metrics']]
        rouge_llm = [r['metrics']['rouge1_f1'] for r in detailed 
                    if 'llm_judge' in r['metrics']]
        
        ax5.scatter(rouge_llm, llm_overall, alpha=0.6, s=80, color='#667eea')
        ax5.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax5.set_xlabel('ROUGE-1 F1', fontsize=11, fontweight='bold')
        ax5.set_ylabel('LLM Judge Overall', fontsize=11, fontweight='bold')
        ax5.set_title('ROUGE vs LLM Judge', fontsize=12, fontweight='bold')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.grid(alpha=0.3)
        
        # Correlation
        if len(rouge_llm) > 2:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(rouge_llm, llm_overall)
            ax5.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax5.transAxes,
                    fontsize=10, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. All metrics comparison
        ax6 = plt.subplot(rows, cols, 6)
        all_metrics = ['ROUGE-1', 'Semantic', 'LLM Overall']
        all_values = [summary['avg_rouge1_f1'], summary['avg_semantic_similarity'], 
                     summary.get('avg_llm_overall', 0)]
        
        bars = ax6.bar(all_metrics, all_values, color=['#667eea', '#f093fb', '#98D8C8'], alpha=0.8)
        ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax6.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
        ax6.set_ylim(0, 1.0)
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, all_values):
            ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = f"{output_prefix}_generation_charts.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file}")
    plt.close()
    
    if generate_latex:
        generate_generation_latex(data)


def generate_generation_latex(data: dict):
    """Generuje tabelę LaTeX dla generation."""
    summary = data['summary']
    
    print("\n📋 TABELA LATEX - Generation")
    print("="*70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{|l|c|}")
    print(r"\hline")
    print(r"\textbf{Metric} & \textbf{Score} \\")
    print(r"\hline")
    print(f"ROUGE-1 F1 & {summary['avg_rouge1_f1']:.3f} \\\\")
    print(f"Semantic Similarity & {summary['avg_semantic_similarity']:.3f} \\\\")
    print(f"Latency (s) & {summary['avg_latency']:.2f} \\\\")
    
    if 'avg_llm_overall' in summary:
        print(r"\hline")
        print(f"LLM Correctness & {summary.get('avg_llm_correctness', 0):.3f} \\\\")
        print(f"LLM Completeness & {summary.get('avg_llm_completeness', 0):.3f} \\\\")
        print(f"LLM Relevance & {summary.get('avg_llm_relevance', 0):.3f} \\\\")
        print(f"LLM Overall & {summary.get('avg_llm_overall', 0):.3f} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ============================================================================
# MAIN
# ============================================================================

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_stats(data: dict, result_type: str):
    """
    Drukuje statystyki podsumowujące dla wyników.
    Przydatne do sekcji "Wyniki" w pracy inżynierskiej.
    """
    print(f"\n{'='*70}")
    print("📊 STATYSTYKI PODSUMOWUJĄCE")
    print(f"{'='*70}\n")
    
    if result_type == 'experiments':
        # Zbierz wszystkie wyniki
        all_rouge = []
        all_semantic = []
        all_latency = []
        
        for exp_name in ['chunk_size', 'k_values', 'overlap']:
            if exp_name not in data:
                continue
            for r in data[exp_name]['results']:
                all_rouge.append(r['avg_rouge1_f1'])
                all_semantic.append(r['avg_semantic_similarity'])
                all_latency.append(r['avg_latency'])
        
        print(f"Liczba konfiguracji: {len(all_rouge)}")
        
        print(f"\n📈 ROUGE-1 F1:")
        print(f"   Min:      {min(all_rouge):.3f}")
        print(f"   Max:      {max(all_rouge):.3f}")
        print(f"   Średnia:  {np.mean(all_rouge):.3f}")
        print(f"   Mediana:  {np.median(all_rouge):.3f}")
        print(f"   Std:      {np.std(all_rouge):.3f}")
        
        print(f"\n📈 Semantic Similarity:")
        print(f"   Min:      {min(all_semantic):.3f}")
        print(f"   Max:      {max(all_semantic):.3f}")
        print(f"   Średnia:  {np.mean(all_semantic):.3f}")
        print(f"   Mediana:  {np.median(all_semantic):.3f}")
        print(f"   Std:      {np.std(all_semantic):.3f}")
        
        print(f"\n⏱️  Latencja (s):")
        print(f"   Min:      {min(all_latency):.2f}")
        print(f"   Max:      {max(all_latency):.2f}")
        print(f"   Średnia:  {np.mean(all_latency):.2f}")
        
        # Najlepsze konfiguracje
        print(f"\n🏆 NAJLEPSZE KONFIGURACJE:")
        for exp_name in ['chunk_size', 'k_values', 'overlap']:
            if exp_name not in data:
                continue
            best_name = data[exp_name]['best']
            for r in data[exp_name]['results']:
                if r['config']['name'] == best_name:
                    print(f"   {exp_name}: {best_name}")
                    print(f"      ROUGE: {r['avg_rouge1_f1']:.3f}, Semantic: {r['avg_semantic_similarity']:.3f}")
                    break
    
    elif result_type == 'retrieval':
        summary = data['summary']
        print(f"Liczba pytań: {len(data.get('detailed', []))}")
        
        print(f"\n📈 Retrieval Metrics (k=5):")
        print(f"   Precision@5:  {summary['avg_precision@5']:.3f} ± {summary.get('std_precision@5', 0):.3f}")
        print(f"   Recall@5:     {summary['avg_recall@5']:.3f} ± {summary.get('std_recall@5', 0):.3f}")
        print(f"   F1@5:         {summary['avg_f1@5']:.3f} ± {summary.get('std_f1@5', 0):.3f}")
        print(f"   MRR:          {summary['avg_mrr']:.3f} ± {summary.get('std_mrr', 0):.3f}")
        print(f"   NDCG@5:       {summary['avg_ndcg@5']:.3f} ± {summary.get('std_ndcg@5', 0):.3f}")
    
    elif result_type == 'generation':
        summary = data['summary']
        detailed = data.get('detailed', [])
        print(f"Liczba pytań: {len(detailed)}")
        
        print(f"\n📈 Generation Metrics:")
        print(f"   ROUGE-1 F1:          {summary['avg_rouge1_f1']:.3f}")
        print(f"   Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
        print(f"   Latencja:            {summary['avg_latency']:.2f}s")
        
        if 'avg_llm_overall' in summary:
            print(f"\n🤖 LLM Judge Metrics:")
            print(f"   Correctness:   {summary.get('avg_llm_correctness', 0):.3f}")
            print(f"   Completeness:  {summary.get('avg_llm_completeness', 0):.3f}")
            print(f"   Relevance:     {summary.get('avg_llm_relevance', 0):.3f}")
            print(f"   Groundedness:  {summary.get('avg_llm_groundedness', 'N/A')}")
            print(f"   Overall:       {summary.get('avg_llm_overall', 0):.3f}")
        
        # Metryki per kategoria
        categories = set(r['category'] for r in detailed)
        if len(categories) > 1:
            print(f"\n📊 Per Category:")
            for cat in sorted(categories):
                cat_results = [r for r in detailed if r['category'] == cat]
                cat_rouge = np.mean([r['metrics']['rouge1_f1'] for r in cat_results])
                cat_semantic = np.mean([r['metrics']['semantic_similarity'] for r in cat_results])
                print(f"   {cat}: ROUGE={cat_rouge:.3f}, Semantic={cat_semantic:.3f}, n={len(cat_results)}")
    
    print(f"\n{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    VISUALIZATION TOOL                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Użycie:                                                             ║
║      python visualize.py <results.json>                              ║
║      python visualize.py <results.json> --latex                      ║
║                                                                      ║
║  Automatycznie wykrywa typ wyników:                                  ║
║      • experiments_results_*.json  → Wykresy eksperymentów           ║
║      • retrieval_results_*.json    → Wykresy retrieval               ║
║      • generation_results_*.json   → Wykresy generation              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
    
    results_file = sys.argv[1]
    generate_latex = '--latex' in sys.argv
    
    print(f"\n📂 Wczytuję: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result_type = detect_result_type(data)
    print(f"   Wykryty typ: {result_type}")
    
    output_prefix = results_file.replace('.json', '')
    
    if result_type == 'experiments':
        visualize_experiments(data, output_prefix, generate_latex)
        print_summary_stats(data, result_type)
    elif result_type == 'retrieval':
        visualize_retrieval(data, output_prefix, generate_latex)
        print_summary_stats(data, result_type)
    elif result_type == 'generation':
        visualize_generation(data, output_prefix, generate_latex)
        print_summary_stats(data, result_type)
    elif result_type == 'experiments_old':
        print("⚠️  Stary format eksperymentów - użyj nowego run_experiments.py")
    else:
        print("❌ Nieznany format danych!")
        sys.exit(1)
    
    print("\n✅ Wizualizacja zakończona!\n")


if __name__ == "__main__":
    main()