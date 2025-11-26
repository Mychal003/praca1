"""
Wizualizacja retrieval metrics dla pracy inżynierskiej.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import sys

def create_retrieval_charts(results_file: str):
    """
    Tworzy kompleksowe wykresy retrieval metrics.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data['summary']
    base_output = results_file.replace('.json', '')
    
    # =========================================================================
    # WYKRES 1: Precision@k vs Recall@k
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    k_values = [1, 3, 5, 10]
    precision_values = [summary[f'avg_precision@{k}'] for k in k_values]
    recall_values = [summary[f'avg_recall@{k}'] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, precision_values, width, label='Precision@k', color='#667eea', alpha=0.8)
    bars2 = ax1.bar(x + width/2, recall_values, width, label='Recall@k', color='#f093fb', alpha=0.8)
    
    ax1.set_xlabel('k (Top-k documents)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Precision vs Recall at Different k', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Dodaj wartości na słupkach
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file1 = f"{base_output}_precision_recall.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file1}")
    plt.close()
    
    # =========================================================================
    # WYKRES 2: Wszystkie metryki retrieval
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    metrics = ['precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5']
    metric_labels = ['P@5', 'R@5', 'F1@5', 'MRR', 'NDCG@5']
    values = [summary[f'avg_{m}'] for m in metrics]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax2.bar(range(len(metrics)), values, color=colors, alpha=0.8)
    
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Retrieval Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # Dodaj wartości
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 0.02,
                f'{value:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file2 = f"{base_output}_retrieval_metrics.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file2}")
    plt.close()
    
    # =========================================================================
    # WYKRES 3: NDCG@k dla różnych k
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ndcg_values = [summary[f'avg_ndcg@{k}'] for k in k_values]
    
    ax3.plot(k_values, ndcg_values, marker='o', linewidth=2, markersize=10, 
             color='#667eea', label='NDCG@k')
    ax3.fill_between(k_values, ndcg_values, alpha=0.3, color='#667eea')
    
    ax3.set_xlabel('k (Top-k documents)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('NDCG Score', fontsize=12, fontweight='bold')
    ax3.set_title('NDCG@k: Ranking Quality', fontsize=14, fontweight='bold')
    ax3.set_xticks(k_values)
    ax3.set_xticklabels([f'k={k}' for k in k_values])
    ax3.set_ylim(0, 1.0)
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    # Dodaj wartości
    for k, ndcg in zip(k_values, ndcg_values):
        ax3.annotate(f'{ndcg:.3f}', (k, ndcg), 
                    textcoords="offset points", xytext=(0,8), 
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file3 = f"{base_output}_ndcg.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"   ✅ Zapisano: {output_file3}")
    plt.close()

def create_comparison_table(results_file: str):
    """
    Tworzy tabelę LaTeX do pracy.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data['summary']
    
    print("\n" + "="*70)
    print("📋 TABELA LATEX - Wyniki Ewaluacji")
    print("="*70 + "\n")
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Wyniki ewaluacji systemu RAG}")
    print(r"\label{tab:rag_evaluation}")
    print(r"\begin{tabular}{|l|c|c|}")
    print(r"\hline")
    print(r"\textbf{Kategoria} & \textbf{Metryka} & \textbf{Wynik} \\")
    print(r"\hline")
    print(r"\multirow{3}{*}{Generation} & ROUGE-1 F1 & " + f"{summary['avg_rouge1_f1']:.3f}" + r" \\")
    print(r"                              & Semantic Similarity & " + f"{summary['avg_semantic_similarity']:.3f}" + r" \\")
    print(r"                              & Latencja (s) & " + f"{summary['avg_latency']:.2f}" + r" \\")
    print(r"\hline")
    print(r"\multirow{5}{*}{Retrieval}   & Precision@5 & " + f"{summary['avg_precision@5']:.3f}" + r" \\")
    print(r"                              & Recall@5 & " + f"{summary['avg_recall@5']:.3f}" + r" \\")
    print(r"                              & F1@5 & " + f"{summary['avg_f1@5']:.3f}" + r" \\")
    print(r"                              & MRR & " + f"{summary['avg_mrr']:.3f}" + r" \\")
    print(r"                              & NDCG@5 & " + f"{summary['avg_ndcg@5']:.3f}" + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python evaluation/visualize_retrieval.py <full_evaluation_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print("📊 Tworzę wykresy retrieval metrics...")
    create_retrieval_charts(results_file)
    
    print("\n📋 Generuję tabelę LaTeX...")
    create_comparison_table(results_file)
    
    print("\n✅ Gotowe!")