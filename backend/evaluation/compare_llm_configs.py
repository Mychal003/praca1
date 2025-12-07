"""
Skrypt do porównania wyników LLM Judge dla różnych konfiguracji.
Generuje tabele LaTeX do pracy inżynierskiej.

Użycie:
    python compare_llm_configs.py <plik1.json> <plik2.json> [plik3.json ...]
    
Przykład:
    python compare_llm_configs.py generation_results_with_llm_judge_20251204_023303.json generation_results_1200_10_0_with_llm_judge_20251205_222026.json
"""

import json
import sys
from pathlib import Path


def load_results(filepath):
    """Wczytuje wyniki z pliku JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_config_name(data, filepath):
    """Wyciąga nazwę konfiguracji z danych lub nazwy pliku."""
    config = data.get('config', {})
    chunk_size = config.get('chunk_size', '?')
    k = config.get('pipeline_k', '?')
    overlap = config.get('chunk_overlap', '?')
    return f"{chunk_size}/{k}/{overlap}"


def compare_configs(files):
    """Porównuje konfiguracje i generuje raport."""
    
    configs = []
    
    for filepath in files:
        print(f"📂 Wczytuję: {filepath}")
        data = load_results(filepath)
        config_name = extract_config_name(data, filepath)
        
        configs.append({
            'name': config_name,
            'file': filepath,
            'summary': data['summary'],
            'config': data.get('config', {})
        })
        print(f"   ✓ Konfiguracja: {config_name}")
    
    print()
    
    # =========================================================================
    # TABELA 1: Porównanie głównych metryk
    # =========================================================================
    
    print("=" * 80)
    print("📊 PORÓWNANIE KONFIGURACJI")
    print("=" * 80)
    
    # Nagłówki
    header = f"{'Metryka':<25}"
    for c in configs:
        header += f" {c['name']:^15}"
    print(header)
    print("-" * (25 + 16 * len(configs)))
    
    # Metryki
    metrics = [
        ('ROUGE-1 F1', 'avg_rouge1_f1'),
        ('Semantic Similarity', 'avg_semantic_similarity'),
        ('Latencja (s)', 'avg_latency'),
        ('', None),  # separator
        ('LLM Correctness', 'avg_llm_correctness'),
        ('LLM Completeness', 'avg_llm_completeness'),
        ('LLM Relevance', 'avg_llm_relevance'),
        ('LLM Groundedness', 'avg_llm_groundedness'),
        ('LLM Overall', 'avg_llm_overall'),
    ]
    
    for label, key in metrics:
        if key is None:
            print()
            continue
        
        row = f"{label:<25}"
        values = []
        for c in configs:
            val = c['summary'].get(key, 0)
            values.append(val)
            row += f" {val:^15.3f}"
        
        # Zaznacz najlepszą wartość
        if values and key != 'avg_latency':
            best_idx = values.index(max(values))
        elif key == 'avg_latency':
            best_idx = values.index(min(values))
        else:
            best_idx = -1
        
        print(row)
    
    print("=" * 80)
    
    # =========================================================================
    # TABELA 2: Per kategoria
    # =========================================================================
    
    print("\n📊 METRYKI PER KATEGORIA")
    print("-" * 80)
    
    categories = ['factual', 'procedural', 'troubleshooting']
    
    for cat in categories:
        print(f"\n{cat.upper()}:")
        
        row_rouge = f"  {'ROUGE-1':<20}"
        row_semantic = f"  {'Semantic':<20}"
        
        for c in configs:
            rouge = c['summary'].get(f'{cat}_rouge1_f1', 0)
            semantic = c['summary'].get(f'{cat}_semantic', 0)
            row_rouge += f" {rouge:^15.3f}"
            row_semantic += f" {semantic:^15.3f}"
        
        print(row_rouge)
        print(row_semantic)
    
    # =========================================================================
    # LATEX: Tabela główna
    # =========================================================================
    
    print("\n")
    print("=" * 80)
    print("📋 TABELA LATEX - Porównanie konfiguracji LLM Judge")
    print("=" * 80)
    print()
    
    # Nagłówek tabeli
    col_spec = "|l|" + "c|" * len(configs)
    header_row = "\\textbf{Metryka}"
    for c in configs:
        header_row += f" & \\textbf{{{c['name']}}}"
    header_row += " \\\\"
    
    print(f"""\\begin{{table}}[H]
\\centering
\\caption{{Porównanie wyników LLM Judge dla różnych konfiguracji}}
\\label{{tab:llm_judge_comparison}}
\\begin{{tabular}}{{{col_spec}}}
\\hline
{header_row}
\\hline""")
    
    # Wiersze z metrykami
    latex_metrics = [
        ('ROUGE-1 F1', 'avg_rouge1_f1'),
        ('Semantic Similarity', 'avg_semantic_similarity'),
        ('Latencja (s)', 'avg_latency'),
    ]
    
    for label, key in latex_metrics:
        row = label
        values = [c['summary'].get(key, 0) for c in configs]
        
        # Znajdź najlepszą wartość
        if key == 'avg_latency':
            best_val = min(values)
        else:
            best_val = max(values)
        
        for val in values:
            if val == best_val:
                row += f" & \\textbf{{{val:.3f}}}"
            else:
                row += f" & {val:.3f}"
        row += " \\\\"
        print(row)
    
    print("\\hline")
    
    # LLM Judge metryki
    llm_metrics = [
        ('LLM Correctness', 'avg_llm_correctness'),
        ('LLM Completeness', 'avg_llm_completeness'),
        ('LLM Relevance', 'avg_llm_relevance'),
        ('LLM Groundedness', 'avg_llm_groundedness'),
        ('LLM Overall', 'avg_llm_overall'),
    ]
    
    for label, key in llm_metrics:
        row = label
        values = [c['summary'].get(key, 0) for c in configs]
        best_val = max(values)
        
        for val in values:
            if val == best_val:
                row += f" & \\textbf{{{val:.3f}}}"
            else:
                row += f" & {val:.3f}"
        row += " \\\\"
        print(row)
    
    print("""\\hline
\\end{tabular}
\\end{table}""")
    
    # =========================================================================
    # LATEX: Tabela per kategoria
    # =========================================================================
    
    print("\n")
    print("=" * 80)
    print("📋 TABELA LATEX - Wyniki per kategoria")
    print("=" * 80)
    print()
    
    # Dla każdej konfiguracji osobna mini-tabela lub jedna duża
    print(f"""\\begin{{table}}[H]
\\centering
\\caption{{Porównanie metryk według kategorii pytań}}
\\label{{tab:category_comparison}}
\\begin{{tabular}}{{|l|l|{"c|" * len(configs)}}}
\\hline
\\textbf{{Kategoria}} & \\textbf{{Metryka}} & """ + " & ".join([f"\\textbf{{{c['name']}}}" for c in configs]) + """ \\\\
\\hline""")
    
    for cat in categories:
        cat_label = cat.capitalize()
        
        # ROUGE row
        row = f"{cat_label} & ROUGE-1"
        values = [c['summary'].get(f'{cat}_rouge1_f1', 0) for c in configs]
        best_val = max(values)
        for val in values:
            if val == best_val:
                row += f" & \\textbf{{{val:.3f}}}"
            else:
                row += f" & {val:.3f}"
        row += " \\\\"
        print(row)
        
        # Semantic row
        row = f" & Semantic"
        values = [c['summary'].get(f'{cat}_semantic', 0) for c in configs]
        best_val = max(values)
        for val in values:
            if val == best_val:
                row += f" & \\textbf{{{val:.3f}}}"
            else:
                row += f" & {val:.3f}"
        row += " \\\\"
        print(row)
        
        print("\\hline")
    
    print("""\\end{tabular}
\\end{table}""")
    
    # =========================================================================
    # WNIOSKI
    # =========================================================================
    
    print("\n")
    print("=" * 80)
    print("💡 WSTĘPNE WNIOSKI")
    print("=" * 80)
    
    # Znajdź najlepszą konfigurację dla LLM Overall
    overall_scores = [(c['name'], c['summary'].get('avg_llm_overall', 0)) for c in configs]
    best_config = max(overall_scores, key=lambda x: x[1])
    
    print(f"\n✅ Najlepsza konfiguracja według LLM Overall: {best_config[0]} ({best_config[1]:.3f})")
    
    # Porównanie szczegółowe
    if len(configs) == 2:
        c1, c2 = configs[0], configs[1]
        
        print(f"\n📊 Porównanie {c1['name']} vs {c2['name']}:")
        
        diff_metrics = [
            ('ROUGE-1 F1', 'avg_rouge1_f1'),
            ('Semantic', 'avg_semantic_similarity'),
            ('LLM Overall', 'avg_llm_overall'),
            ('LLM Groundedness', 'avg_llm_groundedness'),
        ]
        
        for label, key in diff_metrics:
            v1 = c1['summary'].get(key, 0)
            v2 = c2['summary'].get(key, 0)
            diff = v1 - v2
            winner = c1['name'] if diff > 0 else c2['name']
            print(f"   {label:<20}: {v1:.3f} vs {v2:.3f} (diff: {diff:+.3f}) → {winner}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║           COMPARE LLM JUDGE CONFIGURATIONS                           ║
╚══════════════════════════════════════════════════════════════════════╝

Użycie:
    python compare_llm_configs.py <plik1.json> <plik2.json> [plik3.json ...]

Przykład:
    python compare_llm_configs.py generation_results_with_llm_judge_20251204_023303.json generation_results_1200_10_0_with_llm_judge_20251205_222026.json
        """)
        sys.exit(1)
    
    files = sys.argv[1:]
    compare_configs(files)