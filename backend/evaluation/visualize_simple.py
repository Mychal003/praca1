"""
Prosty skrypt do tworzenia wykresów z wyników ewaluacji.
"""
import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import sys


def create_charts(results_file: str):
    """
    Tworzy 2 podstawowe wykresy z wyników ewaluacji.
    
    Args:
        results_file: Ścieżka do pliku JSON z wynikami
    """
    # Wczytaj wyniki
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Przygotuj dane
    configs = [r['config']['name'] for r in results]
    rouge_scores = [r['summary']['avg_rouge1_f1'] for r in results]
    latencies = [r['summary']['avg_latency'] for r in results]
    
    # WYKRES 1: Porównanie ROUGE-1 F1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres słupkowy
    bars = ax1.barh(configs, rouge_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_xlabel('ROUGE-1 F1 Score', fontsize=12)
    ax1.set_ylabel('Konfiguracja', fontsize=12)
    ax1.set_title('Porównanie jakości odpowiedzi', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    
    # Dodaj wartości na słupkach
    for i, (bar, score) in enumerate(zip(bars, rouge_scores)):
        ax1.text(score + 0.02, i, f'{score:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # WYKRES 2: Jakość vs Performance
    ax2.scatter(latencies, rouge_scores, s=300, alpha=0.6, 
               c=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # Dodaj etykiety
    for i, name in enumerate(configs):
        ax2.annotate(name, (latencies[i], rouge_scores[i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Latencja (sekundy)', fontsize=12)
    ax2.set_ylabel('ROUGE-1 F1 Score', fontsize=12)
    ax2.set_title('Trade-off: Jakość vs Performance', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Zapisz
    output_file = results_file.replace('.json', '_charts.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Wykres zapisany: {output_file}")
    
    plt.show()


def create_simple_table(results_file: str):
    """
    Tworzy prostą tabelę tekstową do wklejenia w pracę.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("TABELA DO PRACY INŻYNIERSKIEJ")
    print("="*70)
    print(f"{'Konfiguracja':<20} {'chunk_size':<12} {'ROUGE-1 F1':<12} {'Token Overlap':<15} {'Latencja (s)'}")
    print("-"*70)
    
    for r in results:
        name = r['config']['name']
        chunk = r['config']['chunk_size']
        rouge = r['summary']['avg_rouge1_f1']
        overlap = r['summary']['avg_token_overlap']
        latency = r['summary']['avg_latency']
        
        print(f"{name:<20} {chunk:<12} {rouge:<12.3f} {overlap:<15.3f} {latency:.2f}")
    
    print("="*70 + "\n")
    
    # Znajdź najlepszy
    best = max(results, key=lambda x: x['summary']['avg_rouge1_f1'])
    print(f"🏆 Najlepsza konfiguracja: {best['config']['name']}")
    print(f"   • chunk_size: {best['config']['chunk_size']}")
    print(f"   • ROUGE-1 F1: {best['summary']['avg_rouge1_f1']:.3f}")
    print(f"   • Latencja: {best['summary']['avg_latency']:.2f}s\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python visualize_simple.py <results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print("📊 Tworzę wykresy...")
    create_charts(results_file)
    
    print("\n📋 Generuję tabelę...")
    create_simple_table(results_file)
    
    print("\n✅ Gotowe!")
