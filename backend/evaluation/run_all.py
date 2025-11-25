"""
Główny orchestrator ewaluacji RAG.
Menu do uruchamiania poszczególnych etapów.

Użycie:
    python run_all.py
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def print_menu():
    """Wyświetla menu."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    RAG EVALUATION SYSTEM                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [1] Przygotuj dataset (Ground Truth + Relevance Annotations)        ║
║      → Generuje: dataset_ready.json                                  ║
║                                                                      ║
║  [2] Ewaluacja RETRIEVAL (Precision, Recall, MRR, NDCG)              ║
║      → Wymaga: dataset_ready.json                                    ║
║                                                                      ║
║  [3] Ewaluacja GENERATION (ROUGE, Semantic Similarity)               ║
║      → Wymaga: dataset_ready.json                                    ║
║                                                                      ║
║  [4] Ewaluacja GENERATION + LLM Judge                                ║
║      → Wymaga: dataset_ready.json (droższe!)                         ║
║                                                                      ║
║  [5] Eksperymenty (chunk_size, k, overlap)                           ║
║      → Wymaga: dataset_ready.json                                    ║
║                                                                      ║
║  [6] PEŁNA EWALUACJA (Retrieval + Generation + LLM Judge)            ║
║      → Uruchamia wszystko po kolei                                   ║
║                                                                      ║
║  [0] Wyjście                                                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def get_paths():
    """Pobiera ścieżki od użytkownika."""
    print("\n📁 Podaj ścieżki:")
    
    pdf_path = input("   PDF path [uploads/Archer_D7UN_V1_UG.pdf]: ").strip()
    if not pdf_path:
        pdf_path = "uploads/Archer_D7UN_V1_UG.pdf"
    
    dataset_path = input("   Dataset path [dataset_ready.json]: ").strip()
    if not dataset_path:
        dataset_path = "dataset_ready.json"
    
    return pdf_path, dataset_path


def run_command(cmd: list):
    """Uruchamia komendę."""
    print(f"\n🚀 Uruchamiam: {' '.join(cmd)}\n")
    print("-" * 70)
    result = subprocess.run(cmd, shell=False)
    print("-" * 70)
    return result.returncode


def main():
    """Główna pętla."""
    
    while True:
        print_menu()
        choice = input("Wybierz opcję [0-6]: ").strip()
        
        if choice == "0":
            print("\n👋 Do widzenia!\n")
            break
        
        elif choice == "1":
            # Przygotuj dataset
            pdf_path = input("\n📁 PDF path [uploads/Archer_D7UN_V1_UG.pdf]: ").strip()
            if not pdf_path:
                pdf_path = "uploads/Archer_D7UN_V1_UG.pdf"
            
            output_path = input("   Output path [dataset_ready.json]: ").strip()
            if not output_path:
                output_path = "dataset_ready.json"
            
            run_command([
                sys.executable, "evaluation/prepare_dataset.py", 
                pdf_path, output_path
            ])
        
        elif choice == "2":
            # Ewaluacja Retrieval
            pdf_path, dataset_path = get_paths()
            run_command([
                sys.executable, "evaluation/evaluate_retrieval.py",
                pdf_path, dataset_path
            ])
        
        elif choice == "3":
            # Ewaluacja Generation (bez LLM Judge)
            pdf_path, dataset_path = get_paths()
            run_command([
                sys.executable, "evaluation/evaluate_generation.py",
                pdf_path, dataset_path
            ])
        
        elif choice == "4":
            # Ewaluacja Generation + LLM Judge
            pdf_path, dataset_path = get_paths()
            run_command([
                sys.executable, "evaluation/evaluate_generation.py",
                pdf_path, dataset_path, "--llm-judge"
            ])
        
        elif choice == "5":
            # Eksperymenty
            pdf_path, dataset_path = get_paths()
            run_command([
                sys.executable, "evaluation/run_experiments.py",
                pdf_path, dataset_path
            ])
        
        elif choice == "6":
            # Pełna ewaluacja
            pdf_path, dataset_path = get_paths()
            
            print("\n" + "="*70)
            print("🔄 PEŁNA EWALUACJA - Uruchamiam wszystkie etapy...")
            print("="*70)
            
            # Sprawdź czy dataset istnieje
            if not os.path.exists(dataset_path):
                print(f"\n⚠️  Dataset nie istnieje: {dataset_path}")
                print("   Najpierw uruchamiam prepare_dataset...\n")
                run_command([
                    sys.executable, "evaluation/prepare_dataset.py",
                    pdf_path, dataset_path
                ])
            
            # 1. Retrieval
            print("\n" + "="*70)
            print("📍 ETAP 1/3: RETRIEVAL EVALUATION")
            print("="*70)
            run_command([
                sys.executable, "evaluation/evaluate_retrieval.py",
                pdf_path, dataset_path
            ])
            
            # 2. Generation + LLM Judge
            print("\n" + "="*70)
            print("📍 ETAP 2/3: GENERATION EVALUATION (+ LLM Judge)")
            print("="*70)
            run_command([
                sys.executable, "evaluation/evaluate_generation.py",
                pdf_path, dataset_path, "--llm-judge"
            ])
            
            # 3. Eksperymenty
            print("\n" + "="*70)
            print("📍 ETAP 3/3: EXPERIMENTS")
            print("="*70)
            run_command([
                sys.executable, "evaluation/run_experiments.py",
                pdf_path, dataset_path
            ])
            
            print("\n" + "="*70)
            print("✅ PEŁNA EWALUACJA ZAKOŃCZONA!")
            print("="*70)
            print("\nWygenerowane pliki:")
            print("   - retrieval_results_*.json")
            print("   - generation_results_with_llm_judge_*.json")
            print("   - experiments_results_*.json")
        
        else:
            print("\n❌ Nieprawidłowa opcja. Wybierz 0-6.\n")
        
        input("\n⏎ Naciśnij Enter aby kontynuować...")


if __name__ == "__main__":
    # Zmień working directory na backend
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    os.chdir(backend_dir)
    
    print(f"\n📂 Working directory: {os.getcwd()}")
    
    main()