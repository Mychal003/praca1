# 🎯 Kompletny Przewodnik Ewaluacji Systemu RAG

Dokument opisuje krok po kroku jak uruchomić pełną ewaluację systemu RAG 
z generation i retrieval metrics.

---

## 📋 Spis Treści

1. [Wymagania Wstępne](#wymagania-wstępne)
2. [Struktura Plików](#struktura-plików)
3. [Krok 1: Przygotowanie Ground Truth Dataset](#krok-1-przygotowanie-ground-truth-dataset)
4. [Krok 2: Annotacja Relevant Chunks](#krok-2-annotacja-relevant-chunks)
5. [Krok 3: Uruchomienie Pełnej Ewaluacji](#krok-3-uruchomienie-pełnej-ewaluacji)
6. [Krok 4: Wizualizacja Wyników](#krok-4-wizualizacja-wyników)
7. [Krok 5: Dodatkowe Analizy](#krok-5-dodatkowe-analizy)
8. [Interpretacja Wyników](#interpretacja-wyników)
9. [Troubleshooting](#troubleshooting)

---

## 🔧 Wymagania Wstępne

### Instalacja Zależności
```bash
# Aktywuj virtual environment
source venv/bin/activate  # Linux/Mac
# LUB
venv\Scripts\activate     # Windows

# Zainstaluj wymagane pakiety
pip install -r requirements.txt
```

### Wymagane Pakiety
```txt
langchain
langchain-openai
langchain-community
openai
faiss-cpu
pymupdf
python-dotenv
sentence-transformers
matplotlib
numpy
```

### Konfiguracja API Key

Upewnij się, że masz plik `.env` w folderze `backend/`:
```env
OPENAI_API_KEY=sk-your-api-key-here
```

---

## 📁 Struktura Plików
```
backend/
├── evaluation/
│   ├── evaluate_simple.py                 # Główny system ewaluacji
│   ├── extract_ground_truth.py            # Generator ground truth
│   ├── annotate_relevant_chunks.py        # Annotator relevance
│   ├── retrieval_metrics.py               # Metryki retrieval
│   ├── run_full_evaluation.py             # Skrypt ewaluacji
│   ├── visualize_retrieval.py             # Wizualizacje
│   ├── advanced_experiments.py            # Eksperymenty zaawansowane
│   └── compare_datasets.py                # Porównanie datasetów
├── src/
│   ├── rag_pipeline.py                    # Pipeline RAG
│   ├── pdf_processor.py                   # Processor PDF
│   ├── query_classifier.py                # Klasyfikator pytań
│   └── vector_store.py                    # Vector store
├── uploads/
│   └── Archer_D7UN_V1_UG.pdf             # Dokument testowy
├── test_dataset_ground_truth.json         # Ground truth (generowany)
├── test_dataset_ground_truth_with_relevance.json  # Z annotacjami (generowany)
├── full_evaluation_results.json           # Wyniki ewaluacji (generowany)
└── full_evaluation_results_retrieval_charts.png  # Wykresy (generowane)
```

---

## 🚀 Krok 1: Przygotowanie Ground Truth Dataset

### Co to jest Ground Truth?

Ground truth dataset to zbiór pytań z **oczekiwanymi odpowiedziami wyekstrahowanymi 
bezpośrednio z dokumentu** przez GPT-4o-mini. To zapewnia, że odpowiedzi używają 
tych samych słów co dokument, co daje lepsze wyniki ROUGE.

### Uruchomienie
```bash
cd backend

python evaluation/extract_ground_truth.py uploads/Archer_D7UN_V1_UG.pdf
```

### Co się dzieje?

1. System ładuje dokument PDF
2. Dla każdego z 25 pytań:
   - Znajduje relevant chunki dokumentu
   - Używa GPT-4o-mini do ekstrakcji **dokładnej odpowiedzi** z tekstu
3. Zapisuje do `test_dataset_ground_truth.json`

### Oczekiwany Output
```
═══════════════════════════════════════════════════════════════
         GROUND TRUTH EXTRACTION                             
═══════════════════════════════════════════════════════════════

1️⃣  Loading document...
⚙️ Rozpoczynam przetwarzanie dokumentu...
✅ Dokument przetworzony pomyślnie w 2.34s

2️⃣  Extracting ground truth answers...

[1/25] Extracting: What is the full model name and type of this router?...
   ✓ Extracted: The router is TP-Link Archer D7, an AC1750 Wireless...

[2/25] Extracting: What is the default web address to access...
   ✓ Extracted: The default web address is http://tplinkmodem.net...

...

✅ Ground truth dataset saved: test_dataset_ground_truth.json
   Total questions: 25
```

### Czas Wykonania

- **~3-5 minut** (w zależności od API)
- Koszt: ~$0.05-0.10 (GPT-4o-mini)

### Wygenerowany Plik

`test_dataset_ground_truth.json`:
```json
[
  {
    "question": "What is the full model name and type of this router?",
    "expected_answer": "The router is TP-Link Archer D7, an AC1750 Wireless Dual Band Gigabit ADSL2+ Modem Router.",
    "source_chunks": ["...", "...", "..."],
    "category": "factual"
  },
  ...
]
```

---

## 🏷️ Krok 2: Annotacja Relevant Chunks

### Co to jest Annotacja?

Annotacja to proces **dodania informacji o tym, które chunki dokumentu są relevant** 
dla każdego pytania. To jest kluczowe dla obliczania **retrieval metrics** 
(Precision@k, Recall@k, MRR, NDCG).

### Uruchomienie
```bash
python evaluation/annotate_relevant_chunks.py \
    uploads/Archer_D7UN_V1_UG.pdf \
    test_dataset_ground_truth.json
```

**Windows PowerShell (jedna linia):**
```powershell
python evaluation/annotate_relevant_chunks.py uploads/Archer_D7UN_V1_UG.pdf test_dataset_ground_truth.json
```

### Co się dzieje?

1. System ładuje dokument i ground truth dataset
2. Dla każdego pytania:
   - Retrieves top-20 chunków
   - GPT-4o-mini ocenia czy każdy chunk jest relevant dla pytania
   - Zapisuje listę relevant chunk indices
3. Zapisuje do `test_dataset_ground_truth_with_relevance.json`

### Oczekiwany Output
```
======================================================================
🏷️  ANNOTACJA RELEVANT CHUNKS
======================================================================

[1/25] What is the full model name and type of this router?...
   ✓ Chunk 0 is relevant
   ✓ Chunk 1 is relevant
   ✓ Chunk 3 is relevant
   ✓ Chunk 6 is relevant
   → Found 4 relevant chunks

[2/25] What is the default web address to access the router...
   ✓ Chunk 1 is relevant
   ✓ Chunk 2 is relevant
   → Found 2 relevant chunks

...

✅ Annotated dataset saved: test_dataset_ground_truth_with_relevance.json

======================================================================
📊 STATYSTYKI ANNOTACJI
======================================================================
Pytania: 25
Średnia liczba relevant chunks na pytanie: 5.84
======================================================================
```

### Czas Wykonania

- **~10-15 minut** (w zależności od API)
- Koszt: ~$0.50-1.00 (GPT-4o-mini sprawdza 25 × 20 = 500 par)

### Wygenerowany Plik

`test_dataset_ground_truth_with_relevance.json`:
```json
[
  {
    "question": "What is the full model name and type of this router?",
    "expected_answer": "The router is TP-Link Archer D7...",
    "category": "factual",
    "relevant_chunk_indices": [0, 1, 3, 6, 9, 10, 19],
    "total_chunks_evaluated": 20
  },
  ...
]
```

---

## 📊 Krok 3: Uruchomienie Pełnej Ewaluacji

### Uruchomienie
```bash
python evaluation/run_full_evaluation.py
```

### Co się dzieje?

System przechodzi przez wszystkie 25 pytań i dla każdego:

1. **Klasyfikuje pytanie** (factual/procedural/troubleshooting)
2. **Retrieves dokumenty** z vector store
3. **Generuje odpowiedź** używając LLM
4. **Oblicza Generation Metrics:**
   - ROUGE-1 F1
   - Token Overlap
   - Semantic Similarity
   - Latencja
5. **Oblicza Retrieval Metrics:**
   - Precision@k (k=1,3,5,10)
   - Recall@k
   - F1@k
   - MRR (Mean Reciprocal Rank)
   - NDCG@k (Normalized Discounted Cumulative Gain)
   - Average Precision

### Oczekiwany Output
```
======================================================================
🚀 PEŁNA EWALUACJA RAG Z RETRIEVAL METRICS
======================================================================

📂 Loading annotated dataset: test_dataset_ground_truth_with_relevance.json
✅ Loaded 25 questions with relevance annotations

🔧 Initializing RAG pipeline...
✅ Pipeline ready!

============================================================
🚀 EWALUACJA - 25 pytań
   (Including RETRIEVAL METRICS)
============================================================

[1/25] What is the full model name and type of this route...
  ✓ ROUGE: 0.880 | Semantic: 0.821 | P@5: 0.600 | R@5: 0.429 | 5.72s

[2/25] What is the default web address to access the rout...
  ✓ ROUGE: 0.545 | Semantic: 0.881 | P@5: 0.400 | R@5: 0.167 | 4.12s

...

[25/25] How can you remotely access USB storage connected...
  ✓ ROUGE: 0.218 | Semantic: 0.547 | P@5: 0.800 | R@5: 0.250 | 10.02s

============================================================
📊 PODSUMOWANIE
============================================================

🎯 GENERATION METRICS:
  ROUGE-1 F1:          0.414
  Token Overlap:       0.468
  Semantic Similarity: 0.756
  Latencja:            6.70s

🔍 RETRIEVAL METRICS:
  precision@1          0.800
  precision@5          0.568
  recall@5             0.532
  f1@5                 0.509
  mrr                  0.847
  ndcg@5               0.695
============================================================

======================================================================
✅ EWALUACJA ZAKOŃCZONA!
======================================================================

💾 Wyniki zapisane: full_evaluation_results.json
```

### Czas Wykonania

- **~3-5 minut** (25 pytań × ~7s = ~175s)
- Koszt: ~$0.15-0.25 (GPT-4 mini dla generacji)

### Wygenerowany Plik

`full_evaluation_results.json`:
```json
{
  "summary": {
    "avg_rouge1_f1": 0.414,
    "avg_semantic_similarity": 0.756,
    "avg_latency": 6.70,
    "avg_precision@5": 0.568,
    "avg_recall@5": 0.532,
    "avg_mrr": 0.847,
    "avg_ndcg@5": 0.695
  },
  "detailed_results": [
    {
      "question": "...",
      "expected": "...",
      "generated": "...",
      "rouge1_f1": 0.880,
      "semantic_similarity": 0.821,
      "latency": 5.72,
      "retrieval_metrics": {
        "precision@5": 0.600,
        "recall@5": 0.429,
        ...
      }
    },
    ...
  ]
}
```

---

## 📈 Krok 4: Wizualizacja Wyników

### Uruchomienie
```bash
python evaluation/visualize_retrieval.py full_evaluation_results.json
```

### Co się generuje?

1. **Wykresy (PNG):**
   - Precision@k vs Recall@k dla różnych k
   - Porównanie wszystkich retrieval metrics
   - NDCG@k dla różnych k

2. **Tabela LaTeX:**
   - Gotowa do wklejenia w pracę inżynierską
   - Zawiera wszystkie kluczowe metryki

### Oczekiwany Output
```
📊 Tworzę wykresy retrieval metrics...
✅ Wykresy zapisane: full_evaluation_results_retrieval_charts.png

📋 Generuję tabelę LaTeX...
======================================================================
📋 TABELA LATEX - Wyniki Ewaluacji
======================================================================

\begin{table}[h]
\centering
\caption{Wyniki ewaluacji systemu RAG}
\label{tab:rag_evaluation}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Kategoria} & \textbf{Metryka} & \textbf{Wynik} \\
\hline
\multirow{3}{*}{Generation} & ROUGE-1 F1 & 0.414 \\
                              & Semantic Similarity & 0.756 \\
                              & Latencja (s) & 6.70 \\
\hline
\multirow{5}{*}{Retrieval}   & Precision@5 & 0.568 \\
                              & Recall@5 & 0.532 \\
                              & F1@5 & 0.509 \\
                              & MRR & 0.847 \\
                              & NDCG@5 & 0.695 \\
\hline
\end{tabular}
\end{table}

✅ Gotowe!
```

### Wygenerowane Pliki

- `full_evaluation_results_retrieval_charts.png` - wykresy (300 DPI)

---

## 🔬 Krok 5: Dodatkowe Analizy

### 5.1 Porównanie różnych konfiguracji
```bash
# Testuje chunk_size, k, overlap
python evaluation/advanced_experiments.py uploads/Archer_D7UN_V1_UG.pdf
```

**Czas wykonania:** ~20-30 minut  
**Generuje:** `advanced_results_TIMESTAMP.json`

### 5.2 Wizualizacja zaawansowana
```bash
python evaluation/visualize_advanced.py advanced_results_TIMESTAMP.json
```

**Generuje:**
- 6 wykresów porównawczych
- Tabele LaTeX dla każdego eksperymentu
- Statystyki podsumowujące

### 5.3 Porównanie ROUGE vs Semantic Similarity
```bash
python evaluation/visualize_rouge_vs_semantic.py full_evaluation_results.json
```

**Generuje:**
- Scatter plot ROUGE vs Semantic
- Histogram różnic
- Statystyki porównania

---

## 📖 Interpretacja Wyników

### Generation Metrics

| Metryka | Zakres | Dobry Wynik | Twój Wynik |
|---------|--------|-------------|------------|
| **ROUGE-1 F1** | 0-1 | >0.5 | 0.414 |
| **Semantic Similarity** | 0-1 | >0.7 | 0.756 ✅ |
| **Latencja** | - | <10s | 6.70s ✅ |

**Interpretacja:**
- Wysoka Semantic Similarity (0.756) przy średnim ROUGE (0.414) oznacza, 
  że system generuje **merytorycznie poprawne odpowiedzi**, ale używa 
  innych słów niż dokument
- To jest **normalny i pożądany** efekt dla systemów generatywnych

### Retrieval Metrics

| Metryka | Zakres | Dobry Wynik | Twój Wynik |
|---------|--------|-------------|------------|
| **Precision@5** | 0-1 | >0.5 | 0.568 ✅ |
| **Recall@5** | 0-1 | >0.5 | 0.532 ✅ |
| **MRR** | 0-1 | >0.8 | 0.847 ✅ |
| **NDCG@5** | 0-1 | >0.65 | 0.695 ✅ |

**Interpretacja:**

#### MRR = 0.847 (Świetny!)
- Pierwszy relevant dokument jest średnio na pozycji **1.18** (1/0.847)
- Retriever **bardzo dobrze** identyfikuje najważniejsze dokumenty

#### Precision@5 = 0.568
- Z 5 retrievanych dokumentów, średnio **2.84 jest relevant**
- To dobry wynik (>0.5)

#### Recall@5 = 0.532
- System znajduje **53% wszystkich relevant dokumentów** w top-5
- Można poprawić przez zwiększenie k lub hybrid retrieval

#### NDCG@5 = 0.695
- Relevant dokumenty są dobrze **uszeregowane** (wyżej = bardziej relevant)

---

