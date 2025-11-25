# 📊 System Ewaluacji RAG - Instrukcja Użycia

## 📁 Struktura plików

```
backend/evaluation/
├── prepare_dataset.py      # Przygotowanie datasetu (GT + annotacje)
├── evaluate_retrieval.py   # Metryki retrieval (P@k, R@k, MRR, NDCG)
├── evaluate_generation.py  # Metryki generation (ROUGE, Semantic, LLM Judge)
├── run_experiments.py      # Eksperymenty (chunk_size, k, overlap)
├── run_all.py              # Menu interaktywne
├── visualize.py            # Wizualizacja + tabele LaTeX
└── metrics/
    ├── __init__.py
    ├── retrieval_metrics.py
    └── llm_judge.py
```

---

## 🚀 Szybki start

### Opcja 1: Menu interaktywne (najłatwiejsza)

```bash
cd backend
python evaluation/run_all.py
```

Wybierz opcję z menu (0-6).

### Opcja 2: Komendy ręczne (więcej kontroli)

Patrz sekcje poniżej.

---

## 📋 Krok po kroku

### 1️⃣ Przygotowanie datasetu (WYMAGANE NA POCZĄTEK)

```bash
cd backend
python evaluation/prepare_dataset.py uploads/Archer_D7UN_V1_UG.pdf
```

**Co robi:**
- Ekstrahuje ground truth answers z dokumentu (GPT-4o)
- Annotuje relevant chunks dla każdego pytania (GPT-4o)
- Kategoryzuje pytania (factual/procedural/troubleshooting)

**Output:** `dataset_ready.json`

**Czas:** ~5-10 minut (25 pytań × 2 wywołania LLM)

**Koszt:** ~$0.50-1.00 (GPT-4o)

---

### 2️⃣ Ewaluacja Retrieval

```bash
python evaluation/evaluate_retrieval.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json
```

**Co robi:**
- Mierzy jakość retrievalu (czy system znajduje właściwe chunki)
- Oblicza: Precision@k, Recall@k, F1@k, MRR, NDCG@k

**Output:** `retrieval_results_YYYYMMDD_HHMMSS.json`

**Czas:** ~1-2 minuty

**Koszt:** ~$0.10 (tylko embeddingi)

---

### 3️⃣ Ewaluacja Generation (bez LLM Judge)

```bash
python evaluation/evaluate_generation.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json
```

**Co robi:**
- Generuje odpowiedzi na wszystkie pytania
- Oblicza: ROUGE-1 F1, Semantic Similarity, Token Overlap
- Raportuje metryki per kategoria

**Output:** `generation_results_YYYYMMDD_HHMMSS.json`

**Czas:** ~3-5 minut

**Koszt:** ~$0.50 (GPT-4o generation)

---

### 4️⃣ Ewaluacja Generation + LLM Judge (pełna)

```bash
python evaluation/evaluate_generation.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json --llm-judge
```

**Co robi:**
- Wszystko z punktu 3
- Dodatkowo: LLM Judge ocenia każdą odpowiedź w 5 wymiarach
- Analiza korelacji między metrykami

**Output:** `generation_results_with_llm_judge_YYYYMMDD_HHMMSS.json`

**Czas:** ~10-15 minut

**Koszt:** ~$2-3 (GPT-4o generation + 5× judge per pytanie)

---

### 5️⃣ Eksperymenty (chunk_size, k, overlap)

```bash
python evaluation/run_experiments.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json
```

**Co robi:**
- Testuje 15 konfiguracji:
  - chunk_size: 300, 500, 800, 1200, 1500
  - k: 1, 3, 5, 7, 10
  - overlap: 0, 50, 100, 200, 300
- Znajduje najlepszą konfigurację per eksperyment
- Analiza błędów (najtrudniejsze/najłatwiejsze pytania)

**Output:** `experiments_results_YYYYMMDD_HHMMSS.json`

**Czas:** ~30-45 minut

**Koszt:** ~$5-8 (15 konfiguracji × 25 pytań)

---

### 6️⃣ Wizualizacja wyników

```bash
# Auto-wykrywa typ wyników
python evaluation/visualize.py experiments_results_*.json

# Z tabelami LaTeX
python evaluation/visualize.py retrieval_results_*.json --latex

# Generation
python evaluation/visualize.py generation_results_with_llm_judge_*.json --latex
```

**Co robi:**
- Generuje wykresy PNG (dpi=300)
- Opcjonalnie: tabele LaTeX do pracy
- Statystyki podsumowujące (Min/Max/Mean/Median/Std)

**Output:** `*_charts.png`

---

## 🎯 Pełna ewaluacja (wszystko naraz)

### Przez menu:

```bash
python evaluation/run_all.py
# Wybierz opcję [6] Pełna ewaluacja
```

### Lub ręcznie:

```bash
cd backend

# 1. Przygotuj dataset (raz)
python evaluation/prepare_dataset.py uploads/Archer_D7UN_V1_UG.pdf

# 2. Retrieval
python evaluation/evaluate_retrieval.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json

# 3. Generation + LLM Judge
python evaluation/evaluate_generation.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json --llm-judge

# 4. Eksperymenty
python evaluation/run_experiments.py uploads/Archer_D7UN_V1_UG.pdf dataset_ready.json

# 5. Wizualizacja
python evaluation/visualize.py retrieval_results_*.json --latex
python evaluation/visualize.py generation_results_with_llm_judge_*.json --latex
python evaluation/visualize.py experiments_results_*.json --latex
```

---

## 📊 Generowane pliki

| Etap | Plik | Zawartość |
|------|------|-----------|
| Dataset | `dataset_ready.json` | Pytania, GT, relevant chunks, kategorie |
| Retrieval | `retrieval_results_*.json` | P@k, R@k, F1@k, MRR, NDCG |
| Generation | `generation_results_*.json` | ROUGE, Semantic, per-category |
| Gen + Judge | `generation_results_with_llm_judge_*.json` | + LLM scores, korelacje |
| Experiments | `experiments_results_*.json` | Wszystkie konfiguracje |
| Wykresy | `*_charts.png` | Wizualizacje (300 dpi) |

---

## 💰 Szacunkowe koszty (GPT-4o)

| Etap | Czas | Koszt |
|------|------|-------|
| prepare_dataset | 5-10 min | $0.50-1.00 |
| evaluate_retrieval | 1-2 min | $0.10 |
| evaluate_generation | 3-5 min | $0.50 |
| evaluate_generation --llm-judge | 10-15 min | $2-3 |
| run_experiments | 30-45 min | $5-8 |
| **RAZEM (pełna ewaluacja)** | **~1h** | **~$8-12** |

---

## ⚠️ Ważne uwagi

### 1. Baseline config
Retrieval metrics działają **tylko dla baseline config** (chunk_size=800, overlap=100).
Różne chunk_size = różne chunk_ids = nieporównywalne retrieval metrics.

### 2. Dataset jest reużywalny
Po wygenerowaniu `dataset_ready.json` możesz go używać wielokrotnie.
Nie trzeba go generować za każdym razem.

### 3. Working directory
Wszystkie komendy zakładają że jesteś w `backend/`.

### 4. Zmienne środowiskowe
Upewnij się że masz `.env` z `OPENAI_API_KEY`.

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
cd backend  # Upewnij się że jesteś w backend/
```

### "FileNotFoundError: dataset_ready.json"
```bash
python evaluation/prepare_dataset.py uploads/Archer_D7UN_V1_UG.pdf
```

### "Dataset nie ma annotacji 'relevant_chunk_indices'"
Użyj nowego datasetu wygenerowanego przez `prepare_dataset.py`, nie starego.

### "Za mało wyników z LLM Judge do analizy korelacji"
Użyj flagi `--llm-judge` przy `evaluate_generation.py`.

---

## 📈 Metryki - co znaczą?

### Retrieval Metrics
| Metryka | Co mierzy | Interpretacja |
|---------|-----------|---------------|
| **Precision@k** | % relevant w top-k | Czy nie zwracamy śmieci? |
| **Recall@k** | % znalezionych relevant | Czy znajdujemy wszystko? |
| **MRR** | Pozycja pierwszego relevant | Czy relevant jest wysoko? |
| **NDCG@k** | Jakość rankingu | Czy ranking jest dobry? |

### Generation Metrics
| Metryka | Co mierzy | Interpretacja |
|---------|-----------|---------------|
| **ROUGE-1 F1** | Overlap słów | Pokrycie leksykalne |
| **Semantic Similarity** | Podobieństwo znaczeniowe | Czy znaczenie się zgadza? |
| **LLM Judge** | Ocena eksperta | Rzeczywista jakość |

### LLM Judge Dimensions
| Wymiar | Co ocenia |
|--------|-----------|
| **Correctness** | Czy fakty są poprawne? |
| **Completeness** | Czy nic nie brakuje? |
| **Relevance** | Czy odpowiada na pytanie? |
| **Groundedness** | Czy oparte na kontekście? |
| **Overall** | Ogólna jakość |

---



### Sekcja "Metodologia"
- Opis metryk (retrieval + generation)
- Opis LLM Judge
- Baseline config i dlaczego

### Sekcja "Eksperymenty"
- Wpływ chunk_size na jakość
- Wpływ k na jakość
- Wpływ overlap na jakość
- Trade-off jakość vs latencja

### Sekcja "Wyniki"
- Tabele LaTeX (generowane przez `--latex`)
- Wykresy PNG
- Analiza korelacji metryk
- Najtrudniejsze/najłatwiejsze pytania

---

## 🎉 Gotowe!

Po wykonaniu wszystkich kroków masz:
- ✅ Kompletny dataset z ground truth
- ✅ Metryki retrieval
- ✅ Metryki generation + LLM Judge
- ✅ Eksperymenty z różnymi konfiguracjami
- ✅ Wykresy do pracy
- ✅ Tabele LaTeX