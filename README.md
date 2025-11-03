# RAG Documentation Assistant

> Retrieval-Augmented Generation system for answering questions about technical documentation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-orange)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-purple)](https://openai.com/)

## 📋 Spis treści

- [O projekcie](#o-projekcie)
- [Architektura](#architektura)
- [Funkcjonalności](#funkcjonalności)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Ewaluacja](#ewaluacja)
- [Struktura projektu](#struktura-projektu)
- [Technologie](#technologie)
- [Wyniki](#wyniki)
- [Autor](#autor)

---

## 🎯 O projekcie

System RAG (Retrieval-Augmented Generation) umożliwiający użytkownikom zadawanie pytań w języku naturalnym o zawartość dokumentacji technicznej w formacie PDF. System automatycznie:

1. **Przetwarza dokumenty PDF** - ekstrahuje tekst i dzieli na semantyczne fragmenty
2. **Indeksuje wiedzę** - tworzy wektorową bazę danych z embeddingami
3. **Retrieval** - wyszukuje najbardziej relevantne fragmenty do pytania
4. **Generuje odpowiedzi** - wykorzystuje LLM do syntezy odpowiedzi na podstawie znalezionych fragmentów

**Grupa docelowa:** 
- Klienci firm potrzebujący wsparcia technicznego
- Pracownicy działu wsparcia technicznego
- Użytkownicy urządzeń technicznych

---

## 🏗️ Architektura

```
┌─────────────┐
│   Frontend  │  (HTML/CSS/JS)
│  Interface  │
└──────┬──────┘
       │
       │ HTTP API
       ▼
┌─────────────────────────────────────────┐
│          Flask Backend                  │
├─────────────────────────────────────────┤
│  ┌──────────────────────────────────┐   │
│  │      RAG Pipeline                │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  1. PDF Processor          │  │   │
│  │  │     (PyMuPDF)              │  │   │
│  │  └────────────────────────────┘  │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  2. Text Chunking          │  │   │
│  │  │     (RecursiveTextSplitter)│  │   │
│  │  └────────────────────────────┘  │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  3. Embeddings             │  │   │
│  │  │     (OpenAI text-embed-3)  │  │   │
│  │  └────────────────────────────┘  │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  4. Vector Store (FAISS)   │  │   │
│  │  └────────────────────────────┘  │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  5. Query Classifier       │  │   │
│  │  │     (GPT-3.5 zero-shot)    │  │   │
│  │  └────────────────────────────┘  │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  6. LLM Generator          │  │   │
│  │  │     (GPT-3.5-turbo)        │  │   │
│  │  └────────────────────────────┘  │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Przepływ danych:

1. **Upload dokumentu** → PDF → Ekstrakcja tekstu → Chunking → Embeddings → Vector Store
2. **Zadanie pytania** → Klasyfikacja → Retrieval → Generacja odpowiedzi → Zwrócenie wyniku

---

## ✨ Funkcjonalności

### Core Features:
- ✅ **Upload PDF** - wgrywanie dokumentacji technicznej
- ✅ **Semantic Search** - wyszukiwanie semantyczne z FAISS
- ✅ **Query Classification** - klasyfikacja pytań (factual/procedural/troubleshooting)
- ✅ **Context-aware Answers** - odpowiedzi uwzględniające kontekst
- ✅ **Source Attribution** - wskazanie źródeł informacji w dokumencie

### Advanced Features:
- 🔬 **Evaluation System** - automatyczna ewaluacja z metrykami ROUGE-1, Token Overlap
- 📊 **Experimentation Framework** - testowanie różnych konfiguracji
- 📈 **Visualization Tools** - generowanie wykresów i tabel
- 🎯 **Custom Prompts** - dostosowane prompty dla różnych typów pytań

---

## 📦 Wymagania

- **Python:** 3.9 lub nowszy
- **Node.js:** (opcjonalnie, dla development frontendu)
- **OpenAI API Key:** wymagany do działania systemu

### Zależności Python:
```
flask>=3.0
flask-cors>=4.0
langchain>=0.1.0
langchain-openai>=0.0.5
openai>=1.0.0
faiss-cpu>=1.7.4
pymupdf>=1.23.0
python-dotenv>=1.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## 🚀 Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/your-username/rag-doc-assistant.git
cd rag-doc-assistant
```

### 2. Utworzenie środowiska wirtualnego

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 4. Konfiguracja klucza API

Utwórz plik `.env` w katalogu `backend/`:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

⚠️ **Ważne:** Dodaj `.env` do `.gitignore` (już dodane w projekcie)

---

## 💻 Użycie

### Uruchomienie backendu

```bash
cd backend
python app.py
```

Backend wystartuje na `http://localhost:5000`

### Uruchomienie frontendu

Otwórz `frontend/index.html` w przeglądarce lub użyj Live Server:

```bash
cd frontend
# Jeśli masz Python:
python -m http.server 8000
# Następnie otwórz: http://localhost:8000
```

### Korzystanie z aplikacji

1. **Wgraj dokument PDF** - kliknij "Wybierz plik" i wybierz PDF z dokumentacją
2. **Poczekaj na przetworzenie** - system podzieli dokument i utworzy indeks wektorowy
3. **Zadawaj pytania** - wpisz pytanie w języku naturalnym
4. **Otrzymaj odpowiedź** - system zwróci odpowiedź wraz ze źródłami

---

## 🧪 Ewaluacja

### Podstawowa ewaluacja (3 konfiguracje)

```bash
cd backend
python evaluation/evaluate_simple.py uploads/your_document.pdf
```

Wyniki zostaną zapisane w pliku `evaluation_results_TIMESTAMP.json`

### Wizualizacja wyników

```bash
python evaluation/visualize_simple.py evaluation_results_TIMESTAMP.json
```

Generuje:
- Wykres PNG z porównaniem konfiguracji
- Tabelę tekstową gotową do wklejenia w pracę

### Rozszerzone eksperymenty (14 konfiguracji)

```bash
python evaluation/advanced_experiments.py uploads/your_document.pdf
```

Testuje:
- **Chunk size:** 300, 500, 800, 1200, 1500
- **K (liczba dokumentów):** 1, 2, 3, 5, 7
- **Overlap:** 0, 50, 100, 200

⏱️ **Czas wykonania:** ~45-60 minut

### Metryki ewaluacji

- **ROUGE-1 F1:** Miara pokrycia słów między wygenerowaną a oczekiwaną odpowiedzią
- **Token Overlap:** Procent wspólnych tokenów
- **Latency:** Czas odpowiedzi w sekundach

---

## 📁 Struktura projektu

```
rag-doc-assistant/
│
├── backend/
│   ├── app.py                      # Flask backend
│   ├── .env                        # Klucz API (NIE commituj!)
│   │
│   ├── src/
│   │   ├── pdf_processor.py        # Ekstrakcja tekstu z PDF
│   │   ├── query_classifier.py     # Klasyfikacja pytań
│   │   ├── rag_pipeline.py         # Główny pipeline RAG
│   │   └── vector_store.py         # Zarządzanie FAISS
│   │
│   ├── evaluation/
│   │   ├── evaluate_simple.py      # Podstawowa ewaluacja
│   │   ├── visualize_simple.py     # Wizualizacja wyników
│   │   ├── advanced_experiments.py # Rozszerzone eksperymenty
│   │   └── visualize_advanced.py   # Zaawansowane wykresy
│   │
│   └── uploads/                    # Folder na PDF-y
│
├── frontend/
│   ├── index.html                  # Interfejs użytkownika
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
├── requirements.txt                # Zależności Python
├── .gitignore
└── README.md
```

---

## 🛠️ Technologie

### Backend:
- **Flask** - framework webowy
- **LangChain** - framework do budowy aplikacji LLM
- **OpenAI API** - embeddings (text-embedding-3-small) i generacja (GPT-3.5-turbo)
- **FAISS** - wektorowa baza danych (Facebook AI Similarity Search)
- **PyMuPDF** - ekstrakcja tekstu z PDF

### Frontend:
- **HTML5/CSS3/JavaScript** - interfejs użytkownika
- **Fetch API** - komunikacja z backendem

### Evaluation:
- **Matplotlib** - generowanie wykresów
- **NumPy** - obliczenia numeryczne
- **Custom metrics** - implementacja ROUGE-1 F1

---

## 📊 Wyniki

### Najlepsza konfiguracja

Na podstawie eksperymentów z dokumentacją TP-Link Archer D7:

| Parametr | Wartość optymalna |
|----------|-------------------|
| **chunk_size** | 500-800 |
| **chunk_overlap** | 100 |
| **k (liczba dokumentów)** | 3 |
| **ROUGE-1 F1** | ~0.36 |
| **Średnia latencja** | ~2.5s |

### Wnioski:

1. **Chunk size:** Mniejsze chunki (500) dają lepszą jakość, ale większe (800) są szybsze
2. **Overlap:** Moderate overlap (100) zapewnia najlepszy balans
3. **K:** 3 dokumenty to sweet spot - więcej nie poprawia jakości znacząco

**Trade-off:** Jakość vs Wydajność
- Małe chunki (500): wyższa jakość (+2-3% ROUGE), wolniejsze (2x)
- Średnie chunki (800): dobry kompromis
- Duże chunki (1200): szybsze, ale niższa jakość

---

## 👤 Autor

**Paweł [Nazwisko]**
- GitHub: [@your-username](https://github.com/your-username)
- Praca inżynierska: Politechnika Wrocławska, 2025

---

## 📄 Licencja

Ten projekt został stworzony na potrzeby pracy inżynierskiej.

---

---

## 📚 Dokumentacja dodatkowa

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**⭐ Jeśli ten projekt był pomocny, zostaw gwiazdkę na GitHub!**