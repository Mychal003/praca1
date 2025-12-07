"""
Skrypt do ponownego uruchomienia pojedynczego pytania (Q19)
i wyświetlenia wartości do wpisania w JSON.

Użycie:
    python fix_single_question.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline
from collections import Counter
from sentence_transformers import SentenceTransformer, util

print("🤖 Ładuję modele...")
SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')

# ============================================================================
# METRYKI
# ============================================================================

def rouge_1_f1(prediction: str, reference: str) -> float:
    pred_words = Counter(prediction.lower().split())
    ref_words = Counter(reference.lower().split())
    if not pred_words or not ref_words:
        return 0.0
    overlap = sum((pred_words & ref_words).values())
    if sum(pred_words.values()) == 0 or sum(ref_words.values()) == 0:
        return 0.0
    precision = overlap / sum(pred_words.values())
    recall = overlap / sum(ref_words.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def semantic_similarity(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    emb1 = SEMANTIC_MODEL.encode(prediction, convert_to_tensor=True)
    emb2 = SEMANTIC_MODEL.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return max(0.0, min(1.0, similarity))

def token_overlap(prediction: str, reference: str) -> float:
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0.0
    return len(pred_words & ref_words) / len(ref_words)

# ============================================================================
# LLM JUDGE
# ============================================================================

class LLMJudge:
    def __init__(self, model: str = "gpt-4o"):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def _get_score(self, prompt: str) -> float:
        try:
            # Dodaj delay żeby uniknąć rate limit
            time.sleep(1)
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"   ⚠️ Błąd: {e}")
            return 0.5
    
    def evaluate(self, question: str, generated: str, reference: str, context: str = None):
        print("   Oceniam correctness...")
        correctness = self._get_score(f"""Rate the CORRECTNESS of the generated answer (0.0-1.0).
Question: {question}
Reference: {reference}
Generated: {generated}
1.0 = completely correct, 0.0 = completely wrong
Respond with ONLY a number:""")
        
        print("   Oceniam completeness...")
        completeness = self._get_score(f"""Rate the COMPLETENESS of the generated answer (0.0-1.0).
Question: {question}
Reference: {reference}
Generated: {generated}
1.0 = contains all key information, 0.0 = missing all
Respond with ONLY a number:""")
        
        print("   Oceniam relevance...")
        relevance = self._get_score(f"""Rate the RELEVANCE of the answer (0.0-1.0).
Question: {question}
Generated: {generated}
1.0 = directly answers, 0.0 = irrelevant
Respond with ONLY a number:""")
        
        groundedness = None
        if context:
            print("   Oceniam groundedness...")
            groundedness = self._get_score(f"""Rate if the answer is GROUNDED in context (0.0-1.0).
Question: {question}
Context: {context[:10000]}
Generated: {generated}
1.0 = all facts from context, 0.0 = hallucinations
Respond with ONLY a number:""")
        
        print("   Oceniam overall...")
        overall = self._get_score(f"""Rate the OVERALL QUALITY (0.0-1.0).
Question: {question}
Reference: {reference}
Generated: {generated}
Consider: accuracy, completeness, clarity, usefulness.
Respond with ONLY a number:""")
        
        valid = [v for v in [correctness, completeness, relevance, groundedness, overall] if v is not None]
        average = sum(valid) / len(valid) if valid else 0.0
        
        return {
            'correctness': correctness,
            'completeness': completeness,
            'relevance': relevance,
            'groundedness': groundedness,
            'overall': overall,
            'average': average
        }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Pytanie 19 (index 18)
    question = "How do you recover access if you forgot the router's login password?"
    expected = "Web Management page password:\nRestore the modem router to its factory default settings and then set a new password using 1-15 characters."
    
    print(f"\n📝 Pytanie: {question[:60]}...")
    print(f"📝 Expected: {expected[:60]}...")
    
    # Stwórz pipeline z konfiguracją 1200/10/0
    print("\n🔧 Tworzenie pipeline (1200/10/0)...")
    pipeline = RAGPipeline(
        chunk_size=1200,
        chunk_overlap=0,
        k=10
    )
    pipeline.process_document("uploads/Archer_D7UN_V1_UG.pdf")
    print(f"   ✓ Utworzono {pipeline.num_chunks} chunków")
    
    # Generuj odpowiedź
    print("\n🤖 Generuję odpowiedź...")
    start = time.time()
    generated = pipeline.query(question)
    latency = time.time() - start
    print(f"   ✓ Wygenerowano w {latency:.2f}s")
    print(f"   Generated: {generated[:100]}...")
    
    # Oblicz metryki
    print("\n📊 Obliczam metryki...")
    rouge = rouge_1_f1(generated, expected)
    semantic = semantic_similarity(generated, expected)
    overlap = token_overlap(generated, expected)
    
    print(f"   ROUGE-1 F1: {rouge:.6f}")
    print(f"   Semantic:   {semantic:.6f}")
    print(f"   Overlap:    {overlap:.6f}")
    print(f"   Latency:    {latency:.6f}")
    
    # LLM Judge
    print("\n🧑‍⚖️ Uruchamiam LLM Judge...")
    judge = LLMJudge()
    
    # Pobierz kontekst
    sources = pipeline.get_sources(question, k=10)
    context = "\n\n".join([s['text'] for s in sources])
    
    llm_scores = judge.evaluate(question, generated, expected, context)
    
    # Wyświetl wyniki do skopiowania
    print("\n" + "="*70)
    print("📋 SKOPIUJ PONIŻSZE WARTOŚCI DO JSON (index 18):")
    print("="*70)
    print(f"""
    {{
      "question": "{question}",
      "expected": "Web Management page password:\\nRestore the modem router to its factory default settings and then set a new password using 1-15 characters.",
      "generated": "{generated.replace('"', '\\"').replace(chr(10), '\\n')}",
      "category": "procedural",
      "metrics": {{
        "rouge1_f1": {rouge},
        "semantic_similarity": {semantic},
        "token_overlap": {overlap},
        "latency": {latency},
        "llm_judge": {{
          "correctness": {llm_scores['correctness']},
          "completeness": {llm_scores['completeness']},
          "relevance": {llm_scores['relevance']},
          "groundedness": {llm_scores['groundedness']},
          "overall": {llm_scores['overall']},
          "average": {llm_scores['average']}
        }}
      }}
    }}
""")
    print("="*70)
    print("\n✅ Gotowe! Skopiuj powyższy blok i zamień w pliku JSON (element o indeksie 18).")