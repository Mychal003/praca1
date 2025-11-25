"""
LLM as Judge - ewaluacja jakości odpowiedzi RAG przy użyciu LLM.
Bardziej zaawansowana alternatywa dla metryk leksykalnych (ROUGE).
"""

from langchain_openai import ChatOpenAI
from typing import Dict, List
import json
from dotenv import load_dotenv

load_dotenv()


class LLMJudge:
    """
    Używa LLM (GPT-4) do oceny jakości odpowiedzi RAG.
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        """
        Args:
            model: Model do użycia jako judge (gpt-4o-mini jest tańszy od gpt-4)
            temperature: 0 dla deterministycznych ocen
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
    
    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = None
    ) -> Dict[str, float]:
        """
        Kompleksowa ocena wygenerowanej odpowiedzi.
        
        Args:
            question: Pytanie użytkownika
            generated_answer: Odpowiedź wygenerowana przez RAG
            reference_answer: Ground truth answer
            context: Opcjonalnie - kontekst użyty do generacji
            
        Returns:
            Dict z ocenami dla każdego kryterium (0-1)
        """
        
        # Ocena 1: Correctness (poprawność merytoryczna)
        correctness = self._evaluate_correctness(
            question, generated_answer, reference_answer
        )
        
        # Ocena 2: Completeness (kompletność)
        completeness = self._evaluate_completeness(
            question, generated_answer, reference_answer
        )
        
        # Ocena 3: Relevance (relevancja do pytania)
        relevance = self._evaluate_relevance(
            question, generated_answer
        )
        
        # Ocena 4: Groundedness (czy oparte na kontekście, nie halucynuje)
        groundedness = None
        if context:
            groundedness = self._evaluate_groundedness(
                question,
                generated_answer, 
                context
            )
        
        # Ocena 5: Overall quality (ogólna jakość)
        overall = self._evaluate_overall(
            question, generated_answer, reference_answer
        )
        
        return {
            "correctness": correctness,
            "completeness": completeness,
            "relevance": relevance,
            "groundedness": groundedness,
            "overall": overall,
            "average": self._compute_average([
                correctness, completeness, relevance, 
                groundedness if groundedness else 0, overall
            ])
        }
    
    def _evaluate_correctness(
        self, 
        question: str, 
        generated: str, 
        reference: str
    ) -> float:
        """
        Ocena poprawności merytorycznej odpowiedzi.
        """
        prompt = f"""You are an expert evaluator assessing the correctness of an answer.

Question: {question}

Reference Answer (Ground Truth):
{reference}

Generated Answer:
{generated}

TASK: Rate the CORRECTNESS of the generated answer compared to the reference.
- 1.0 = Completely correct, all facts match the reference
- 0.7-0.9 = Mostly correct, minor inaccuracies
- 0.4-0.6 = Partially correct, some significant errors
- 0.0-0.3 = Mostly incorrect or contradicts reference

Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.85)
No explanation, just the score."""

        try:
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            print(f"⚠️  Error in correctness evaluation: {e}")
            return 0.5  # Neutral fallback
    
    def _evaluate_completeness(
        self, 
        question: str, 
        generated: str, 
        reference: str
    ) -> float:
        """
        Ocena kompletności - czy zawiera wszystkie kluczowe informacje?
        """
        prompt = f"""You are an expert evaluator assessing answer completeness.

Question: {question}

Reference Answer (what a complete answer should include):
{reference}

Generated Answer:
{generated}

TASK: Rate the COMPLETENESS of the generated answer.
- 1.0 = Contains all key information from reference
- 0.7-0.9 = Contains most key information, minor omissions
- 0.4-0.6 = Missing some important information
- 0.0-0.3 = Missing most key information

Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.75)
No explanation, just the score."""

        try:
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  Error in completeness evaluation: {e}")
            return 0.5
    
    def _evaluate_relevance(self, question: str, generated: str) -> float:
        """
        Ocena relevancji - czy odpowiedź faktycznie odpowiada na pytanie?
        """
        prompt = f"""You are an expert evaluator assessing answer relevance.

Question: {question}

Generated Answer:
{generated}

TASK: Rate how RELEVANT the answer is to the question.
- 1.0 = Directly answers the question, no off-topic content
- 0.7-0.9 = Mostly relevant, minor tangents
- 0.4-0.6 = Partially relevant, some off-topic content
- 0.0-0.3 = Mostly irrelevant or doesn't answer the question

Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.90)
No explanation, just the score."""

        try:
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  Error in relevance evaluation: {e}")
            return 0.5
    
    def _evaluate_groundedness(
        self, 
        question: str,
        generated: str, 
        context: str
    ) -> float:
        """
        Ocena groundedness - czy odpowiedź jest oparta na kontekście?
        """
        prompt = f"""You are an expert evaluator checking if an answer is grounded in the provided context.

ORIGINAL QUESTION:
{question}

CONTEXT (information available to the system):
{context}

GENERATED ANSWER:
{generated}

TASK: Rate how GROUNDED the answer is in the provided context.

INSTRUCTIONS:
1. First, check if the CONTEXT contains information to answer the ORIGINAL QUESTION
2. Then evaluate the GENERATED ANSWER:

If the answer provides factual information:
- 1.0 = All facts are supported by the context
- 0.7-0.9 = Mostly grounded, minor unsupported details
- 0.4-0.6 = Some facts not supported by context
- 0.0-0.3 = Contains significant hallucinations

If the answer says "I don't have this information" or similar:
- 1.0 = Context truly lacks the information to answer the question (honest response)
- 0.0 = Context DOES contain information to answer the question (system failed to use it)

IMPORTANT: Consider both EXPLICIT and IMPLICIT information in the context.
For example, if the question asks "What is the default password?" and context says 
"You must set a password on first login", this IMPLICITLY answers "no default password exists".

Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.85)
No explanation, just the score."""

        try:
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  Error in groundedness evaluation: {e}")
            return 0.5
    
    def _evaluate_overall(
        self, 
        question: str, 
        generated: str, 
        reference: str
    ) -> float:
        """
        Ogólna ocena jakości odpowiedzi.
        """
        prompt = f"""You are an expert evaluator assessing overall answer quality.

Question: {question}

Reference Answer:
{reference}

Generated Answer:
{generated}

TASK: Rate the OVERALL QUALITY of the generated answer.
Consider: accuracy, completeness, clarity, and usefulness.
- 1.0 = Excellent answer, matches or exceeds reference
- 0.7-0.9 = Good answer, minor issues
- 0.4-0.6 = Acceptable but has notable problems
- 0.0-0.3 = Poor answer, significant issues

Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.80)
No explanation, just the score."""

        try:
            response = self.llm.predict(prompt).strip()
            score = float(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  Error in overall evaluation: {e}")
            return 0.5
    
    def _compute_average(self, scores: List[float]) -> float:
        """Oblicz średnią, ignorując None"""
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Test LLM Judge
    judge = LLMJudge(model="gpt-4o-mini")
    
    question = "What is the default web address to access the router interface?"
    reference = "The default web address is http://tplinkmodem.net or 192.168.1.1."
    generated = "You can access the router at http://tplinkmodem.net."
    
    print("\n🧪 Testing LLM Judge...\n")
    print(f"Question: {question}")
    print(f"\nReference: {reference}")
    print(f"\nGenerated: {generated}")
    
    scores = judge.evaluate_answer(question, generated, reference)
    
    print("\n📊 Scores:")
    for metric, score in scores.items():
        if score is not None:
            print(f"  {metric.capitalize():<15} {score:.3f}")
    
    print("\n✅ LLM Judge working correctly!")