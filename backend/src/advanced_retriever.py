"""
Zaawansowany retriever łączący:
- BM25 (keyword search)
- Semantic search (embeddings)
- Reranking (cross-encoder)
"""

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import CrossEncoder
from typing import List
from langchain.schema import Document

class HybridRetriever:
    """
    Retriever łączący keyword search i semantic search z reranking.
    """
    
    def __init__(self, chunks: List[str], embeddings: OpenAIEmbeddings, k: int = 7):
        """
        Args:
            chunks: Lista text chunków
            embeddings: Model embeddingów
            k: Liczba dokumentów do retrieval
        """
        self.k = k
        self.embeddings = embeddings
        
        # Przekonwertuj chunki na Documents
        self.documents = [Document(page_content=chunk) for chunk in chunks]
        
        # 1. BM25 Retriever (keyword)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = k * 2  # Pobierz więcej dla rerankingu
        
        # 2. Semantic Retriever (embeddings)
        self.vectorstore = FAISS.from_documents(self.documents, embeddings)
        self.semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k * 2}
        )
        
        # 3. Ensemble - łączy oba
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.semantic_retriever],
            weights=[0.4, 0.6]  # 40% keyword, 60% semantic
        )
        
        # 4. Reranker (cross-encoder) - opcjonalny
        self.reranker = None
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Reranker załadowany")
        except Exception as e:
            print(f"⚠️  Reranker niedostępny: {e}")
    
    def retrieve(self, query: str, use_rerank: bool = True) -> List[Document]:
        """
        Główna metoda retrieval.
        
        Args:
            query: Pytanie użytkownika
            use_rerank: Czy użyć rerankingu
            
        Returns:
            Lista najlepszych dokumentów
        """
        # 1. Hybrid retrieval
        initial_docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # Deduplikacja
        unique_docs = self._deduplicate(initial_docs)
        
        # 2. Rerank jeśli dostępny
        if use_rerank and self.reranker and len(unique_docs) > 0:
            final_docs = self._rerank(query, unique_docs)
        else:
            final_docs = unique_docs[:self.k]
        
        return final_docs
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Usuń duplikaty z listy dokumentów"""
        seen = set()
        unique = []
        
        for doc in docs:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique.append(doc)
        
        return unique
    
    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank dokumentów używając cross-encoder.
        Cross-encoder jest dokładniejszy niż bi-encoder (embeddings).
        """
        if not docs:
            return []
        
        # Przygotuj pary (query, dokument)
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Oblicz scores
        scores = self.reranker.predict(pairs)
        
        # Sortuj po scores
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Zwróć top K
        reranked_docs = [doc for doc, score in doc_score_pairs[:self.k]]
        
        return reranked_docs


class QueryExpander:
    """
    Rozszerza query o alternatywne phrasings dla lepszego retrieval.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def expand(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Generuje alternatywne wersje pytania.
        
        Args:
            query: Oryginalne pytanie
            num_variants: Liczba wariantów (domyślnie 2)
            
        Returns:
            Lista: [original_query, variant1, variant2, ...]
        """
        prompt = f"""Given this question, generate {num_variants} alternative phrasings that mean the same thing.
Use synonyms and different word orders, but keep the same meaning.

Original question: {query}

Alternative phrasings (one per line, without numbers):"""

        try:
            response = self.llm.predict(prompt)
            
            # Parse odpowiedzi
            variants = [query]  # Zawsze dodaj oryginał
            for line in response.strip().split('\n'):
                line = line.strip()
                # Usuń numery jeśli są (1., 2., etc.)
                line = line.lstrip('0123456789.-) ')
                if line and line != query:
                    variants.append(line)
            
            return variants[:num_variants + 1]  # original + variants
            
        except Exception as e:
            print(f"⚠️  Query expansion failed: {e}")
            return [query]  # Zwróć tylko oryginał