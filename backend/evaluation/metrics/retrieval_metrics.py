"""
Retrieval metrics dla ewaluacji RAG systems.
Implementuje: Precision@k, Recall@k, F1@k, MRR, NDCG.
"""

from typing import List, Dict, Set
import numpy as np


def precision_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """
    Precision@k: Jaki % z top-k retrievanych dokumentów jest relevant?
    
    Formula: P@k = (# relevant docs in top-k) / k
    
    Example:
        Retrieved top-5: [0, 2, 5, 7, 9]
        Relevant: [0, 2, 3, 7, 10]
        P@5 = 3/5 = 0.6 (bo 0, 2, 7 są relevant)
    
    Args:
        retrieved_indices: Indices of retrieved chunks (ordered by relevance)
        relevant_indices: Indices of ground-truth relevant chunks
        k: Cutoff (top-k)
    
    Returns:
        Precision@k score (0-1)
    """
    if k == 0:
        return 0.0
    
    # Weź tylko top-k
    retrieved_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    # Ile z top-k jest relevant?
    num_relevant_retrieved = len(retrieved_k & relevant_set)
    
    return num_relevant_retrieved / k


def recall_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """
    Recall@k: Jaki % wszystkich relevant dokumentów znalazłem w top-k?
    
    Formula: R@k = (# relevant docs in top-k) / (total # relevant docs)
    
    Example:
        Retrieved top-5: [0, 2, 5, 7, 9]
        Relevant: [0, 2, 3, 7, 10]  (5 relevant total)
        R@5 = 3/5 = 0.6 (znalazłem 3 z 5 relevant)
    
    Args:
        retrieved_indices: Indices of retrieved chunks
        relevant_indices: Indices of ground-truth relevant chunks
        k: Cutoff
    
    Returns:
        Recall@k score (0-1)
    """
    if not relevant_indices:
        return 0.0
    
    retrieved_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    num_relevant_retrieved = len(retrieved_k & relevant_set)
    
    return num_relevant_retrieved / len(relevant_set)


def f1_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """
    F1@k: Harmonic mean of Precision@k and Recall@k.
    
    Formula: F1@k = 2 * (P@k * R@k) / (P@k + R@k)
    
    Args:
        retrieved_indices: Indices of retrieved chunks
        relevant_indices: Indices of ground-truth relevant chunks
        k: Cutoff
    
    Returns:
        F1@k score (0-1)
    """
    precision = precision_at_k(retrieved_indices, relevant_indices, k)
    recall = recall_at_k(retrieved_indices, relevant_indices, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(retrieved_indices: List[int], relevant_indices: List[int]) -> float:
    """
    MRR: Mean Reciprocal Rank
    
    Mierzy na jakiej pozycji znajduje się PIERWSZY relevant dokument.
    
    Formula: MRR = 1 / (rank of first relevant doc)
    
    Example:
        Retrieved: [5, 2, 7, 0, 9]  
        Relevant: [0, 2]
        First relevant (2) is at position 2 (1-indexed)
        MRR = 1/2 = 0.5
    
    High MRR (close to 1) = relevant docs are ranked high
    Low MRR (close to 0) = relevant docs are ranked low
    
    Args:
        retrieved_indices: Indices of retrieved chunks (ordered)
        relevant_indices: Indices of ground-truth relevant chunks
    
    Returns:
        MRR score (0-1)
    """
    relevant_set = set(relevant_indices)
    
    # Znajdź pozycję pierwszego relevant dokumentu
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    
    # Nie znaleziono żadnego relevant dokumentu
    return 0.0


def ndcg_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """
    NDCG@k: Normalized Discounted Cumulative Gain
    
    Bierze pod uwagę zarówno relevance JAK I pozycję w rankingu.
    Dokumenty wyżej w rankingu mają większą wagę.
    
    DCG@k = Σ (rel_i / log2(i+1)) for i in 1..k
    NDCG@k = DCG@k / IDCG@k (normalized by ideal DCG)
    
    Example:
        Retrieved: [0, 5, 2, 7] (top-4)
        Relevant: [0, 2, 7]
        Binary relevance: [1, 0, 1, 1]
        DCG = 1/log2(2) + 0 + 1/log2(4) + 1/log2(5) = 1.9
        IDCG (ideal): [1, 1, 1, 0]
        IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) + 0 = 2.13
        NDCG = 1.9/2.13 = 0.89
    
    Args:
        retrieved_indices: Indices of retrieved chunks
        relevant_indices: Indices of ground-truth relevant chunks
        k: Cutoff
    
    Returns:
        NDCG@k score (0-1)
    """
    relevant_set = set(relevant_indices)
    
    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k], start=1):
        relevance = 1.0 if idx in relevant_set else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # IDCG: Ideal DCG (gdyby wszystkie relevant docs były na początku)
    ideal_relevance = [1.0] * min(len(relevant_set), k)
    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevance, start=1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def average_precision(retrieved_indices: List[int], relevant_indices: List[int]) -> float:
    """
    AP: Average Precision
    
    Średnia precision po każdym relevant dokumencie.
    Uwzględnia zarówno recall jak i pozycję.
    
    Formula: AP = (Σ P(k) * rel(k)) / (# relevant docs)
    gdzie P(k) = precision at k, rel(k) = 1 if doc at k is relevant
    
    Example:
        Retrieved: [2, 5, 0, 7, 9]
        Relevant: [0, 2, 7]
        
        At pos 1: doc 2 is relevant, P(1) = 1/1 = 1.0
        At pos 3: doc 0 is relevant, P(3) = 2/3 = 0.67
        At pos 4: doc 7 is relevant, P(4) = 3/4 = 0.75
        
        AP = (1.0 + 0.67 + 0.75) / 3 = 0.81
    
    Args:
        retrieved_indices: Indices of retrieved chunks
        relevant_indices: Indices of ground-truth relevant chunks
    
    Returns:
        Average Precision score (0-1)
    """
    if not relevant_indices:
        return 0.0
    
    relevant_set = set(relevant_indices)
    
    num_relevant_so_far = 0
    sum_precisions = 0.0
    
    for k, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_set:
            num_relevant_so_far += 1
            precision_at_k = num_relevant_so_far / k
            sum_precisions += precision_at_k
    
    return sum_precisions / len(relevant_set)


def evaluate_retrieval(
    retrieved_indices: List[int],
    relevant_indices: List[int],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation z wszystkimi metrykami.
    
    Args:
        retrieved_indices: Indices of retrieved chunks (ranked)
        relevant_indices: Indices of ground-truth relevant chunks
        k_values: List of k values to evaluate
    
    Returns:
        Dict z wszystkimi metrykami
    """
    results = {}
    
    # Precision, Recall, F1 dla różnych k
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(retrieved_indices, relevant_indices, k)
        results[f'recall@{k}'] = recall_at_k(retrieved_indices, relevant_indices, k)
        results[f'f1@{k}'] = f1_at_k(retrieved_indices, relevant_indices, k)
        results[f'ndcg@{k}'] = ndcg_at_k(retrieved_indices, relevant_indices, k)
    
    # MRR i AP (nie zależą od k)
    results['mrr'] = mean_reciprocal_rank(retrieved_indices, relevant_indices)
    results['average_precision'] = average_precision(retrieved_indices, relevant_indices)
    
    return results


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║            RETRIEVAL METRICS - EXAMPLES                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Example 1: Perfect retrieval
    print("\n📊 EXAMPLE 1: Perfect Retrieval")
    print("-" * 60)
    retrieved = [0, 2, 5, 7, 9]
    relevant = [0, 2, 5, 7, 9]
    
    metrics = evaluate_retrieval(retrieved, relevant, k_values=[3, 5])
    for metric, value in metrics.items():
        print(f"{metric:<25} {value:.3f}")
    
    # Example 2: Poor retrieval
    print("\n📊 EXAMPLE 2: Poor Retrieval")
    print("-" * 60)
    retrieved = [1, 3, 4, 6, 8]
    relevant = [0, 2, 5, 7, 9]
    
    metrics = evaluate_retrieval(retrieved, relevant, k_values=[3, 5])
    for metric, value in metrics.items():
        print(f"{metric:<25} {value:.3f}")
    
    # Example 3: Mixed retrieval
    print("\n📊 EXAMPLE 3: Mixed Retrieval")
    print("-" * 60)
    retrieved = [0, 1, 2, 4, 5, 7]  # 4 relevant z 5
    relevant = [0, 2, 5, 7, 9]
    
    metrics = evaluate_retrieval(retrieved, relevant, k_values=[3, 5])
    for metric, value in metrics.items():
        print(f"{metric:<25} {value:.3f}")