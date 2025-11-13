"""
Szybki test Semantic Similarity - porównanie z ROUGE
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.evaluate_simple import rouge_1_f1, token_overlap, semantic_similarity

# Przykłady gdzie ROUGE jest niski, ale semantic similarity powinien być wysoki
test_cases = [
    {
        "name": "Synonimiczne sformułowanie",
        "expected": "The document does not provide specific default login credentials.",
        "generated": "I don't have this information in the document.",
    },
    {
        "name": "Różne słowa, ten sam sens",
        "expected": "The router supports USB storage device sharing and USB printer sharing.",
        "generated": "It supports USB Storage Sharing, Print Server, FTP Server and Media Server.",
    },
    {
        "name": "Identyczne odpowiedzi",
        "expected": "The default web address is http://tplinkmodem.net or 192.168.1.1.",
        "generated": "The default web address is http://tplinkmodem.net or http://192.168.1.1.",
    },
    {
        "name": "Całkowicie różne",
        "expected": "The router has two operation modes.",
        "generated": "The USB port supports external flash drives.",
    }
]

print("\n" + "="*70)
print("🧪 TEST SEMANTIC SIMILARITY vs ROUGE")
print("="*70 + "\n")

for i, case in enumerate(test_cases, 1):
    print(f"{i}. {case['name']}")
    print(f"   Expected:  {case['expected'][:60]}...")
    print(f"   Generated: {case['generated'][:60]}...")
    
    rouge = rouge_1_f1(case['generated'], case['expected'])
    overlap = token_overlap(case['generated'], case['expected'])
    semantic = semantic_similarity(case['generated'], case['expected'])
    
    print(f"\n   📊 Metryki:")
    print(f"      ROUGE-1:    {rouge:.3f}")
    print(f"      Overlap:    {overlap:.3f}")
    print(f"      Semantic:   {semantic:.3f}")
    
    # Interpretacja
    if semantic > 0.75 and rouge < 0.5:
        print(f"      💡 INSIGHT: Semantic similarity pokazuje że odpowiedzi są podobne,")
        print(f"                  mimo niskiego ROUGE (różne słowa, ten sam sens)")
    elif semantic < 0.5:
        print(f"      ⚠️  Odpowiedzi są semantycznie różne")
    
    print()

print("="*70 + "\n")