import json
import sys
import time
from datetime import datetime
from typing import List, Dict
import re
from collections import Counter
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from evaluation.retrieval_metrics import evaluate_retrieval
load_dotenv()
print("🤖 Ładuję model Semantic Similarity (to może potrwać chwilę przy pierwszym uruchomieniu)...")
SEMANTIC_MODEL = SentenceTransformer('all-mpnet-base-v2')
print("✅ Model załadowany!\n")
# ============================================================================
# METRYKI
# ============================================================================

def rouge_1_f1(prediction: str, reference: str) -> float:
    """
    ROUGE-1 F1: Miara pokrycia pojedynczych słów.
    Najprostsza i najważniejsza metryka do pracy.
    """
    # Tokenizacja
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not pred_words or not ref_words:
        return 0.0
    
    # Overlap
    overlap = len(pred_words & ref_words)
    
    # Precision & Recall
    precision = overlap / len(pred_words)
    recall = overlap / len(ref_words)
    
    # F1
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def token_overlap(prediction: str, reference: str) -> float:
    """
    Prosty % wspólnych słów - łatwa do zrozumienia metryka.
    """
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    overlap = len(pred_words & ref_words)
    return overlap / len(ref_words)

def semantic_similarity(prediction: str, reference: str) -> float:
    """
    Semantic Similarity: Miara podobieństwa semantycznego (0-1).
    
    Używa sentence-transformers do obliczenia jak bardzo dwa teksty
    są semantycznie podobne, niezależnie od użytych słów.
    
    1.0 = semantycznie identyczne
    0.0 = całkowicie różne
    
    Przykład:
        "I don't have this info" vs "Document does not provide this information"
        ROUGE-1: ~0.1 (słabe)
        Semantic: ~0.85 (świetne!)
    """
    if not prediction or not reference:
        return 0.0
    
    try:
        # Embeddingi
        emb1 = SEMANTIC_MODEL.encode(prediction, convert_to_tensor=True)
        emb2 = SEMANTIC_MODEL.encode(reference, convert_to_tensor=True)
        
        # Cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        return max(0.0, min(1.0, similarity))  # Clamp do [0, 1]
    except Exception as e:
        print(f"⚠️  Błąd semantic similarity: {e}")
        return 0.0

TEST_DATASET = [
    # ============================================================================
    # FACTUAL QUESTIONS (fakty o routerze)
    # ============================================================================
    
    {
        "question": "What is the full model name and type of this router?",
        "expected_answer": "The router is TP-Link Archer D7, an AC1750 Wireless Dual Band Gigabit ADSL2+ Modem Router.",
        "category": "factual"
    },
    
    {
        "question": "What is the default web address to access the router interface?",
        "expected_answer": "The default web address is http://tplinkmodem.net or 192.168.1.1.",
        "category": "factual"
    },
    
    {
    "question": "What are the default login credentials for the router?",
    "expected_answer": "The router has no default credentials. You must set your own password when accessing the router for the first time.",
    "category": "factual"
    },
    
    {
        "question": "What port is used to connect the router to the Internet?",
        "expected_answer": "The ADSL port is used to connect the modem router to the Internet via DSL cable to phone jack or splitter.",
        "category": "factual"
    },
    
    {
        "question": "How many operation modes does the Archer D7 support?",
        "expected_answer": "The Archer D7 supports two operation modes: DSL Modem Router Mode and Wireless Router Mode.",
        "category": "factual"
    },
    
    {
        "question": "What USB features does the router support?",
        "expected_answer": "The router supports USB storage device sharing and USB printer sharing through the USB port.",
        "category": "factual"
    },
    
    {
        "question": "What is the WPS button used for on the Archer D7?",
        "expected_answer": "The WPS button is used to quickly establish a secure wireless connection between the router and WPS-enabled devices.",
        "category": "factual"
    },
    
    {
        "question": "What wireless security functions does the router provide?",
        "expected_answer": "The router provides MAC Filtering, Access Control, and IP & MAC Binding for network security.",
        "category": "factual"
    },
    
    # ============================================================================
    # PROCEDURAL QUESTIONS (instrukcje krok po kroku)
    # ============================================================================
    
    {
    "question": "How do you perform a factory reset on the Archer D7?",
    "expected_answer": "With the router powered on, use a pin to press and hold the RESET button on the rear panel for 8 seconds until all LEDs turn on momentarily, then release the button.",
    "category": "procedural"
    },
    
    {
    "question": "How do you access the router's web interface for the first time?",
    "expected_answer": "Visit http://tplinkmodem.net or 192.168.1.1 in a web browser, then set a strong password using 1-15 characters and click Save.",
    "category": "procedural"
    },
    
    {
        "question": "How do you set up the router using Quick Setup Wizard?",
        "expected_answer": "Log in to the router interface, click Quick Setup, select your ISP from the dropdown list, then follow the on-screen instructions to complete the setup.",
        "category": "procedural"
    },
    
    {
        "question": "How do you change the wireless network name and password?",
        "expected_answer": "During Quick Setup or in wireless settings, you can change the preset wireless network name (SSID) and wireless password. After changes, all wireless devices must use the new credentials to connect.",
        "category": "procedural"
    },
    
    {
        "question": "How do you turn on or off the WiFi function on the router?",
        "expected_answer": "Use the WiFi ON/OFF switch button on the router's back panel to turn the WiFi function on or off.",
        "category": "procedural"
    },
    
    {
    "question": "How do you access a USB disk connected to the router via network?",
    "expected_answer": "For Windows, access via \\\\tplinkmodem.net or the default server name ARCHER_D7 in File Explorer. For Mac, access via smb://tplinkmodem.net in Finder. You can customize this name in USB Settings.",
    "category": "procedural"
    },
    
    {
        "question": "How do you set up parental controls on the router?",
        "expected_answer": "Access the Parental Controls section in the router interface, enable the function, then configure what types of websites to block and set access schedules for specific devices.",
        "category": "procedural"
    },
    
    {
        "question": "How do you customize the USB disk server name?",
        "expected_answer": "Log in to router, go to Advanced > USB Settings > Sharing Access, ensure Network Neighborhood is ticked, enter a custom Network/Media Server Name (e.g., MyShare), then click Save.",
        "category": "procedural"
    },
    
    {
    "question": "How do you enable MAC Filtering to control wireless access?",
    "expected_answer": "Go to Advanced > Wireless > MAC Filtering in the router interface, enable the function, then add MAC addresses to either allow or block specific devices from accessing the wireless network.",
    "category": "procedural"
    },
    
    # ============================================================================
    # TROUBLESHOOTING QUESTIONS (rozwiązywanie problemów)
    # ============================================================================
    
    {
        "question": "What should you do if you cannot access the router's web interface?",
        "expected_answer": "Check that your device is connected to the router (wired or wireless), verify you're using the correct address (tplinkmodem.net or 192.168.1.1), try a different web browser, and ensure no firewall is blocking access.",
        "category": "troubleshooting"
    },
    
    {
    "question": "How do you recover access if you forgot the router's login password?",
    "expected_answer": "If you forgot the password, you must reset the router to factory defaults by pressing and holding the RESET button for 8 seconds. After reset, you'll need to set a new password when accessing the router.",
    "category": "troubleshooting"
    },
    
    {
        "question": "What does it mean if the ADSL LED is not lit on the router?",
        "expected_answer": "If the ADSL LED is not lit, it means there's no DSL connection. Check that the DSL cable is properly connected to the phone jack or splitter, verify the cable is not damaged, and confirm DSL service is active with your ISP.",
        "category": "troubleshooting"
    },
    
    {
        "question": "What are the possible causes if wireless devices cannot connect to the network?",
        "expected_answer": "Possible causes include: WiFi is turned off on the router, incorrect wireless password entered, MAC Filtering blocking the device, incompatible wireless standard, or router needs to be restarted.",
        "category": "troubleshooting"
    },
    
    {
        "question": "Why might bandwidth control not work as expected?",
        "expected_answer": "Bandwidth control requires proper configuration of bandwidth rules for each device or IP range. Ensure you've set up rules correctly, assigned appropriate bandwidth limits, and that devices are properly identified in the system.",
        "category": "troubleshooting"
    },
    
    # ============================================================================
    # MORE MIXED QUESTIONS
    # ============================================================================
    
    {
        "question": "What is the purpose of IP & MAC Binding feature?",
        "expected_answer": "IP & MAC Binding (ARP) prevents ARP spoofing and ARP attacks by binding specific IP addresses to specific MAC addresses, ensuring only authorized devices can use those IP addresses on the network.",
        "category": "factual"
    },
    
    {
        "question": "What is Access Control and how does it differ from MAC Filtering?",
        "expected_answer": "Access Control allows blocking or allowing specific devices for both wired and wireless networks, while MAC Filtering works only for wireless network access. Access Control uses blacklist or whitelist approach.",
        "category": "factual"
    },
    
    {
        "question": "How can you remotely access USB storage connected to the router?",
        "expected_answer": "Enable the FTP Server feature in USB Settings. You can then access your USB disk remotely via FTP outside your local network, useful for sharing large files without cloud services.",
        "category": "procedural"
    },
]

# ============================================================================
# EVALUATOR
# ============================================================================

def evaluate_system(rag_pipeline, test_dataset: List[Dict] = None, evaluate_retrieval_metrics: bool = True):
    """
    Główna funkcja ewaluacji - z retrieval metrics!
    
    Args:
        rag_pipeline: Twój RAGPipeline
        test_dataset: Lista pytań
        evaluate_retrieval_metrics: Czy ewaluować retrieval (wymaga annotacji)
    
    Returns:
        Dict z wynikami
    """
    if test_dataset is None:
        test_dataset = TEST_DATASET
    
    print(f"\n{'='*60}")
    print(f"🚀 EWALUACJA - {len(test_dataset)} pytań")
    if evaluate_retrieval_metrics:
        print("   (Including RETRIEVAL METRICS)")
    print(f"{'='*60}\n")
    
    results = []
    
    # Aggregated retrieval metrics
    all_retrieval_metrics = []
    
    for i, item in enumerate(test_dataset, 1):
        print(f"[{i}/{len(test_dataset)}] {item['question'][:50]}...")
        
        start = time.time()
        
        # Generuj odpowiedź
        try:
            generated = rag_pipeline.query(item['question'])
        except Exception as e:
            print(f"  ❌ Błąd: {e}")
            generated = "ERROR"
        
        latency = time.time() - start
        
        # GENERATION METRICS
        rouge1 = rouge_1_f1(generated, item['expected_answer'])
        overlap = token_overlap(generated, item['expected_answer'])
        semantic_sim = semantic_similarity(generated, item['expected_answer'])
        
        result = {
            "question": item['question'],
            "expected": item['expected_answer'],
            "generated": generated,
            "rouge1_f1": rouge1,
            "token_overlap": overlap,
            "semantic_similarity": semantic_sim,
            "latency": latency
        }
        
        # RETRIEVAL METRICS (jeśli dostępne)
        if evaluate_retrieval_metrics and 'relevant_chunk_indices' in item:
            # Pobierz retrieved chunks
            retrieved_sources = rag_pipeline.get_sources(item['question'], k=20)
            retrieved_indices = list(range(len(retrieved_sources)))
            
            relevant_indices = item['relevant_chunk_indices']
            
            # Oblicz retrieval metrics
            ret_metrics = evaluate_retrieval(
                retrieved_indices, 
                relevant_indices,
                k_values=[1, 3, 5, 10]
            )
            
            result['retrieval_metrics'] = ret_metrics
            all_retrieval_metrics.append(ret_metrics)
            
            print(f"  ✓ ROUGE: {rouge1:.3f} | Semantic: {semantic_sim:.3f} | P@5: {ret_metrics['precision@5']:.3f} | R@5: {ret_metrics['recall@5']:.3f} | {latency:.2f}s\n")
        else:
            print(f"  ✓ ROUGE: {rouge1:.3f} | Semantic: {semantic_sim:.3f} | {latency:.2f}s\n")
        
        results.append(result)
    
    # Średnie - Generation
    avg_rouge1 = sum(r['rouge1_f1'] for r in results) / len(results)
    avg_overlap = sum(r['token_overlap'] for r in results) / len(results)
    avg_semantic = sum(r.get('semantic_similarity', 0) for r in results) / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    summary = {
        "avg_rouge1_f1": avg_rouge1,
        "avg_token_overlap": avg_overlap,
        "avg_semantic_similarity": avg_semantic,
        "avg_latency": avg_latency,
        "num_questions": len(results)
    }
    
    # Średnie - Retrieval (jeśli dostępne)
    if all_retrieval_metrics:
        # Oblicz średnią dla każdej metryki
        metric_keys = all_retrieval_metrics[0].keys()
        for key in metric_keys:
            avg_value = sum(m[key] for m in all_retrieval_metrics) / len(all_retrieval_metrics)
            summary[f'avg_{key}'] = avg_value
    
    # Podsumowanie
    print(f"\n{'='*60}")
    print("📊 PODSUMOWANIE")
    print(f"{'='*60}")
    print("\n🎯 GENERATION METRICS:")
    print(f"  ROUGE-1 F1:          {avg_rouge1:.3f}")
    print(f"  Token Overlap:       {avg_overlap:.3f}")
    print(f"  Semantic Similarity: {avg_semantic:.3f}")
    print(f"  Latencja:            {avg_latency:.2f}s")
    
    if all_retrieval_metrics:
        print("\n🔍 RETRIEVAL METRICS:")
        for key, value in summary.items():
            if key.startswith('avg_') and 'retrieval' not in key.lower() and key not in ['avg_rouge1_f1', 'avg_token_overlap', 'avg_semantic_similarity', 'avg_latency']:
                metric_name = key.replace('avg_', '')
                print(f"  {metric_name:<20} {value:.3f}")
    
    print(f"{'='*60}\n")
    
    return {
        "summary": summary,
        "detailed_results": results
    }


# ============================================================================
# EKSPERYMENTY
# ============================================================================

def run_experiments(pdf_path: str):
    """
    Uruchamia 3 podstawowe eksperymenty z różnymi konfiguracjami.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from src.rag_pipeline import RAGPipeline
    
    configs = [
        {"name": "Small chunks", "chunk_size": 500, "k": 3},
        {"name": "Medium chunks", "chunk_size": 800, "k": 3},
        {"name": "Large chunks", "chunk_size": 1200, "k": 3},
    ]
    
    print(f"\n{'='*60}")
    print("🔬 EKSPERYMENTY - 3 konfiguracje")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for config in configs:
        print(f"\n📦 {config['name']}")
        print(f"   chunk_size={config['chunk_size']}, k={config['k']}\n")
        
        # Stwórz pipeline
        pipeline = RAGPipeline(
            chunk_size=config['chunk_size'],
            chunk_overlap=100,
            k=config['k']
        )
        
        # Przetwórz dokument
        pipeline.process_document(pdf_path)
        
        # Ewaluacja
        results = evaluate_system(pipeline)
        
        # Zapisz
        results['config'] = config
        all_results.append(results)
    
    # Porównanie
    print(f"\n{'='*60}")
    print("📊 PORÓWNANIE")
    print(f"{'='*60}")
    print(f"{'Konfiguracja':<20} {'ROUGE-1':<12} {'Overlap':<12} {'Latency'}")
    print("-" * 60)
    
    for res in all_results:
        name = res['config']['name']
        rouge = res['summary']['avg_rouge1_f1']
        overlap = res['summary']['avg_token_overlap']
        latency = res['summary']['avg_latency']
        
        print(f"{name:<20} {rouge:<12.3f} {overlap:<12.3f} {latency:.2f}s")
    
    # Znajdź najlepszy
    best = max(all_results, key=lambda x: x['summary']['avg_rouge1_f1'])
    print(f"\n🏆 Najlepszy: {best['config']['name']}")
    print(f"   ROUGE-1 F1: {best['summary']['avg_rouge1_f1']:.3f}")
    
    # Zapisz do JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Wyniki zapisane: {filename}\n")
    
    return all_results


# ============================================================================
# UŻYCIE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         PROSTY SYSTEM EWALUACJI RAG                      ║
    ╚══════════════════════════════════════════════════════════╝
    
    Użycie:
    
    1. POJEDYNCZA EWALUACJA:
       from evaluate_simple import evaluate_system, TEST_DATASET
       
       pipeline = RAGPipeline()
       pipeline.process_document("doc.pdf")
       
       results = evaluate_system(pipeline, TEST_DATASET)
    
    2. EKSPERYMENTY (3 konfiguracje):
       from evaluate_simple import run_experiments
       
       run_experiments("doc.pdf")
    
    3. CUSTOM PYTANIA:
       custom_dataset = [
           {
               "question": "Twoje pytanie?",
               "expected_answer": "Oczekiwana odpowiedź",
               "category": "factual"
           }
       ]
       
       results = evaluate_system(pipeline, custom_dataset)
    
    ══════════════════════════════════════════════════════════
    """)
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        run_experiments(pdf_path)
    else:
        print("📖 Aby uruchomić eksperymenty: python evaluate_simple.py <pdf_path>")
