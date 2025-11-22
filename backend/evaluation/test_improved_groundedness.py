"""
Test poprawionej wersji groundedness.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.llm_judge import LLMJudge

judge = LLMJudge(model="gpt-4o-mini")

question = "What are the default login credentials for the router?"

context_800 = """
You are required to set the admin account at first login. 
Launch a web browser and type http://tplinkmodem.net. 
Set a strong password using 1-15 characters and click Save.
"""

context_1200 = """
Admin account is used to log in to the modem router's web-based management page.
You are required to set the admin account at first login.
Enter the old password. Enter the new password and enter again to confirm.
[... więcej tekstu ...]
"""

generated = "I don't have this information in the document."

print("\n🧪 TEST: Improved Groundedness Evaluation\n")
print(f"Question: {question}")
print(f"Generated: {generated}\n")

# Test 1: chunk_size=800
print("─"*80)
print("TEST 1: chunk_size=800 context")
print("─"*80)
score_800 = judge._evaluate_groundedness(question, generated, context_800)
print(f"Groundedness: {score_800:.2f}")
print(f"Expected: ~0.0 (context HAS implicit answer)")
print()

# Test 2: chunk_size=1200
print("─"*80)
print("TEST 2: chunk_size=1200 context")
print("─"*80)
score_1200 = judge._evaluate_groundedness(question, generated, context_1200)
print(f"Groundedness: {score_1200:.2f}")
print(f"Expected: ~0.0 (context HAS implicit answer)")
print()

print("─"*80)
print("✅ Both should be ~0.0 now (consistent evaluation)")
print("─"*80)