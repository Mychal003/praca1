from langchain_ollama import ChatOllama

class QueryClassifier:
    """
    Klasyfikator pytań użytkownika na podstawie ich typu/intencji.
    Wykorzystuje LLM do zero-shot classification.
    """
    
    CATEGORIES = {
        "factual": "Pytanie o konkretny fakt z dokumentacji",
        "procedural": "Pytanie o instrukcję/jak coś zrobić",
        "troubleshooting": "Pytanie o rozwiązanie problemu"
    }
    
    def __init__(self, model: str = "qwen2.5:7b-instruct"):
        """
        Inicjalizacja klasyfikatora.
        
        Args:
            model: Nazwa modelu Ollama do użycia
        """
        self.llm = ChatOllama(model=model, temperature=0)
    
    def classify(self, query: str) -> str:
        """
        Klasyfikuje pytanie użytkownika do jednej z kategorii.
        
        Args:
            query: Pytanie użytkownika
            
        Returns:
            Nazwa kategorii: "factual", "procedural", lub "troubleshooting"
        """
        prompt = f"""Sklasyfikuj poniższe pytanie użytkownika do jednej z trzech kategorii:

Kategorie:
- factual: pytanie o konkretny fakt, wartość, parametr (np. "Jaka jest maksymalna temperatura?")
- procedural: pytanie o instrukcję, jak coś zrobić krok po kroku (np. "Jak zresetować urządzenie?")
- troubleshooting: pytanie o rozwiązanie problemu lub błędu (np. "Dlaczego urządzenie się nie włącza?")

Pytanie: "{query}"

Odpowiedz TYLKO nazwą kategorii (factual, procedural lub troubleshooting), bez dodatkowych wyjaśnień."""

        try:
            response = self.llm.predict(prompt).strip().lower()
            
            # Walidacja
            if response not in self.CATEGORIES:
                return "factual"
            
            return response
        except Exception as e:
            print(f"Błąd klasyfikacji: {e}")
            return "factual"
    
    def get_category_description(self, category: str) -> str:
        """Zwraca opis kategorii"""
        return self.CATEGORIES.get(category, "Nieznana kategoria")