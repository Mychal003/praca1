from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.pdf_processor import PDFProcessor
from src.query_classifier import QueryClassifier
import os

class RAGPipeline:
    def __init__(self, chunk_size=800, chunk_overlap=100, k=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        
        # Komponenty
        self.pdf_processor = PDFProcessor()
        self.classifier = QueryClassifier()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Vector store (inicjalizowany po upload lub load)
        self.vectorstore = None
        self.retriever = None
        self.num_chunks = 0
    
    def process_document(self, pdf_path: str, save_path: str = None):
        """Przetwarza PDF i tworzy vector store Z CHUNK IDs"""
        # 1. Ekstrakcja tekstu
        text = self.pdf_processor.extract_text(pdf_path)
        
        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        from langchain.schema import Document
        
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        # 4. Embeddingi + FAISS
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # 5. Zapisz liczbę chunków
        self.num_chunks = len(chunks)
        
        # 6. Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )
        
        # 7. NOWE: Zapisz na dysk jeśli podano ścieżkę
        if save_path:
            self.save(save_path)
    
    def save(self, path: str):
        """Zapisuje vector store na dysk"""
        if self.vectorstore:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vectorstore.save_local(path)
    
    def load(self, path: str) -> bool:
        """Ładuje vector store z dysku"""
        try:
            if os.path.exists(path):
                self.vectorstore = FAISS.load_local(
                    path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Wymagane dla FAISS
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": self.k}
                )
                return True
            return False
        except Exception as e:
            print(f"Błąd ładowania vector store: {e}")
            return False
    
    def classify_query(self, query: str) -> str:
        """Klasyfikuje typ pytania"""
        return self.classifier.classify(query)
    
    def _get_prompt_template(self, category: str) -> PromptTemplate:
        """Zwraca template promptu dostosowany do kategorii pytania"""
        
        if category == "factual":
            template = """Use the following context to answer the factual question. Be precise and concise.

INSTRUCTIONS FOR FACTUAL QUESTIONS:
- State the fact directly in 1-2 sentences maximum
- Include only the specific information requested
- Use exact terms and values from the context
- Do NOT add explanations or background information
- Answer based on the context. If the context doesn't contain the exact information, provide what you can find and note what's missing."

Context:
{context}

Question: {question}

Answer:"""
        
        elif category == "procedural":
            template = """Use the following context to provide step-by-step instructions. Be clear and concise.

INSTRUCTIONS FOR PROCEDURAL QUESTIONS:
- List the steps in logical order
- Keep each step brief (one sentence)
- Include only essential details
- Maximum 5-6 steps
- Do NOT add warnings, tips, or extra explanations unless critical
- Answer based on the context. If the context doesn't contain enough information, provide what you can and note what's missing.

Context:
{context}

Question: {question}

Answer:"""
        
        elif category == "troubleshooting":
            template = """Use the following context to diagnose the problem and suggest solutions. Be direct and practical.

INSTRUCTIONS FOR TROUBLESHOOTING QUESTIONS:
- Briefly state what the problem indicates (1 sentence)
- List 2-4 specific solutions from the context
- Keep solutions concise (one sentence each)
- Focus on actionable steps
- Do NOT add general advice not in the context
- Answer based on the context. If the context doesn't contain complete troubleshooting info, provide what you can and note what's missing."

Context:
{context}

Question: {question}

Answer:"""
        
        else:
            template = """Use the following context to answer the question. Be precise and concise.

INSTRUCTIONS:
- Answer directly in 2-3 sentences maximum
- Use ONLY information from the context
- Do NOT add extra details or explanations
- If the information is not in the context, say "I don't have this information in the document"

Context:
{context}

Question: {question}

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> str:
        """Generuje odpowiedź na pytanie z dynamicznym promptem"""
        if self.vectorstore is None:
            raise ValueError("Brak przetworzonego dokumentu")
        
        # 1. Klasyfikuj pytanie
        category = self.classify_query(question)
        
        # 2. Pobierz odpowiedni prompt template
        prompt_template = self._get_prompt_template(category)
        
        # 3. Stwórz QA chain z customowym promptem
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # 4. Wygeneruj odpowiedź
        result = qa_chain({"query": question})
        return result["result"]
    
    def get_sources(self, question: str, k: int = 3) -> list:
        """Zwraca źródłowe chunki dla pytania Z CHUNK IDs"""
        if self.vectorstore is None:
            return []
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
        return [
            {
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "text": doc.page_content,
                "similarity_score": float(score),
                "rank": i
            }
            for i, (doc, score) in enumerate(docs_with_scores)
        ]