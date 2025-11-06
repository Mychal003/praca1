from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.pdf_processor import PDFProcessor
from src.query_classifier import QueryClassifier
import os

class RAGPipeline:
    def __init__(self, chunk_size=1200, chunk_overlap=100, k=7):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        
        # Komponenty
        self.pdf_processor = PDFProcessor()
        self.classifier = QueryClassifier()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Vector store (inicjalizowany po upload)
        self.vectorstore = None
        self.qa_chain = None
    
    def process_document(self, pdf_path: str):
        """Przetwarza PDF i tworzy vector store"""
        # 1. Ekstrakcja tekstu
        text = self.pdf_processor.extract_text(pdf_path)
        
        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # 3. Embeddingi + FAISS
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
        
        # 4. Retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" = wszystkie chunki w jednym promptcie
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            ),
            return_source_documents=True
        )
    
    def classify_query(self, query: str) -> str:
        """Klasyfikuje typ pytania"""
        return self.classifier.classify(query)
    
    def query(self, question: str) -> str:
        """Generuje odpowiedź na pytanie"""
        if self.qa_chain is None:
            raise ValueError("Brak przetworzonego dokumentu")
        
        category = self.classify_query(question)
        custom_prompt = self._get_prompt_for_category(category)
        
        result = self.qa_chain({"query": question})
        return result["result"]
    
    def get_sources(self, question: str, k: int = 3) -> list:
        """Zwraca źródłowe chunki dla pytania"""
        if self.vectorstore is None:
            return []
        
        docs = self.vectorstore.similarity_search(question, k=k)
        return [{"text": doc.page_content, "score": i} for i, doc in enumerate(docs)]
    
    def _get_prompt_for_category(self, category: str) -> str:
        """Customowy prompt w zależności od kategorii"""
        prompts = {
            "factual": "Odpowiedz krótko i precyzyjnie:",
            "procedural": "Podaj instrukcję krok po kroku:",
            "troubleshooting": "Zdiagnozuj problem i zaproponuj rozwiązanie:"
        }
        return prompts.get(category, "Odpowiedz na pytanie:")