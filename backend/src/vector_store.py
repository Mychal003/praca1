from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )
    
    def create_vectorstore(self, text: str) -> FAISS:
        chunks = self.text_splitter.split_text(text)
        return FAISS.from_texts(chunks, self.embeddings)
    
    def search(self, vectorstore: FAISS, query: str, k: int = 3):
        return vectorstore.similarity_search(query, k=k)