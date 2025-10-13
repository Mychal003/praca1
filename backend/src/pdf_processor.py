import pymupdf

class PDFProcessor:
    def extract_text(self, pdf_path: str) -> str:
        """Ekstrakcja tekstu z PDF"""
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_with_metadata(self, pdf_path: str) -> list[dict]:
        """Ekstrakcja z metadanymi stron"""
        doc = pymupdf.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            pages.append({
                "page_number": i + 1,
                "text": page.get_text()
            })
        return pages