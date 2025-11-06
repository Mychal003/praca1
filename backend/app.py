from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# KONFIGURACJA LOGGINGU
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Globalna instancja pipeline (w produkcji: sesje użytkownika)
pipeline = None


def allowed_file(filename):
    """Sprawdza czy plik ma dozwolone rozszerzenie."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Endpoint: Upload PDF i stwórz vector store.
    
    Request:
        - file: PDF file (multipart/form-data)
    
    Response:
        - 200: {'message': str, 'filename': str, 'processing_time': float}
        - 400: {'error': str}
        - 500: {'error': str}
    """
    global pipeline
    
    logger.info("📤 Otrzymano żądanie uploadu")
    
    # Walidacja: czy plik jest w requestcie
    if 'file' not in request.files:
        logger.warning("❌ Brak pliku w requeście")
        return jsonify({'error': 'Brak pliku'}), 400
    
    file = request.files['file']
    
    # Walidacja: czy wybrano plik
    if file.filename == '':
        logger.warning("❌ Pusta nazwa pliku")
        return jsonify({'error': 'Nie wybrano pliku'}), 400
    
    # Walidacja: czy dozwolone rozszerzenie
    if not allowed_file(file.filename):
        logger.warning(f"❌ Nieprawidłowe rozszerzenie: {file.filename}")
        return jsonify({'error': 'Nieprawidłowy format pliku. Dozwolone: PDF'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Zapisz plik
            file.save(filepath)
            logger.info(f"💾 Plik zapisany: {filepath}")
            
            # Przetwórz PDF
            start_time = datetime.now()
            logger.info("⚙️ Rozpoczynam przetwarzanie dokumentu...")
            
            pipeline = RAGPipeline()
            pipeline.process_document(filepath)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Dokument przetworzony pomyślnie w {processing_time:.2f}s")
            
            return jsonify({
                'message': 'Dokument przetworzony pomyślnie',
                'filename': filename,
                'processing_time': round(processing_time, 2)
            }), 200
            
        except FileNotFoundError as e:
            logger.error(f"❌ Plik nie znaleziony: {str(e)}")
            return jsonify({'error': f'Plik nie znaleziony: {str(e)}'}), 404
            
        except ValueError as e:
            logger.error(f"❌ Błąd walidacji: {str(e)}")
            return jsonify({'error': f'Błąd przetwarzania: {str(e)}'}), 400
            
        except Exception as e:
            logger.error(f"❌ Nieoczekiwany błąd: {str(e)}", exc_info=True)
            return jsonify({'error': f'Błąd serwera: {str(e)}'}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """
    Endpoint: Zadaj pytanie do dokumentu.
    
    Request JSON:
        {'question': str}
    
    Response:
        - 200: {'answer': str, 'category': str, 'sources': list, 'latency': float}
        - 400: {'error': str}
        - 500: {'error': str}
    """
    global pipeline
    
    # Walidacja: czy dokument został załadowany
    if pipeline is None:
        logger.warning("❌ Próba query bez załadowanego dokumentu")
        return jsonify({'error': 'Najpierw wgraj dokument'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    # Walidacja: czy pytanie nie jest puste
    if not question:
        logger.warning("❌ Puste pytanie")
        return jsonify({'error': 'Brak pytania'}), 400
    
    logger.info(f"💬 Pytanie: {question[:100]}...")
    
    try:
        start_time = datetime.now()
        
        # Klasyfikuj pytanie
        category = pipeline.classify_query(question)
        logger.info(f"📂 Kategoria: {category}")
        
        # Generuj odpowiedź
        answer = pipeline.query(question)
        
        # Pobierz źródła
        sources = pipeline.get_sources(question, k=3)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Odpowiedź wygenerowana w {latency:.2f}s")
        
        return jsonify({
            'answer': answer,
            'category': category,
            'sources': sources,
            'latency': round(latency, 2)
        }), 200
        
    except ValueError as e:
        logger.error(f"❌ Błąd walidacji: {str(e)}")
        return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"❌ Błąd generowania odpowiedzi: {str(e)}", exc_info=True)
        return jsonify({'error': f'Błąd generowania odpowiedzi: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """
    Endpoint: Health check.
    
    Response:
        {'status': str, 'document_loaded': bool}
    """
    return jsonify({
        'status': 'ok',
        'document_loaded': pipeline is not None
    }), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    """
    Endpoint: Statystyki systemu (NOWY).
    
    Response:
        {'chunk_size': int, 'k': int, 'vectorstore_size': int}
    """
    if pipeline is None:
        return jsonify({'error': 'Najpierw wgraj dokument'}), 400
    
    try:
        # Pobierz statystyki
        vectorstore_size = pipeline.vectorstore.index.ntotal if pipeline.vectorstore else 0
        
        return jsonify({
            'chunk_size': pipeline.chunk_size,
            'chunk_overlap': pipeline.chunk_overlap,
            'k': pipeline.k,
            'vectorstore_size': vectorstore_size,
            'model': 'gpt-3.5-turbo',
            'embedding_model': 'text-embedding-3-small'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Błąd pobierania statystyk: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handler dla zbyt dużych plików."""
    logger.warning("❌ Plik zbyt duży (>16MB)")
    return jsonify({'error': 'Plik jest zbyt duży. Maksymalny rozmiar: 16MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handler dla błędów serwera."""
    logger.error(f"❌ Błąd serwera: {str(error)}", exc_info=True)
    return jsonify({'error': 'Wewnętrzny błąd serwera'}), 500


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    # Utwórz folder uploads jeśli nie istnieje
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    logger.info("="*60)
    logger.info("🚀 Uruchamianie RAG Documentation Assistant")
    logger.info("="*60)
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"📏 Max file size: 16MB")
    logger.info(f"🌐 Server: http://localhost:5000")
    logger.info("="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')