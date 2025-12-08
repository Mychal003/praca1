from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
from src.rag_pipeline import RAGPipeline
from src.conversation_manager import ConversationManager
from models import db, User, Conversation, Message
from auth import token_required, create_token
from config import Config
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
app.config.from_object(Config)
CORS(app, supports_credentials=True)

# Inicjalizacja bazy danych
db.init_app(app)

# Słownik przechowujący pipeline dla każdej sesji użytkownika
# Klucz: user_id, Wartość: {conversation_id: pipeline}
user_pipelines = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


def get_user_pipeline(user_id, conversation_id):
    """Pobiera pipeline dla użytkownika i konwersacji"""
    if user_id not in user_pipelines:
        return None
    return user_pipelines[user_id].get(conversation_id)


def set_user_pipeline(user_id, conversation_id, pipeline):
    """Zapisuje pipeline dla użytkownika i konwersacji"""
    if user_id not in user_pipelines:
        user_pipelines[user_id] = {}
    user_pipelines[user_id][conversation_id] = pipeline

def get_vectorstore_path(user_id, conversation_id):
    """Zwraca ścieżkę do zapisanego vector store"""
    return os.path.join('vectorstores', f'user_{user_id}', f'conv_{conversation_id}')

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Rejestracja nowego użytkownika"""
    data = request.get_json()
    
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    
    # Walidacja
    if not username or not email or not password:
        return jsonify({'error': 'Wszystkie pola są wymagane'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Hasło musi mieć minimum 6 znaków'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Nazwa użytkownika jest już zajęta'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email jest już zarejestrowany'}), 400
    
    # Tworzenie użytkownika
    user = User(username=username, email=email)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    # Generuj token
    token = create_token(user.id, app.config['SECRET_KEY'])
    
    logger.info(f"✅ Zarejestrowano użytkownika: {username}")
    
    return jsonify({
        'message': 'Rejestracja pomyślna',
        'token': token,
        'user': user.to_dict()
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Logowanie użytkownika"""
    data = request.get_json()
    
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'error': 'Podaj login i hasło'}), 400
    
    # Szukaj po username lub email
    user = User.query.filter(
        (User.username == username) | (User.email == username)
    ).first()
    
    if not user or not user.check_password(password):
        return jsonify({'error': 'Nieprawidłowy login lub hasło'}), 401
    
    # Generuj token
    token = create_token(user.id, app.config['SECRET_KEY'])
    
    logger.info(f"✅ Zalogowano użytkownika: {username}")
    
    return jsonify({
        'message': 'Logowanie pomyślne',
        'token': token,
        'user': user.to_dict()
    }), 200


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(user):
    """Pobiera dane zalogowanego użytkownika"""
    return jsonify({'user': user.to_dict()}), 200


# ============================================================================
# CONVERSATION ENDPOINTS
# ============================================================================

@app.route('/api/conversations', methods=['GET'])
@token_required
def get_conversations(user):
    """Pobiera listę konwersacji użytkownika"""
    conversations = ConversationManager.get_user_conversations(user.id)
    return jsonify({
        'conversations': [c.to_dict() for c in conversations]
    }), 200


@app.route('/api/conversations', methods=['POST'])
@token_required
def create_conversation(user):
    """Tworzy nową konwersację"""
    conversation = ConversationManager.create_conversation(user.id)
    logger.info(f"📝 Utworzono konwersację {conversation.id} dla użytkownika {user.id}")
    return jsonify({
        'conversation': conversation.to_dict()
    }), 201


@app.route('/api/conversations/<int:conversation_id>', methods=['GET'])
@token_required
def get_conversation(user, conversation_id):
    """Pobiera szczegóły konwersacji z wiadomościami"""
    conversation = ConversationManager.get_conversation(conversation_id, user.id)
    
    if not conversation:
        return jsonify({'error': 'Konwersacja nie znaleziona'}), 404
    
    return jsonify({
        'conversation': conversation.to_dict(include_messages=True)
    }), 200


@app.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
@token_required
def delete_conversation(user, conversation_id):
    """Usuwa konwersację"""
    if ConversationManager.delete_conversation(conversation_id, user.id):
        # Usuń też pipeline
        if user.id in user_pipelines and conversation_id in user_pipelines[user.id]:
            del user_pipelines[user.id][conversation_id]
        return jsonify({'message': 'Konwersacja usunięta'}), 200
    return jsonify({'error': 'Konwersacja nie znaleziona'}), 404


# ============================================================================
# DOCUMENT UPLOAD (zmodyfikowany)
# ============================================================================

@app.route('/api/conversations/<int:conversation_id>/upload', methods=['POST'])
@token_required
def upload_file(user, conversation_id):
    """Upload PDF do konwersacji"""
    
    # Sprawdź czy konwersacja należy do użytkownika
    conversation = ConversationManager.get_conversation(conversation_id, user.id)
    if not conversation:
        return jsonify({'error': 'Konwersacja nie znaleziona'}), 404
    
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Nieprawidłowy format pliku. Dozwolone: PDF'}), 400
    
    filename = secure_filename(file.filename)
    unique_filename = f"{user.id}_{conversation_id}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)
        logger.info(f"💾 Plik zapisany: {filepath}")
        
        start_time = datetime.now()
        
        # Stwórz pipeline dla tej konwersacji
        pipeline = RAGPipeline()
        
        # NOWE: Ścieżka do zapisu vector store
        vectorstore_path = get_vectorstore_path(user.id, conversation_id)
        
        # Przetwórz dokument I zapisz vector store na dysk
        pipeline.process_document(filepath, save_path=vectorstore_path)
        
        # Zapisz pipeline w pamięci
        set_user_pipeline(user.id, conversation_id, pipeline)
        
        # Zaktualizuj konwersację
        conversation.document_name = filename
        conversation.title = filename.rsplit('.', 1)[0][:50]  # Usuń .pdf i ogranicz do 50 znaków
        db.session.commit()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Dokument przetworzony i zapisany dla konwersacji {conversation_id}")
        
        return jsonify({
            'message': 'Dokument przetworzony pomyślnie',
            'filename': filename,
            'processing_time': round(processing_time, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Błąd: {str(e)}", exc_info=True)
        return jsonify({'error': f'Błąd serwera: {str(e)}'}), 500

# ============================================================================
# QUERY ENDPOINT (zmodyfikowany)
# ============================================================================

@app.route('/api/conversations/<int:conversation_id>/query', methods=['POST'])
@token_required
def query(user, conversation_id):
    """Zadaj pytanie w kontekście konwersacji"""
    
    # Sprawdź konwersację
    conversation = ConversationManager.get_conversation(conversation_id, user.id)
    if not conversation:
        return jsonify({'error': 'Konwersacja nie znaleziona'}), 404
    
    # Pobierz pipeline z pamięci
    pipeline = get_user_pipeline(user.id, conversation_id)
    
    # NOWE: Jeśli nie ma w pamięci, spróbuj załadować z dysku
    if not pipeline:
        vectorstore_path = get_vectorstore_path(user.id, conversation_id)
        pipeline = RAGPipeline()
        
        if pipeline.load(vectorstore_path):
            # Udało się załadować - zapisz w pamięci
            set_user_pipeline(user.id, conversation_id, pipeline)
            logger.info(f"📂 Załadowano vector store z dysku dla konwersacji {conversation_id}")
        else:
            return jsonify({'error': 'Najpierw wgraj dokument do tej konwersacji'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Brak pytania'}), 400
    
    logger.info(f"💬 Pytanie w konwersacji {conversation_id}: {question[:100]}...")
    
    try:
        start_time = datetime.now()
        
        # Zapisz pytanie użytkownika
        ConversationManager.add_message(conversation_id, 'user', question)
        
        # Klasyfikuj i generuj odpowiedź
        category = pipeline.classify_query(question)
        answer = pipeline.query(question)
        sources = pipeline.get_sources(question, k=3)
        
        # Zapisz odpowiedź asystenta
        ConversationManager.add_message(
            conversation_id, 
            'assistant', 
            answer,
            category=category,
            sources=sources
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'answer': answer,
            'category': category,
            'sources': sources,
            'latency': round(latency, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Błąd: {str(e)}", exc_info=True)
        return jsonify({'error': f'Błąd generowania odpowiedzi: {str(e)}'}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Plik jest zbyt duży. Maksymalny rozmiar: 16MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Wewnętrzny błąd serwera'}), 500


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Tworzenie tabel w bazie danych
    with app.app_context():
        db.create_all()
        logger.info("✅ Baza danych zainicjalizowana")
    
    logger.info("="*60)
    logger.info("🚀 Uruchamianie RAG Documentation Assistant")
    logger.info("="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')