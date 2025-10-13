from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from src.rag_pipeline import RAGPipeline

app = Flask(__name__)
CORS(app)  # Pozwól na requesty z frontendu

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Globalna instancja pipeline (w produkcji: sesje użytkownika)
pipeline = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Endpoint: Upload PDF i stwórz vector store"""
    global pipeline
    
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Przetwórz PDF
            pipeline = RAGPipeline()
            pipeline.process_document(filepath)
            
            return jsonify({
                'message': 'Dokument przetworzony pomyślnie',
                'filename': filename
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Nieprawidłowy format pliku'}), 400

@app.route('/api/query', methods=['POST'])
def query():
    """Endpoint: Zadaj pytanie do dokumentu"""
    global pipeline
    
    if pipeline is None:
        return jsonify({'error': 'Najpierw wgraj dokument'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Brak pytania'}), 400
    
    try:
        # Opcjonalnie: klasyfikuj pytanie
        category = pipeline.classify_query(question)
        
        # Generuj odpowiedź
        answer = pipeline.query(question)
        
        # Opcjonalnie: pobierz źródła
        sources = pipeline.get_sources(question, k=3)
        
        return jsonify({
            'answer': answer,
            'category': category,
            'sources': sources
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint: Sprawdź czy serwer działa"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)