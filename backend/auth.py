from functools import wraps
from flask import request, jsonify
import jwt
from datetime import datetime, timedelta
from models import User

def create_token(user_id, secret_key, expires_hours=24):
    """Tworzy JWT token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=expires_hours),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, secret_key, algorithm='HS256')


def decode_token(token, secret_key):
    """Dekoduje JWT token"""
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def token_required(f):
    """Dekorator wymagający autoryzacji"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Pobierz token z headera
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Brak tokenu autoryzacji'}), 401
        
        # Dekoduj token
        from flask import current_app
        user_id = decode_token(token, current_app.config['SECRET_KEY'])
        
        if not user_id:
            return jsonify({'error': 'Token nieprawidłowy lub wygasł'}), 401
        
        # Pobierz użytkownika
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'Użytkownik nie znaleziony'}), 401
        
        # Przekaż użytkownika do funkcji
        return f(user, *args, **kwargs)
    
    return decorated